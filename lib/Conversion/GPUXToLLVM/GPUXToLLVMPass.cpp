//===- GPUToSPIRVPass.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file extends upstream GPUToSPIRV Pass that converts GPU ops to SPIR-V
/// by adding more conversion patterns like SCF, math and control flow. This
/// pass only converts gpu.func ops inside gpu.module op.
///
//===----------------------------------------------------------------------===//
#include "imex/Conversion/GPUXToLLVM/GPUXToLLVMPass.h"
#include "imex/Dialect/GPUX/IR/GPUXOps.h"

#include "imex/Transforms/FuncUtils.hpp"
#include "imex/Transforms/TypeConversion.hpp"

#include "../PassDetail.h"

// TODO: remove
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>

namespace imex {

static constexpr const char *kGpuBinaryStorageSuffix = "_spirv_binary";

struct FunctionCallBuilder {
  FunctionCallBuilder(mlir::StringRef functionName, mlir::Type returnType,
                      mlir::ArrayRef<mlir::Type> argumentTypes)
      : functionName(functionName),
        functionType(
            mlir::LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  mlir::LLVM::CallOp create(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::ArrayRef<mlir::Value> arguments) const {
    auto module =
        builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
    auto function = [&] {
      if (auto function =
              module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(functionName))
        return function;
      return mlir::OpBuilder::atBlockEnd(module.getBody())
          .create<mlir::LLVM::LLVMFuncOp>(loc, functionName, functionType);
    }();
    return builder.create<mlir::LLVM::CallOp>(loc, function, arguments);
  }

private:
  mlir::StringRef functionName;
  mlir::LLVM::LLVMFunctionType functionType;
};

template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern
    : public mlir::ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<OpTy>(converter) {}

protected:
  mlir::Value getNumElements(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc, mlir::MemRefType type,
                             mlir::MemRefDescriptor desc) const {
    return type.hasStaticShape()
               ? mlir::ConvertToLLVMPattern::createIndexConstant(
                     rewriter, loc, type.getNumElements())
               // For identity maps (verified by caller), the
               // number of elements is stride[0] * size[0].
               : rewriter.create<mlir::LLVM::MulOp>(
                     loc, desc.stride(rewriter, loc, 0),
                     desc.size(rewriter, loc, 0));
  }

  mlir::MLIRContext *context = &this->getTypeConverter()->getContext();

  mlir::Type llvmVoidType = mlir::LLVM::LLVMVoidType::get(context);
  mlir::Type llvmPointerType =
      mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
  mlir::Type llvmPointerPointerType =
      mlir::LLVM::LLVMPointerType::get(llvmPointerType);
  mlir::Type llvmInt8Type = mlir::IntegerType::get(context, 8);
  mlir::Type llvmInt32Type = mlir::IntegerType::get(context, 32);
  mlir::Type llvmInt64Type = mlir::IntegerType::get(context, 64);
  mlir::Type llvmIndexType = mlir::IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));

  //// ----
  mlir::Type llvmI32PtrType = mlir::LLVM::LLVMPointerType::get(llvmIndexType);

  mlir::Type llvmRangeType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmIndexType});
  mlir::Type llvmRangePointerType =
      mlir::LLVM::LLVMPointerType::get(llvmRangeType);
  mlir::Type llvmAllocResType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmPointerType, llvmPointerType});
  mlir::Type llvmAllocResPtrType =
      mlir::LLVM::LLVMPointerType::get(llvmAllocResType);
  //// ----

  FunctionCallBuilder moduleLoadCallBuilder = {
      "gpuModuleLoad",
      llvmPointerType /* void *module */,
      {llvmPointerType, /* void *stream */
       llvmPointerType, /* void *spirv*/
       llvmIndexType /* size*/}};
  FunctionCallBuilder moduleUnloadCallBuilder = {
      "gpuModuleUnload",
      llvmVoidType,
      {
          llvmPointerType /* void *module */
      }};

  // same as kernelGetCallBuilder
  FunctionCallBuilder kernelGetCallBuilder = {
      "gpuKernelGet",
      llvmPointerType /* void *function */,
      {
          llvmPointerType, /* void* stream */
          llvmPointerType, /* void *module */
          llvmPointerType  /* char *name   */
      }};
  FunctionCallBuilder launchKernelCallBuilder = {
      "gpuLaunchKernel",
      llvmVoidType,
      {
          llvmPointerType,     /* void* stream */
          llvmPointerType,     /* void* func */
          llvmIndexType,       /* intptr_t gridXDim */
          llvmIndexType,       /* intptr_t gridyDim */
          llvmIndexType,       /* intptr_t gridZDim */
          llvmIndexType,       /* intptr_t blockXDim */
          llvmIndexType,       /* intptr_t blockYDim */
          llvmIndexType,       /* intptr_t blockZDim */
          llvmInt64Type,       /* unsigned int sharedMemBytes */
          llvmRangePointerType /* Params */
      }};

  FunctionCallBuilder streamCreateCallBuilder = {
      "gpuCreateStream",
      llvmPointerType, /* void *stream */
      {}};

  FunctionCallBuilder streamDestroyCallBuilder = {
      "gpuStreamDestroy",
      llvmVoidType,
      {
          llvmPointerType /* void *stream */
      }};

  FunctionCallBuilder waitCallBuilder = {"gpuWait",
                                         llvmVoidType,
                                         {
                                             llvmPointerType /* void *stream */
                                         }};

  FunctionCallBuilder allocCallBuilder = {
      "gpuMemAlloc",
      llvmPointerType /* void * */,
      {
          llvmPointerType, /* void *stream */
          llvmIndexType,   // size
          llvmIndexType,   // alignment
          llvmInt32Type    // shared
      }};

  FunctionCallBuilder deallocCallBuilder = {
      "gpuMemFree",
      llvmVoidType,
      {
          llvmPointerType, /* void *stream */
          llvmPointerType  /* void *ptr */
      }};
};

/// A rewrite pattern to convert gpux.alloc operations into a GPU runtime
/// call.
class ConvertAllocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::AllocOp> {
public:
  ConvertAllocOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::AllocOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::AllocOp allocOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!allocOp.getSymbolOperands().empty())
      return mlir::failure();

    mlir::MemRefType memRefType = allocOp.getType();
    auto converter = getTypeConverter();
    auto dstType = converter->convertType(memRefType);
    if (!dstType)
      return mlir::failure();

    bool isShared = allocOp.getHostShared();

    auto loc = allocOp.getLoc();

    // Get shape of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.
    mlir::SmallVector<mlir::Value, 4> shape;
    mlir::SmallVector<mlir::Value, 4> strides;
    mlir::Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, shape, strides, sizeBytes);

    assert(shape.size() == strides.size());

    auto alignment = rewriter.getIntegerAttr(llvmIndexType, 64);
    auto alignmentVar =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmIndexType, alignment);

    auto typeVar = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(isShared));

    mlir::Value allocatedPtr =
        allocCallBuilder
            .create(loc, rewriter,
                    {adaptor.getGpuxStream(), sizeBytes, alignmentVar, typeVar})
            ->getResult(0);

    auto memrefDesc = mlir::MemRefDescriptor::undef(rewriter, loc, dstType);
    auto elemPtrTye = memrefDesc.getElementPtrType();
    memrefDesc.setAllocatedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, allocatedPtr));
    memrefDesc.setAlignedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, allocatedPtr));

    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));

    memrefDesc.setOffset(rewriter, loc, zero);
    for (auto i : llvm::seq(0u, static_cast<unsigned>(shape.size()))) {
      memrefDesc.setSize(rewriter, loc, i, shape[i]);
      memrefDesc.setStride(rewriter, loc, i, strides[i]);
    }

    mlir::Value resMemref = memrefDesc;
    rewriter.replaceOp(allocOp, resMemref);

    return mlir::success();
  }
};

/// A rewrite pattern to convert gpu.dealloc operations into a GPU runtime
/// call.
class ConvertDeallocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::DeallocOp> {
public:
  ConvertDeallocOpToGpuRuntimeCallPattern(
      mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::DeallocOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::DeallocOp deallocOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = deallocOp.getLoc();

    mlir::Value pointer =
        mlir::MemRefDescriptor(adaptor.getMemref()).allocatedPtr(rewriter, loc);
    auto casted =
        rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerType, pointer);
    auto res = deallocCallBuilder.create(loc, rewriter,
                                         {adaptor.getGpuxStream(), casted});
    rewriter.replaceOp(deallocOp, res.getResults());
    return mlir::success();
  }
};

mlir::Value getStream(mlir::OpBuilder &builder) {
  auto func =
      builder.getBlock()->getParent()->getParentOfType<mlir::func::FuncOp>();

  if (!func)
    return {};

  if (!llvm::hasSingleElement(func.getBody()))
    return {};

  auto &block = func.getBody().front();
  auto ops = block.getOps<imex::gpux::CreateStreamOp>();
  if (!ops.empty())
    return (*ops.begin()).getResult();
  else
    return {};
}

/// A rewrite patter to convert gpu.launch_func operations into a sequence of
/// GPU runtime calls.
/// In essence, a gpu.launch_func operations gets compiled into the following
/// sequence of runtime calls:
///
/// * moduleLoad        -- loads the module given the spirv data
/// * moduleGetFunction -- gets a handle to the actual kernel function
/// * getStreamHelper   -- initializes a new compute stream on GPU
/// * launchKernel      -- launches the kernel on a stream
/// * streamSynchronize -- waits for operations on the stream to finish
///
/// Intermediate data structures are allocated on the stack.
class ConvertLaunchFuncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::LaunchFuncOp> {
public:
  ConvertLaunchFuncOpToGpuRuntimeCallPattern(
      mlir::LLVMTypeConverter &typeConverter,
      mlir::StringRef gpuBinaryAnnotation)
      : ConvertOpToGpuRuntimeCallPattern<gpux::LaunchFuncOp>(typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation) {}

private:
  llvm::SmallString<32> gpuBinaryAnnotation;

  mlir::Value generateKernelNameConstant(mlir::StringRef moduleName,
                                         mlir::StringRef name,
                                         mlir::Location loc,
                                         mlir::OpBuilder &builder) const {
    // Make sure the trailing zero is included in the constant.
    std::vector<char> kernelName(name.begin(), name.end());
    kernelName.push_back('\0');

    std::string globalName =
        std::string(llvm::formatv("{0}_{1}_kernel_name", moduleName, name));
    return mlir::LLVM::createGlobalString(
        loc, builder, globalName,
        mlir::StringRef(kernelName.data(), kernelName.size()),
        mlir::LLVM::Linkage::Internal);
  }

  mlir::LogicalResult
  matchAndRewrite(gpux::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = launchOp.getLoc();

    // Create an LLVM global with SPIRV extracted from the kernel annotation
    // and obtain a pointer to the first byte in it.
    auto kernelModule =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::gpu::GPUModuleOp>(
            launchOp, launchOp.getKernelModuleName());
    assert(kernelModule && "expected a kernel module");

    auto binaryAttr =
        kernelModule->getAttrOfType<mlir::StringAttr>(gpuBinaryAnnotation);
    if (!binaryAttr) {
      kernelModule.emitOpError()
          << "missing " << gpuBinaryAnnotation << " attribute";
      return mlir::failure();
    }

    auto blob = binaryAttr.getValue();
    mlir::SmallString<128> nameBuffer(kernelModule.getName());
    nameBuffer.append(kGpuBinaryStorageSuffix);
    mlir::Value data = mlir::LLVM::createGlobalString(
        loc, rewriter, nameBuffer.str(), binaryAttr.getValue(),
        mlir::LLVM::Linkage::Internal);

    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType,
        mlir::IntegerAttr::get(llvmIndexType,
                               static_cast<int64_t>(blob.size())));

    auto module = moduleLoadCallBuilder.create(
        loc, rewriter, {adaptor.getGpuxStream(), data, size});
    // Get the function from the module. The name corresponds to the name of
    // the kernel function.
    auto kernelName = generateKernelNameConstant(
        launchOp.getKernelModuleName().getValue(),
        launchOp.getKernelName().getValue(), loc, rewriter);
    auto function = kernelGetCallBuilder.create(
        loc, rewriter,
        {adaptor.getGpuxStream(), module->getResult(0), kernelName});
    // Create array of pointers to kernel arguments.

    AllocaInsertionPoint allocaHelper(launchOp);
    auto kernelParams = adaptor.getKernelOperands();
    auto paramsCount = static_cast<unsigned>(kernelParams.size());
    auto paramsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmRangeType, paramsCount + 1);
    auto paramsArrayPtrType = mlir::LLVM::LLVMPointerType::get(paramsArrayType);

    auto getKernelParamType = [&](unsigned i) -> mlir::Type {
      if (launchOp.getOperands()[i].getType().isa<mlir::MemRefType>()) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
        return desc.getElementPtrType();
      }

      return kernelParams[i].getType();
    };

    llvm::SmallVector<mlir::Value> paramsStorage(paramsCount);
    auto paramsArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(paramsCount));
      for (auto i : llvm::seq(0u, paramsCount)) {
        auto ptrType = mlir::LLVM::LLVMPointerType::get(getKernelParamType(i));
        paramsStorage[i] =
            rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, size, 0);
      }
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, paramsArrayPtrType,
                                                   size, 0);
    });

    mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(1));
    auto localMemStorageClass = rewriter.getI64IntegerAttr(
        mlir::gpu::GPUDialect::getPrivateAddressSpace());
    auto computeTypeSize = [&](mlir::Type type) -> mlir::Value {
      // %Size = getelementptr %T* null, int 1
      // %SizeI = ptrtoint %T* %Size to i32
      auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, type);
      auto gep = rewriter.create<mlir::LLVM::GEPOp>(loc, type, nullPtr, one);
      return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, llvmIndexType, gep);
    };

    auto getKernelParam =
        [&](unsigned i) -> std::pair<mlir::Value, mlir::Value> {
      auto memrefType = launchOp.getKernelOperands()[i]
                            .getType()
                            .dyn_cast<mlir::MemRefType>();
      auto paramType = paramsStorage[i].getType();
      if (memrefType) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
        if (memrefType.getMemorySpace() == localMemStorageClass) {
          auto rank = static_cast<unsigned>(memrefType.getRank());
          auto typeSize = std::max(memrefType.getElementTypeBitWidth(), 8u) / 8;
          mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(
              loc, llvmIndexType,
              rewriter.getIntegerAttr(llvmIndexType, typeSize));
          for (auto i : llvm::seq(0u, rank)) {
            auto dim = desc.size(rewriter, loc, i);
            size = rewriter.create<mlir::LLVM::MulOp>(loc, llvmIndexType, size,
                                                      dim);
          }
          auto null = rewriter.create<mlir::LLVM::NullOp>(
              loc, desc.getElementPtrType());
          return {size, null};
        }
        auto size = computeTypeSize(paramType);
        return {size, desc.alignedPtr(rewriter, loc)};
      }

      auto size = computeTypeSize(paramType);
      return {size, kernelParams[i]};
    };

    mlir::Value paramsArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, paramsArrayType);

    for (auto i : llvm::seq(0u, paramsCount)) {
      auto param = getKernelParam(i);
      rewriter.create<mlir::LLVM::StoreOp>(loc, param.second, paramsStorage[i]);
      auto ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerType,
                                                        paramsStorage[i]);

      auto typeSize = param.first;

      mlir::Value range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, llvmRangeType);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, ptr, 0);
      range =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, typeSize, 1);

      paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, paramsArray,
                                                               range, i);
    }
    auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    auto nullRange = [&]() {
      auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));
      mlir::Value range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, llvmRangeType);
      range =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, nullPtr, 0);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, zero, 1);
      return range;
    }();
    paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, paramsArray, nullRange, paramsCount);
    rewriter.create<mlir::LLVM::StoreOp>(loc, paramsArray, paramsArrayPtr);

    auto paramsArrayVoidPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, llvmRangePointerType, paramsArrayPtr);
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));
    mlir::Value dynamicSharedMemorySize =
        adaptor.getDynamicSharedMemorySize()
            ? adaptor.getDynamicSharedMemorySize()
            : zero;
    launchKernelCallBuilder.create(
        loc, rewriter,
        {adaptor.getGpuxStream(), function->getResult(0),
         adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ(),
         adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
         adaptor.getBlockSizeZ(), dynamicSharedMemorySize, paramsArrayVoidPtr});

    waitCallBuilder.create(loc, rewriter, adaptor.getGpuxStream());
    rewriter.eraseOp(launchOp);
    return mlir::success();
  }
}; // namespace imex

class ConvertGpuStreamCreatePattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::CreateStreamOp> {
public:
  ConvertGpuStreamCreatePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::CreateStreamOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::CreateStreamOp op,
                  gpux::CreateStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto loc = op.getLoc();

    auto res = streamCreateCallBuilder.create(loc, rewriter, {});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuStreamDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::DestroyStreamOp> {
public:
  ConvertGpuStreamDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::DestroyStreamOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::DestroyStreamOp op,
                  gpux::DestroyStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res =
        streamDestroyCallBuilder.create(loc, rewriter, adaptor.getGpuxStream());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class GPUXToLLVMPass : public ::imex::ConvertGPUXToLLVMBase<GPUXToLLVMPass> {
public:
  explicit GPUXToLLVMPass() {}
  void runOnOperation() override;
};

void GPUXToLLVMPass::runOnOperation() {
  mlir::MLIRContext &context = getContext();
  mlir::LLVMTypeConverter converter(&context);
  mlir::RewritePatternSet patterns(&context);
  mlir::LLVMConversionTarget target(context);

  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  mlir::populateVectorToLLVMConversionPatterns(converter, patterns);
  mlir::populateMemRefToLLVMConversionPatterns(converter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
  mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                          target);

  mlir::populateGpuToLLVMConversionPatterns(
      converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

  populateControlFlowTypeConversionRewritesAndTarget(converter, patterns,
                                                     target);

  populateGpuxToLLVMPatternsAndLegality(converter, patterns, target);

  auto mod = getOperation();
  if (mlir::failed(
          mlir::applyPartialConversion(mod, target, std::move(patterns))))
    signalPassFailure();
}

void populateGpuxToLLVMPatternsAndLegality(mlir::LLVMTypeConverter &converter,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::ConversionTarget &target) {
  auto context = patterns.getContext();
  auto llvmPointerType =
      mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
  converter.addConversion([llvmPointerType](gpux::OpaqueType) -> mlir::Type {
    return llvmPointerType;
  });
  converter.addConversion([llvmPointerType](gpux::StreamType) -> mlir::Type {
    return llvmPointerType;
  });
  converter.addConversion([llvmPointerType](gpux::DeviceType) -> mlir::Type {
    return llvmPointerType;
  });
  converter.addConversion([llvmPointerType](gpux::ContextType) -> mlir::Type {
    return llvmPointerType;
  });

  // mlir::populateGpuToLLVMConversionPatterns(
  //    converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

  patterns.insert<
      // clang-format off
    ConvertGpuStreamCreatePattern,
    ConvertGpuStreamDestroyPattern,
    ConvertAllocOpToGpuRuntimeCallPattern,
    ConvertDeallocOpToGpuRuntimeCallPattern
      // clang-format on
      >(converter);

  patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      converter, mlir::gpu::getDefaultGpuBinaryAnnotation());

  target.addIllegalDialect<mlir::gpu::GPUDialect>();
  target.addIllegalDialect<gpux::GPUXDialect>();
}

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToLLVMPass() {
  return std::make_unique<GPUXToLLVMPass>();
}

} // namespace imex
