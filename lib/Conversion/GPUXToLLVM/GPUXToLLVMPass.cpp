//===- GPUToLLVMPass.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a pass to convert GPUX dialect ops into LLVM IR dialect
/// operations via sequence of GPU runtime wrapper API calls from a library.
/// This wrapper library exports API calls on top of GPU runtimes such as
/// Level-zero and SYCL.
///
//===----------------------------------------------------------------------===//
#include "imex/Conversion/GPUXToLLVM/GPUXToLLVMPass.h"
#include "imex/Dialect/GPUX/IR/GPUXOps.h"

#include "imex/Utils/FuncUtils.hpp"
#include "imex/Utils/GPUSerialize.h"
#include "imex/Utils/TypeConversion.hpp"

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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

namespace {

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

/// This class defines function call builders to call into GPU runtime wrapper
/// APIs.
//  These runtime API can be Level-zero or SYCL that would perform converstion
//  to LLVM dialect ops.
/// This class gets extended further for GPUX dialect ops like Alloc, dealloc,
/// createStream, destroystream, LaunchFunc to perform op specific conversion.
template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern
    : public mlir::ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<OpTy>(converter) {}

protected:
  mlir::MLIRContext *context = &this->getTypeConverter()->getContext();

  mlir::Type llvmVoidType = mlir::LLVM::LLVMVoidType::get(context);
  mlir::Type llvmPointerType = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type llvmInt8Type = mlir::IntegerType::get(context, 8);
  mlir::Type llvmInt32Type = mlir::IntegerType::get(context, 32);
  mlir::Type llvmInt64Type = mlir::IntegerType::get(context, 64);
  mlir::Type llvmIndexType = mlir::IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));

  mlir::Type llvmRangeType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmIndexType});
  mlir::Type llvmRangePointerType = mlir::LLVM::LLVMPointerType::get(context);

  FunctionCallBuilder moduleLoadCallBuilder = {
      "gpuModuleLoad",
      llvmPointerType /* void *module */,
      {
          llvmPointerType, /* void *stream */
          llvmPointerType, /* void *spirv*/
          llvmIndexType    /* size*/
      }};

  FunctionCallBuilder moduleUnloadCallBuilder = {
      "gpuModuleUnload",
      llvmVoidType,
      {
          llvmPointerType /* void *module */
      }};

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
          llvmInt32Type,       /* unsigned int sharedMemBytes */
          llvmRangePointerType /* Params */
      }};

  FunctionCallBuilder streamCreateCallBuilder = {
      "gpuCreateStream",
      llvmPointerType, /* void *stream */
      {
          llvmPointerType, /* void *device */
          llvmPointerType  /* void *context */
      }};

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
          llvmIndexType,   /* intptr_t size */
          llvmIndexType,   /* intptr_t alignment */
          llvmInt32Type    /* unsigned int shared */
      }};

  FunctionCallBuilder deallocCallBuilder = {
      "gpuMemFree",
      llvmVoidType,
      {
          llvmPointerType, /* void *stream */
          llvmPointerType  /* void *ptr */
      }};

  FunctionCallBuilder memcpyCallBuilder = {
      "gpuMemCopy",
      llvmVoidType,
      {
          llvmPointerType, /* void *stream */
          llvmPointerType, /* void *ptr dst */
          llvmPointerType, /* void *ptr src */
          llvmIndexType    /* intptr_t size */
      }};
};

/// A rewrite pattern to convert gpux.memcpy operations into a GPU runtime
/// call.
class ConvertMemcpyOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<imex::gpux::MemcpyOp> {
public:
  ConvertMemcpyOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<imex::gpux::MemcpyOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(imex::gpux::MemcpyOp memcpyOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto loc = memcpyOp.getLoc();
    mlir::MemRefType srcMemRefType =
        llvm::dyn_cast<mlir::MemRefType>(memcpyOp.getSrc().getType());
    mlir::MemRefDescriptor srcDesc(adaptor.getSrc());

    // Compute number of elements.
    mlir::Value numElements = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, getIndexType(), rewriter.getIndexAttr(1));
    for (int pos = 0; pos < srcMemRefType.getRank(); ++pos) {
      auto size = srcDesc.size(rewriter, loc, pos);
      numElements = rewriter.create<mlir::LLVM::MulOp>(loc, numElements, size);
    }

    // Get element size.
    auto sizeInBytes =
        getSizeInBytes(loc, srcMemRefType.getElementType(), rewriter);
    // Compute total.
    mlir::Value totalSize =
        rewriter.create<mlir::LLVM::MulOp>(loc, numElements, sizeInBytes);

    mlir::Type elementType =
        typeConverter->convertType(srcMemRefType.getElementType());

    mlir::Value srcBasePtr = srcDesc.alignedPtr(rewriter, loc);
    mlir::Value srcOffset = srcDesc.offset(rewriter, loc);
    mlir::Value srcPtr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, srcBasePtr.getType(), elementType, srcBasePtr, srcOffset);
    mlir::MemRefDescriptor dstDesc(adaptor.getDst());
    mlir::Value dstBasePtr = dstDesc.alignedPtr(rewriter, loc);
    mlir::Value dstOffset = dstDesc.offset(rewriter, loc);
    mlir::Value dstPtr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, dstBasePtr.getType(), elementType, dstBasePtr, dstOffset);

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    memcpyCallBuilder.create(
        loc, rewriter, {adaptor.getGpuxStream(), dstPtr, srcPtr, totalSize});
    rewriter.eraseOp(memcpyOp);
    return mlir::success();
  }
};

/// A rewrite pattern to convert gpux.alloc operations into a GPU runtime
/// call.
class ConvertAllocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<imex::gpux::AllocOp> {
public:
  ConvertAllocOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<imex::gpux::AllocOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(imex::gpux::AllocOp allocOp, OpAdaptor adaptor,
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

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    mlir::Value allocatedPtr =
        allocCallBuilder
            .create(loc, rewriter,
                    {adaptor.getGpuxStream(), sizeBytes, alignmentVar, typeVar})
            ->getResult(0);

    // Create the MemRef descriptor.
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

/// A rewrite pattern to convert gpux.dealloc operations into a GPU runtime
/// call.
class ConvertDeallocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<imex::gpux::DeallocOp> {
public:
  ConvertDeallocOpToGpuRuntimeCallPattern(
      mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<imex::gpux::DeallocOp>(typeConverter) {
  }

private:
  mlir::LogicalResult
  matchAndRewrite(imex::gpux::DeallocOp deallocOp, OpAdaptor adaptor,
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

/// A rewrite pattern to convert gpux.launch_func operations into a sequence of
/// GPU runtime calls.
/// In essence, a gpux.launch_func operations gets compiled into the following
/// sequence of runtime calls:
///
/// * moduleLoad        -- loads the module given the spirv data
/// * KernelGetFunction -- gets a handle to the actual kernel function
/// * launchKernel      -- launches the kernel on a stream
/// * gpuWait           -- waits for operations on the stream to finish
///
/// Intermediate data structures are allocated on the stack.
class ConvertLaunchFuncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<imex::gpux::LaunchFuncOp> {
public:
  ConvertLaunchFuncOpToGpuRuntimeCallPattern(
      mlir::LLVMTypeConverter &typeConverter,
      mlir::StringRef gpuBinaryAnnotation)
      : ConvertOpToGpuRuntimeCallPattern<imex::gpux::LaunchFuncOp>(
            typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation) {}

private:
  llvm::SmallString<32> gpuBinaryAnnotation;

  // Generates an LLVM IR dialect global that contains the name of the given
  // kernel function as a C string, and returns a pointer to its beginning.
  // The code is essentially:
  //
  // llvm.global constant @kernel_name("function_name\00")
  // func(...) {
  //   %0 = llvm.addressof @kernel_name
  //   %1 = llvm.constant (0 : index)
  //   %2 = llvm.getelementptr %0[%1, %1] : !llvm<"i8*">
  // }
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
  matchAndRewrite(imex::gpux::LaunchFuncOp launchOp, OpAdaptor adaptor,
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

    auto spirvBlob = binaryAttr.getValue();
    mlir::SmallString<128> nameBuffer(kernelModule.getName());
    nameBuffer.append(kGpuBinaryStorageSuffix);
    mlir::Value data = mlir::LLVM::createGlobalString(
        loc, rewriter, nameBuffer.str(), binaryAttr.getValue(),
        mlir::LLVM::Linkage::Internal);

    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType,
        mlir::IntegerAttr::get(llvmIndexType,
                               static_cast<int64_t>(spirvBlob.size())));

    // loads the GPU module given the spirv data
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

    /////////////////////////////////////////////////////////////////////////
    // Create an array of struct containing all kernel parameters and inserts
    // these type-erased pointers to the fields of the struct. The array of
    // struct is then passed to Runtime wrapper Kernel launch call. Generated
    // code is essentially as follows:
    //
    // 1. %struct = alloca(sizeof(struct { Parameters... }))
    // 2. %array = alloca(NumParameters * sizeof(void *))
    // 3.  for (i : [0, NumParameters))
    //    %fieldPtr = llvm.getelementptr %struct[0, i]
    // 4. llvm.store parameters[i], %fieldPtr
    //    %elementPtr = llvm.getelementptr %array[i]
    // 5. llvm.store %fieldPtr, %elementPtr

    imex::AllocaInsertionPoint allocaHelper(launchOp);
    auto kernelParams = adaptor.getKernelOperands();
    auto paramsCount = static_cast<unsigned>(kernelParams.size());
    auto paramsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmRangeType, paramsCount + 1);

    auto getKernelParamType = [&](unsigned i) -> mlir::Type {
      if (mlir::isa<mlir::MemRefType>(
              launchOp.getKernelOperands()[i].getType())) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
        return desc.getElementPtrType();
      }

      return kernelParams[i].getType();
    };

    llvm::SmallVector<mlir::Value> paramsStorage(paramsCount);
    auto paramsArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      for (auto i : llvm::seq(0u, paramsCount)) {
        paramsStorage[i] = rewriter.create<mlir::LLVM::AllocaOp>(
            loc, llvmPointerType, getKernelParamType(i), size, 0);
      }
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmPointerType,
                                                   paramsArrayType, size, 0);
    });

    mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(1));
    auto computeTypeSize = [&](mlir::Type type) -> mlir::Value {
      auto nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, type);
      auto gep = rewriter.create<mlir::LLVM::GEPOp>(loc, type, llvmPointerType,
                                                    nullPtr, one);
      return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, llvmIndexType, gep);
    };

    auto getKernelParam =
        [&](unsigned i) -> std::pair<mlir::Value, mlir::Value> {
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(
          launchOp.getKernelOperands()[i].getType());
      auto paramType = paramsStorage[i].getType();
      if (memrefType) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
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

    auto nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmPointerType);
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

    /////////////////////////////////////////////////////////////////////

    // Get pointer to array of kernel parameters and call launch kernel
    auto paramsArrayVoidPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, llvmRangePointerType, paramsArrayPtr);
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(0));
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

    // waits for operations on the stream to finish

    waitCallBuilder.create(loc, rewriter, adaptor.getGpuxStream());
    rewriter.eraseOp(launchOp);
    return mlir::success();
  }
};

class RemoveGPUModulePattern
    : public mlir::ConvertOpToLLVMPattern<mlir::gpu::GPUModuleOp> {
public:
  RemoveGPUModulePattern(mlir::LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<mlir::gpu::GPUModuleOp>(converter) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::GPUModuleOp op,
                  mlir::gpu::GPUModuleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// A rewrite pattern to convert gpux.create_stream operations into a GPU
/// runtime call.
class ConvertGpuStreamCreatePattern
    : public ConvertOpToGpuRuntimeCallPattern<imex::gpux::CreateStreamOp> {
public:
  ConvertGpuStreamCreatePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<imex::gpux::CreateStreamOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(imex::gpux::CreateStreamOp op,
                  imex::gpux::CreateStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto loc = op.getLoc();

    // TODO: Pass nullptrs now for the current workflow where user is
    // not passing device and context. Add different streambuilders
    // later.
    auto device = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmPointerType);
    auto context = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmPointerType);
    auto res = streamCreateCallBuilder.create(loc, rewriter, {device, context});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

/// A rewrite pattern to convert gpux.destroy_stream operations into a GPU
/// runtime call.
class ConvertGpuStreamDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<imex::gpux::DestroyStreamOp> {
public:
  ConvertGpuStreamDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<imex::gpux::DestroyStreamOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(imex::gpux::DestroyStreamOp op,
                  imex::gpux::DestroyStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res =
        streamDestroyCallBuilder.create(loc, rewriter, adaptor.getGpuxStream());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};
} // namespace

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

  mlir::populateGpuToLLVMConversionPatterns(converter, patterns);

  imex::populateGpuxToLLVMPatternsAndLegality(converter, patterns, target);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

void imex::populateGpuxToLLVMPatternsAndLegality(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  auto context = patterns.getContext();
  auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(context);
  converter.addConversion(
      [llvmPointerType](imex::gpux::OpaqueType) -> mlir::Type {
        return llvmPointerType;
      });
  converter.addConversion(
      [llvmPointerType](imex::gpux::StreamType) -> mlir::Type {
        return llvmPointerType;
      });
  converter.addConversion(
      [llvmPointerType](imex::gpux::DeviceType) -> mlir::Type {
        return llvmPointerType;
      });
  converter.addConversion(
      [llvmPointerType](imex::gpux::ContextType) -> mlir::Type {
        return llvmPointerType;
      });

  patterns.insert<
      // clang-format off
      ConvertGpuStreamCreatePattern,
      ConvertGpuStreamDestroyPattern,
      ConvertAllocOpToGpuRuntimeCallPattern,
      ConvertDeallocOpToGpuRuntimeCallPattern,
      RemoveGPUModulePattern,
      ConvertMemcpyOpToGpuRuntimeCallPattern
      // clang-format on
      >(converter);

  patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
      converter, imex::gpuBinaryAttrName);

  target.addIllegalDialect<mlir::gpu::GPUDialect>();
  target.addIllegalDialect<imex::gpux::GPUXDialect>();
}

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
imex::createConvertGPUXToLLVMPass() {
  return std::make_unique<GPUXToLLVMPass>();
}
