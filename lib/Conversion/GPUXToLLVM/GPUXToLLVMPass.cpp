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

namespace imex {

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

static constexpr llvm::StringLiteral kEventCountAttrName("gpu.event_count");
static constexpr llvm::StringLiteral kEventIndexAttrName("gpu.event_index");

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
  mlir::Type llvmIntPtrType = mlir::IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));

  //// ----
  mlir::Type llvmI32PtrType = mlir::LLVM::LLVMPointerType::get(llvmIntPtrType);

  mlir::Type llvmRangeType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmIntPtrType});
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
      {
          llvmPointerType /* void *cubin */
      }};
  FunctionCallBuilder moduleUnloadCallBuilder = {
      "gpuModuleUnload",
      llvmVoidType,
      {
          llvmPointerType /* void *module */
      }};

  // same as kernelGetCallBuilder
  FunctionCallBuilder moduleGetFunctionCallBuilder = {
      "gpuModuleGetFunction",
      llvmPointerType /* void *function */,
      {
          llvmPointerType, /* void *module */
          llvmPointerType  /* char *name   */
      }};
  FunctionCallBuilder launchKernelCallBuilder = {
      "gpuLaunchKernel",
      llvmVoidType,
      {
          llvmPointerType,        /* void* stream */
          llvmPointerType,        /* void* f */
          llvmIntPtrType,         /* intptr_t gridXDim */
          llvmIntPtrType,         /* intptr_t gridyDim */
          llvmIntPtrType,         /* intptr_t gridZDim */
          llvmIntPtrType,         /* intptr_t blockXDim */
          llvmIntPtrType,         /* intptr_t blockYDim */
          llvmIntPtrType,         /* intptr_t blockZDim */
          llvmInt32Type,          /* unsigned int sharedMemBytes */
          llvmPointerType,        /* void *hstream */
          llvmPointerPointerType, /* void **kernelParams */
          llvmPointerPointerType  /* void **extra */
      }};

  FunctionCallBuilder streamCreateCallBuilder = {
      "GpuStreamCreate",
      llvmPointerType, /* void *stream */
      {
          llvmPointerType, /* void *device */
          llvmPointerType, /* void *context */
          llvmIntPtrType   /* intptr_t eventsCount */
      }};

  FunctionCallBuilder streamDestroyCallBuilder = {
      "GpuStreamDestroy",
      llvmVoidType,
      {
          llvmPointerType /* void *stream */
      }};

  FunctionCallBuilder DeviceCreateCallBuilder = {
      "GpuDeviceCreate",
      llvmPointerType, /* void *device */
      {}};

  FunctionCallBuilder DeviceDestroyCallBuilder = {
      "GpuDeviceDestroy",
      llvmVoidType,
      {
          llvmPointerType /* void *device */
      }};

  FunctionCallBuilder ContextCreateCallBuilder = {
      "GpuContextCreate",
      llvmPointerType, /* void *context */
      {}};

  FunctionCallBuilder ContextDestroyCallBuilder = {
      "GpuContextDestroy",
      llvmVoidType,
      {
          llvmPointerType /* void *context */
      }};

  FunctionCallBuilder kernelGetCallBuilder = {
      "GpuKernelGet",
      llvmPointerType, /* void *kernel */
      {
          llvmPointerType, /* void *module */
          llvmPointerType, /* char *name   */
      }};

  FunctionCallBuilder kernelDestroyCallBuilder = {
      "GpuKernelDestroy",
      llvmVoidType,
      {
          llvmPointerType /* void *kernel */
      }};

  FunctionCallBuilder waitEventCallBuilder = {
      "GpuWait",
      llvmVoidType,
      {
          llvmPointerType /* void *stream */
      }};

  FunctionCallBuilder allocCallBuilder = {
      "GpuAlloc",
      llvmPointerType /* void * */,
      {
          llvmPointerType, /* void *stream */
          llvmIntPtrType,  /* intptr_t sizeBytes */
      }};

  FunctionCallBuilder deallocCallBuilder = {
      "GpuDeAlloc",
      llvmVoidType,
      {
          llvmPointerType, /* void *stream */
          llvmPointerType  /* void *ptr */
      }};

  FunctionCallBuilder memcpyCallBuilder = {
      "gpuMemcpy",
      llvmVoidType,
      {llvmPointerType /* void *dst */, llvmPointerType /* void *src */,
       llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder memsetCallBuilder = {
      "gpuMemset32",
      llvmVoidType,
      {llvmPointerType /* void *dst */, llvmInt32Type /* unsigned int value */,
       llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
};
/// A rewrite pattern to convert gpux.alloc operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertAllocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::AllocOp> {
public:
  ConvertAllocOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::AllocOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::AllocOp allocOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.dealloc operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertDeallocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::DeallocOp> {
public:
  ConvertDeallocOpToGpuRuntimeCallPattern(
      mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::DeallocOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::DeallocOp deallocOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.wait operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertWaitOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::WaitOp> {
public:
  ConvertWaitOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::WaitOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::WaitOp waitOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpux.wait async operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertWaitAsyncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::WaitOp> {
public:
  ConvertWaitAsyncOpToGpuRuntimeCallPattern(
      mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::WaitOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::WaitOp waitOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite patter to convert gpu.launch_func operations into a sequence of
/// GPU runtime calls. Currently it supports CUDA and ROCm (HIP).
///
/// In essence, a gpu.launch_func operations gets compiled into the following
/// sequence of runtime calls:
///
/// * moduleLoad        -- loads the module given the cubin / hsaco data
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
      mlir::StringRef gpuBinaryAnnotation, bool kernelBarePtrCallConv)
      : ConvertOpToGpuRuntimeCallPattern<gpux::LaunchFuncOp>(typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation),
        kernelBarePtrCallConv(kernelBarePtrCallConv) {}

private:
  mlir::Value generateParamsArray(gpux::LaunchFuncOp launchOp,
                                  OpAdaptor adaptor,
                                  mlir::OpBuilder &builder) const;
  mlir::Value generateKernelNameConstant(mlir::StringRef moduleName,
                                         mlir::StringRef name,
                                         mlir::Location loc,
                                         mlir::OpBuilder &builder) const;

  mlir::LogicalResult
  matchAndRewrite(gpux::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  llvm::SmallString<32> gpuBinaryAnnotation;
  bool kernelBarePtrCallConv;
};

/// A rewrite pattern to convert gpu.memcpy operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertMemcpyOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::MemcpyOp> {
public:
  ConvertMemcpyOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::MemcpyOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::MemcpyOp memcpyOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

/// A rewrite pattern to convert gpu.memset operations into a GPU runtime
/// call. Currently it supports CUDA and ROCm (HIP).
class ConvertMemsetOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::MemsetOp> {
public:
  ConvertMemsetOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::MemsetOp>(typeConverter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::MemsetOp memsetOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ConvertGpuStreamCreatePattern
    : public ConvertOpToGpuRuntimeCallPattern<gpux::CreateStreamOp> {
public:
  ConvertGpuStreamCreatePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpux::CreateStreamOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpux::CreateStreamOp op,
                  gpux::CreateStreamOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto eventsCount =
        mlir::getConstantIntValue(mod->getAttr(kEventCountAttrName));
    if (!eventsCount)
      return mlir::failure();

    auto loc = op.getLoc();
    // auto eventsCountVar =
    //     rewriter
    //         .create<mlir::LLVM::ConstantOp>(
    //             loc, llvmIntPtrType,
    //             rewriter.getIntegerAttr(llvmIntPtrType, *eventsCount))
    //         .getResult();
    // auto res = streamCreateCallBuilder.create(loc, rewriter, eventsCountVar);
    // rewriter.replaceOp(op, res.getResult());
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
    // auto res =
    //     streamDestroyCallBuilder.create(loc, rewriter, adaptor.getSrc());
    // rewriter.replaceOp(op, res.getResult());
    return mlir::success();
  }
};

static std::string getUniqueLLVMGlobalName(mlir::ModuleOp mod,
                                           mlir::StringRef srcName) {
  auto globals = mod.getOps<mlir::LLVM::GlobalOp>();
  for (int i = 0;; ++i) {
    auto name =
        (i == 0 ? std::string(srcName) : (srcName + llvm::Twine(i)).str());
    auto isSameName = [&](mlir::LLVM::GlobalOp global) {
      return global.getName() == name;
    };
    if (llvm::find_if(globals, isSameName) == globals.end())
      return name;
  }
}

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

  // patterns.insert<
  //     // clang-format off
  //   ConvertGpuStreamCreatePattern,
  //   ConvertGpuStreamDestroyPattern,
  //   ConvertGpuModuleLoadPattern,
  //   ConvertGpuModuleDestroyPattern,
  //   ConvertGpuKernelGetPattern,
  //   ConvertGpuKernelDestroyPattern,
  //   ConvertGpuKernelLaunchPattern,
  //   ConvertGpuAllocPattern,
  //   ConvertGpuDeAllocPattern
  //     // clang-format on
  //     >(converter);

  target.addIllegalDialect<mlir::gpu::GPUDialect>();
  target.addIllegalDialect<gpux::GPUXDialect>();
}

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToLLVMPass() {
  return std::make_unique<GPUXToLLVMPass>();
}

} // namespace imex
