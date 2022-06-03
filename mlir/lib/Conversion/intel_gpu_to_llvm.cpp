// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir-extensions/Conversion/intel_gpu_to_llvm.hpp"

#include "mlir-extensions/Dialect/intel_gpu/IR/intel_gpu_ops.hpp"

#include "mlir-extensions/Transforms/func_utils.hpp"

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>

using namespace mlir;

static constexpr const char *kGpuBinaryStorageSuffix = "_spirv_binary";

namespace {

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

  mlir::StringRef functionName;
  mlir::LLVM::LLVMFunctionType functionType;
};

template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern : public ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<OpTy>(typeConverter) {}

protected:
  Value getNumElements(ConversionPatternRewriter &rewriter, Location loc,
                       MemRefType type, MemRefDescriptor desc) const {
    return type.hasStaticShape()
               ? ConvertToLLVMPattern::createIndexConstant(
                     rewriter, loc, type.getNumElements())
               // For identity maps (verified by caller), the number of
               // elements is stride[0] * size[0].
               : rewriter.create<LLVM::MulOp>(loc,
                                              desc.stride(rewriter, loc, 0),
                                              desc.size(rewriter, loc, 0));
  }

  MLIRContext *context = &this->getTypeConverter()->getContext();

  Type llvmVoidType = LLVM::LLVMVoidType::get(context);
  Type llvmPointerType =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  Type llvmPointerPointerType = LLVM::LLVMPointerType::get(llvmPointerType);
  Type llvmInt8Type = IntegerType::get(context, 8);
  Type llvmInt32Type = IntegerType::get(context, 32);
  Type llvmInt64Type = IntegerType::get(context, 64);
  Type llvmIntPtrType = IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));

  FunctionCallBuilder moduleLoadCallBuilder = {
      "iGpuModuleLoad",
      llvmPointerType /* void *module */,
      {llvmPointerType, /* void *stream */
       llvmPointerType, /* void *cubin*/
       llvmIntPtrType /* size*/}};
  FunctionCallBuilder moduleUnloadCallBuilder = {
      "iGpuModuleDestroy", llvmVoidType, {llvmPointerType /* void *module */}};
  FunctionCallBuilder moduleGetFunctionCallBuilder = {
      "iGpuKernelGet",
      llvmPointerType /* void *function */,
      {
          llvmPointerType, /* void *stream */
          llvmPointerType, /* void *module */
          llvmPointerType  /* char *name   */
      }};
  FunctionCallBuilder launchKernelCallBuilder = {
      "iGpuLaunchKernel",
      llvmVoidType,
      {
          llvmPointerType,        /* void *stream */
          llvmPointerType,        /* void* f */
          llvmIntPtrType,         /* intptr_t gridXDim */
          llvmIntPtrType,         /* intptr_t gridyDim */
          llvmIntPtrType,         /* intptr_t gridZDim */
          llvmIntPtrType,         /* intptr_t blockXDim */
          llvmIntPtrType,         /* intptr_t blockYDim */
          llvmIntPtrType,         /* intptr_t blockZDim */
          llvmInt32Type,          /* unsigned int sharedMemBytes */
          llvmPointerPointerType, /* void **kernelParams */
          llvmPointerPointerType  /* void **extra */
      }};
  FunctionCallBuilder streamDestroyCallBuilder = {
      "iGpuStreamDestroy", llvmVoidType, {llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamSynchronizeCallBuilder = {
      "iGpuStreamSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *stream */}};
  FunctionCallBuilder allocCallBuilder = {
      "iGpuMemAlloc",
      llvmPointerType /* void * */,
      {llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder deallocCallBuilder = {
      "iGpuMemFree",
      llvmVoidType,
      {llvmPointerType /* void *ptr */, llvmPointerType /* void *stream */}};
  FunctionCallBuilder deviceCallBuilder = {
      "iGpuGetDevice", llvmPointerType /* void * */, {}};
  FunctionCallBuilder contextCreateCallBuilder = {
      "iGpuCreateContext",
      llvmPointerType /* void * */,
      {llvmPointerType /* void* device */}};
  FunctionCallBuilder streamCallBuilder = {
      "iGpuCreateStream",
      llvmPointerType /* void * */,
      {llvmPointerType /* void* stream_ptr */}};
  FunctionCallBuilder streamCreateCallBuilder = {
      "gpuStreamCreate", llvmPointerType /* void *stream */, {}};
  FunctionCallBuilder eventDestroyCallBuilder = {
      "iGpuEventDestroy", llvmVoidType, {llvmPointerType /* void *event */}};
  FunctionCallBuilder eventSynchronizeCallBuilder = {
      "iGpuEventSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *event */}};
  FunctionCallBuilder eventCreateCallBuilder = {
      "iGpuEventCreate", llvmPointerType /* void *event */, {}};
  FunctionCallBuilder eventRecordCallBuilder = {
      "iGpuEventRecord",
      llvmVoidType,
      {llvmPointerType /* void *event */, llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamWaitEventCallBuilder = {
      "iGpuStreamWaitEvent",
      llvmVoidType,
      {llvmPointerType /* void *stream */, llvmPointerType /* void *event */}};
};

/// Checks if all the operands of the op being lowered are of LLVM Types. The
/// types are expected to be converted by the `LLVMTypeConverter` before the op
/// is actually lowered. If the type of an operands is not already converted it
/// hints a missing typeConversion and failure is returned in that case.
static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                     ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      })) {
    return rewriter.notifyMatchFailure(
        op, "cannot convert if operands aren't of LLVM type.");
  }

  return success();
}

static LogicalResult
isAsyncWithOneDependency(ConversionPatternRewriter &rewriter,
                         gpu::AsyncOpInterface op) {
  if (op.getAsyncDependencies().size() != 1)
    return rewriter.notifyMatchFailure(
        op, "Can only convert with exactly one async dependency.");

  if (!op.getAsyncToken())
    return rewriter.notifyMatchFailure(op, "Can convert only async version.");

  return success();
}

// Returns whether `value` is the result of an LLVM::CallOp to `functionName`.
static bool isDefinedByCallTo(Value value, StringRef functionName) {
  assert(value.getType().isa<LLVM::LLVMPointerType>());
  if (auto defOp = value.getDefiningOp<LLVM::CallOp>())
    return defOp.getCallee()->equals(functionName);
  return false;
}

mlir::Value getStream(mlir::OpBuilder &builder) {
  auto module =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();

  Value stream;
  module.walk([&](mlir::LLVM::CallOp op) {
    if (op.getCallee().getValue() == "iGpuCreateStream") {
      stream = op.getResult(0);
    }
  });
  return stream;
}

mlir::Value getGpuStream(mlir::OpBuilder &builder, mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return {};

  if (!llvm::hasSingleElement(func.getBody()))
    return {};

  auto &block = func.getBody().front();
  auto ops = block.getOps<intel_gpu::GetStreamOp>();
  if (!ops.empty())
    return (*ops.begin()).getResult();

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&block);
  auto loc = builder.getUnknownLoc();

  // auto device = builder.create<intel_gpu::GetDeviceOp>(loc).getResult();
  // auto context =
  //    builder.create<intel_gpu::CreateContextOp>(loc, device).getResult();

  auto sycl_stream = builder.create<intel_gpu::GetStreamOp>(loc).getResult();
  builder.setInsertionPoint(block.getTerminator());
  builder.create<intel_gpu::DestroyStreamOp>(loc, sycl_stream);
  return sycl_stream;
}

class ConvertGetDeviceOpPattern
    : public ConvertOpToGpuRuntimeCallPattern<intel_gpu::GetDeviceOp> {
public:
  ConvertGetDeviceOpPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<intel_gpu::GetDeviceOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(intel_gpu::GetDeviceOp op,
                  intel_gpu::GetDeviceOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();
    auto loc = op.getLoc();
    auto res = deviceCallBuilder.create(loc, rewriter, {});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertCreateContextOpPattern
    : public ConvertOpToGpuRuntimeCallPattern<intel_gpu::CreateContextOp> {
public:
  ConvertCreateContextOpPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<intel_gpu::CreateContextOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(intel_gpu::CreateContextOp op,
                  intel_gpu::CreateContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto loc = op.getLoc();
    auto res =
        contextCreateCallBuilder.create(loc, rewriter, {adaptor.device()});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGetStreamOpPattern
    : public ConvertOpToGpuRuntimeCallPattern<intel_gpu::GetStreamOp> {
public:
  ConvertGetStreamOpPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<intel_gpu::GetStreamOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(intel_gpu::GetStreamOp op,
                  intel_gpu::GetStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto loc = op.getLoc();
    // This is the pointer to stream you will receive at runtime
    auto stream_ptr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    auto res = streamCallBuilder.create(loc, rewriter, {stream_ptr});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertDestroyStreamOpPattern
    : public ConvertOpToGpuRuntimeCallPattern<intel_gpu::DestroyStreamOp> {
public:
  ConvertDestroyStreamOpPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<intel_gpu::DestroyStreamOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(intel_gpu::DestroyStreamOp op,
                  intel_gpu::DestroyStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = streamDestroyCallBuilder.create(loc, rewriter, adaptor.stream());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

/// A rewrite pattern to convert gpu.alloc operations into a GPU runtime
/// call.
class ConvertAllocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::AllocOp> {
public:
  ConvertAllocOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::AllocOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType memRefType = allocOp.getType();

    if (failed(areAllLLVMTypes(allocOp, adaptor.getOperands(), rewriter)) ||
        !isConvertibleAndHasIdentityMaps(memRefType) ||
        failed(isAsyncWithOneDependency(rewriter, allocOp)))
      return failure();
    auto loc = allocOp.getLoc();

    // Get shape of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.
    SmallVector<Value, 4> shape;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.dynamicSizes(), rewriter,
                             shape, strides, sizeBytes);

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    Type elementPtrType = this->getElementPtrType(memRefType);

    Value allocatedPtr =
        allocCallBuilder.create(loc, rewriter, {sizeBytes, getStream(rewriter)})
            .getResult(0);
    allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, allocatedPtr);

    // No alignment.
    Value alignedPtr = allocatedPtr;

    // Create the MemRef descriptor.
    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, allocatedPtr, alignedPtr, shape, strides, rewriter);

    rewriter.replaceOp(allocOp, {memRefDescriptor, getStream(rewriter)});

    return success();
  }
};

/// A rewrite pattern to convert gpu.dealloc operations into a GPU runtime
/// call.
class ConvertDeallocOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::DeallocOp> {
public:
  ConvertDeallocOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::DeallocOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::DeallocOp deallocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(areAllLLVMTypes(deallocOp, adaptor.getOperands(), rewriter)))
      return failure();

    Location loc = deallocOp.getLoc();

    Value pointer =
        MemRefDescriptor(adaptor.memref()).allocatedPtr(rewriter, loc);
    auto casted =
        rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pointer);
    Value stream = getStream(rewriter);
    deallocCallBuilder.create(loc, rewriter, {casted, stream});

    rewriter.replaceOp(deallocOp, {stream});
    return success();
  }
};

/// A rewrite pattern to convert gpu.wait operations into a GPU runtime
/// call.
// Converts `gpu.wait` to runtime calls. The converted op synchronizes the
// host with the stream/event operands. The operands are destroyed. That is,
// it assumes that it is not used afterwards or elsewhere. Otherwise we will
// get a runtime error. Eventually, we should guarantee this property.
class ConvertWaitOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp> {
public:
  ConvertWaitOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::WaitOp waitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (waitOp.asyncToken())
      return rewriter.notifyMatchFailure(waitOp, "Cannot convert async op.");

    Location loc = waitOp.getLoc();

    for (auto operand : adaptor.getOperands()) {
      if (isDefinedByCallTo(operand, streamCallBuilder.functionName)) {
        // The converted operand's definition created a stream.
        streamSynchronizeCallBuilder.create(loc, rewriter, {operand});
        // streamDestroyCallBuilder.create(loc, rewriter, {operand});
      } else {
        // Otherwise the converted operand is an event. This assumes that we
        // use events in control flow code as well.
        eventSynchronizeCallBuilder.create(loc, rewriter, {operand});
        eventDestroyCallBuilder.create(loc, rewriter, {operand});
      }
    }

    rewriter.eraseOp(waitOp);
    return success();
  }
};

// Converts `gpu.wait async` to runtime calls. The converted op creates a new
// stream that is synchronized with stream/event operands. The operands are
// destroyed. That is, it assumes that it is not used afterwards or elsewhere.
// Otherwise we will get a runtime error. Eventually, we should guarantee this
// property.
class ConvertWaitAsyncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp> {
public:
  ConvertWaitAsyncOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter)
      : ConvertOpToGpuRuntimeCallPattern<gpu::WaitOp>(typeConverter) {}

private:
  LogicalResult
  matchAndRewrite(gpu::WaitOp waitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!waitOp.asyncToken())
      return rewriter.notifyMatchFailure(waitOp, "Can only convert async op.");

    Location loc = waitOp.getLoc();

    auto insertionPoint = rewriter.saveInsertionPoint();
    SmallVector<Value, 1> events;
    for (auto pair :
         llvm::zip(waitOp.asyncDependencies(), adaptor.getOperands())) {
      auto operand = std::get<1>(pair);
      if (isDefinedByCallTo(operand, streamCallBuilder.functionName)) {
        // The converted operand's definition created a stream. Insert an
        // event into the stream just after the last use of the original token
        // operand.
        auto *defOp = std::get<0>(pair).getDefiningOp();
        rewriter.setInsertionPointAfter(defOp);
        auto event =
            eventCreateCallBuilder.create(loc, rewriter, {}).getResult(0);
        eventRecordCallBuilder.create(loc, rewriter, {event, operand});
        events.push_back(event);
      } else {
        // Otherwise the converted operand is an event. This assumes that we
        // use events in control flow code as well.
        events.push_back(operand);
      }
    }
    rewriter.restoreInsertionPoint(insertionPoint);

    for (auto event : events)
      streamWaitEventCallBuilder.create(loc, rewriter,
                                        {getStream(rewriter), event});
    for (auto event : events)
      eventDestroyCallBuilder.create(loc, rewriter, {event});
    rewriter.replaceOp(waitOp, {getStream(rewriter)});

    return success();
  }
};

class ConvertLaunchFuncOpToGpuRuntimeCallPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp> {
public:
  ConvertLaunchFuncOpToGpuRuntimeCallPattern(LLVMTypeConverter &typeConverter,
                                             StringRef gpuBinaryAnnotation)
      : ConvertOpToGpuRuntimeCallPattern<gpu::LaunchFuncOp>(typeConverter),
        gpuBinaryAnnotation(gpuBinaryAnnotation) {}

private:
  llvm::SmallString<32> gpuBinaryAnnotation;

  Value generateParamsArray(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                            OpBuilder &builder) const {
    auto loc = launchOp.getLoc();
    auto numKernelOperands = launchOp.getNumKernelOperands();
    auto arguments = getTypeConverter()->promoteOperands(
        loc, launchOp.getOperands().take_back(numKernelOperands),
        adaptor.getOperands().take_back(numKernelOperands), builder);
    auto numArguments = arguments.size();
    SmallVector<Type, 4> argumentTypes;
    argumentTypes.reserve(numArguments);
    for (auto argument : arguments)
      argumentTypes.push_back(argument.getType());
    auto structType = LLVM::LLVMStructType::getNewIdentified(
        context, StringRef(), argumentTypes);
    auto one = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                builder.getI32IntegerAttr(1));
    auto structPtr = builder.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(structType), one, /*alignment=*/0);
    auto arraySize = builder.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, builder.getI32IntegerAttr(numArguments));
    auto arrayPtr = builder.create<LLVM::AllocaOp>(loc, llvmPointerPointerType,
                                                   arraySize, /*alignment=*/0);
    auto zero = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type,
                                                 builder.getI32IntegerAttr(0));
    for (const auto &en : llvm::enumerate(arguments)) {
      auto index = builder.create<LLVM::ConstantOp>(
          loc, llvmInt32Type, builder.getI32IntegerAttr(en.index()));
      auto fieldPtr = builder.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(argumentTypes[en.index()]), structPtr,
          ArrayRef<Value>{zero, index.getResult()});
      builder.create<LLVM::StoreOp>(loc, en.value(), fieldPtr);
      auto elementPtr = builder.create<LLVM::GEPOp>(
          loc, llvmPointerPointerType, arrayPtr, index.getResult());
      auto casted =
          builder.create<LLVM::BitcastOp>(loc, llvmPointerType, fieldPtr);
      builder.create<LLVM::StoreOp>(loc, casted, elementPtr);
    }
    return arrayPtr;
  }

  Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                   Location loc, OpBuilder &builder) const {
    // Make sure the trailing zero is included in the constant.
    std::vector<char> kernelName(name.begin(), name.end());
    kernelName.push_back('\0');

    std::string globalName =
        std::string(llvm::formatv("{0}_{1}_kernel_name", moduleName, name));
    return LLVM::createGlobalString(
        loc, builder, globalName,
        StringRef(kernelName.data(), kernelName.size()),
        LLVM::Linkage::Internal);
  }

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(areAllLLVMTypes(launchOp, adaptor.getOperands(), rewriter)))
      return failure();

    if (launchOp.asyncDependencies().size() > 1)
      return rewriter.notifyMatchFailure(
          launchOp, "Cannot convert with more than one async dependency.");

    // Fail when the synchronous version of the op has async dependencies. The
    // lowering destroys the stream, and we do not want to check that there is
    // no use of the stream after this op.
    if (!launchOp.asyncToken() && !launchOp.asyncDependencies().empty())
      return rewriter.notifyMatchFailure(
          launchOp, "Cannot convert non-async op with async dependencies.");

    Location loc = launchOp.getLoc();

    // Create an LLVM global with CUBIN extracted from the kernel annotation
    // and obtain a pointer to the first byte in it.
    auto kernelModule = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
        launchOp, launchOp.getKernelModuleName());
    assert(kernelModule && "expected a kernel module");

    auto binaryAttr =
        kernelModule->getAttrOfType<StringAttr>(gpuBinaryAnnotation);
    if (!binaryAttr) {
      kernelModule.emitOpError()
          << "missing " << gpuBinaryAnnotation << " attribute";
      return failure();
    }

    auto blob = binaryAttr.getValue();
    SmallString<128> nameBuffer(kernelModule.getName());
    nameBuffer.append(kGpuBinaryStorageSuffix);
    Value data = LLVM::createGlobalString(loc, rewriter, nameBuffer.str(),
                                          binaryAttr.getValue(),
                                          LLVM::Linkage::Internal);

    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIntPtrType,
        mlir::IntegerAttr::get(llvmIntPtrType,
                               static_cast<int64_t>(blob.size())));

    Value stream = getStream(rewriter);

    auto module =
        moduleLoadCallBuilder.create(loc, rewriter, {stream, data, size});
    // Get the function from the module. The name corresponds to the name of
    // the kernel function.
    auto kernelName = generateKernelNameConstant(
        launchOp.getKernelModuleName().getValue(),
        launchOp.getKernelName().getValue(), loc, rewriter);
    auto function = moduleGetFunctionCallBuilder.create(
        loc, rewriter, {stream, module.getResult(0), kernelName});
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(0));
    // Create array of pointers to kernel arguments.
    auto kernelParams = generateParamsArray(launchOp, adaptor, rewriter);
    auto nullpointer =
        rewriter.create<LLVM::NullOp>(loc, llvmPointerPointerType);
    Value dynamicSharedMemorySize = launchOp.dynamicSharedMemorySize()
                                        ? launchOp.dynamicSharedMemorySize()
                                        : zero;
    launchKernelCallBuilder.create(loc, rewriter,
                                   {stream, function.getResult(0),
                                    adaptor.gridSizeX(), adaptor.gridSizeY(),
                                    adaptor.gridSizeZ(), adaptor.blockSizeX(),
                                    adaptor.blockSizeY(), adaptor.blockSizeZ(),
                                    dynamicSharedMemorySize, kernelParams,
                                    /*extra=*/nullpointer});

    if (launchOp.asyncToken()) {
      // Async launch: make dependent ops use the same stream.
      rewriter.replaceOp(launchOp, {stream});
    } else {
      // Synchronize with host and destroy stream. This must be the stream
      // created above (with no other uses) because we check that the
      // synchronous version does not have any async dependencies.
      streamSynchronizeCallBuilder.create(loc, rewriter, stream);
      // streamDestroyCallBuilder.create(loc, rewriter, stream);
      rewriter.eraseOp(launchOp);
    }
    moduleUnloadCallBuilder.create(loc, rewriter, module.getResult(0));

    return success();
  }
}; // namespace

// TODO(nbpatel) :Fix this. Entry point is gpu::AllocOp
struct AddIntelGpuOps : public mlir::OpRewritePattern<gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(gpu::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    return mlir::success();
  }
};

struct IntelGpuOpsPass
    : public mlir::PassWrapper<IntelGpuOpsPass, mlir::OperationPass<void>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<intel_gpu::IntelGpuDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<AddIntelGpuOps>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct IntelStreamToLLVMPass
    : public mlir::PassWrapper<IntelStreamToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<intel_gpu::IntelGpuDialect>();
  }

  void runOnOperation() override {
    mlir::LLVMTypeConverter converter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMConversionTarget target(getContext());

    auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));

    converter.addConversion(
        [llvmPointerType](intel_gpu::OpaqueType) -> mlir::Type {
          return llvmPointerType;
        });

    target.addIllegalOp<
        // clang-format off
        intel_gpu::GetStreamOp,
        intel_gpu::DestroyStreamOp
        // clang-format on
        >();

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);

    patterns.add<ConvertGetStreamOpPattern, ConvertDestroyStreamOpPattern>(
        converter);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

struct IntelGPUToLLVMPass
    : public mlir::PassWrapper<IntelGPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<intel_gpu::IntelGpuDialect>();
  }

  void runOnOperation() override {
    mlir::LLVMTypeConverter converter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMConversionTarget target(getContext());

    converter.addConversion(
        [context = &converter.getContext()](gpu::AsyncTokenType type) -> Type {
          return LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
        });

    auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));

    converter.addConversion(
        [llvmPointerType](intel_gpu::OpaqueType) -> mlir::Type {
          return llvmPointerType;
        });

    // target.addIllegalDialect<gpu::GPUDialect>();

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

    patterns.add<ConvertAllocOpToGpuRuntimeCallPattern,
                 ConvertDeallocOpToGpuRuntimeCallPattern,
                 ConvertWaitAsyncOpToGpuRuntimeCallPattern,
                 ConvertWaitOpToGpuRuntimeCallPattern>(converter);

    patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
        converter, mlir::gpu::getDefaultGpuBinaryAnnotation());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> intel_gpu::createIntelGpuOpsPass() {
  return std::make_unique<IntelGpuOpsPass>();
}

std::unique_ptr<mlir::Pass> intel_gpu::createIntelStreamToLLVMPass() {
  return std::make_unique<IntelStreamToLLVMPass>();
}

std::unique_ptr<mlir::Pass> intel_gpu::createIntelGPUToLLVMPass() {
  return std::make_unique<IntelGPUToLLVMPass>();
}
