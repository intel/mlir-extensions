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

#include "mlir-extensions/dialect/intel_gpu/IR/intel_gpu_ops.hpp"

#include "mlir-extensions/transforms/func_utils.hpp"

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>

using namespace mlir;

static constexpr const char *kGpuBinaryStorageSuffix = "_gpubin_cst";

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

private:
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
      "mgpuModuleLoad",
      llvmPointerType /* void *module */,
      {llvmPointerType /* void *cubin */}};
  FunctionCallBuilder moduleUnloadCallBuilder = {
      "mgpuModuleUnload", llvmVoidType, {llvmPointerType /* void *module */}};
  FunctionCallBuilder moduleGetFunctionCallBuilder = {
      "mgpuModuleGetFunction",
      llvmPointerType /* void *function */,
      {
          llvmPointerType, /* void *module */
          llvmPointerType  /* char *name   */
      }};
  FunctionCallBuilder launchKernelCallBuilder = {
      "mgpuLaunchKernel",
      llvmVoidType,
      {
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
      "mgpuStreamCreate", llvmPointerType /* void *stream */, {}};
  FunctionCallBuilder streamDestroyCallBuilder = {
      "mgpuStreamDestroy", llvmVoidType, {llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamSynchronizeCallBuilder = {
      "mgpuStreamSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *stream */}};
  FunctionCallBuilder allocCallBuilder = {
      "gpuMemAlloc",
      llvmPointerType /* void * */,
      {llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder deallocCallBuilder = {
      "gpuMemFree",
      llvmVoidType,
      {llvmPointerType /* void *ptr */, llvmPointerType /* void *stream */}};
  FunctionCallBuilder memcpyCallBuilder = {
      "mgpuMemcpy",
      llvmVoidType,
      {llvmPointerType /* void *dst */, llvmPointerType /* void *src */,
       llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder memsetCallBuilder = {
      "mgpuMemset32",
      llvmVoidType,
      {llvmPointerType /* void *dst */, llvmInt32Type /* unsigned int value */,
       llvmIntPtrType /* intptr_t sizeBytes */,
       llvmPointerType /* void *stream */}};
  FunctionCallBuilder deviceCallBuilder = {
      "gpuGetDevice",
      llvmPointerType /* void * */,
      {llvmPointerType /* void* platform */, llvmInt8Type /* oridinal */}};
  FunctionCallBuilder contextCallBuilder = {
      "gpuCreateContext",
      llvmPointerType /* void * */,
      {llvmPointerType /* void* device */}};
  FunctionCallBuilder streamCallBuilder = {
      "gpuCreateStream",
      llvmPointerType /* void * */,
      {llvmPointerType /* void* context */}};
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

    std::cout << " IN ConvertAllocOpToGpuRuntimeCallPattern " << std::endl;
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
    auto stream = adaptor.asyncDependencies().front();
    Value allocatedPtr =
        allocCallBuilder.create(loc, rewriter, {sizeBytes, stream})
            .getResult(0);
    allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, allocatedPtr);

    // No alignment.
    Value alignedPtr = allocatedPtr;

    // Create the MemRef descriptor.
    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, allocatedPtr, alignedPtr, shape, strides, rewriter);

    rewriter.replaceOp(allocOp, {memRefDescriptor, stream});

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
    if (failed(areAllLLVMTypes(deallocOp, adaptor.getOperands(), rewriter)) ||
        failed(isAsyncWithOneDependency(rewriter, deallocOp)))
      return failure();

    Location loc = deallocOp.getLoc();

    Value pointer =
        MemRefDescriptor(adaptor.memref()).allocatedPtr(rewriter, loc);
    auto casted =
        rewriter.create<LLVM::BitcastOp>(loc, llvmPointerType, pointer);
    Value stream = adaptor.asyncDependencies().front();
    deallocCallBuilder.create(loc, rewriter, {casted, stream});

    rewriter.replaceOp(deallocOp, {stream});
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

    SmallString<128> nameBuffer(kernelModule.getName());
    nameBuffer.append(kGpuBinaryStorageSuffix);
    Value data = LLVM::createGlobalString(loc, rewriter, nameBuffer.str(),
                                          binaryAttr.getValue(),
                                          LLVM::Linkage::Internal);

    // Code for Intel GPU ops
    StringRef platform_name = "IntelGPU";
    auto name = mlir::LLVM::createGlobalString(loc, rewriter, platform_name,
                                               platform_name,
                                               mlir::LLVM::Linkage::Internal);
    Value ordinal = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt8Type,
        mlir::IntegerAttr::get(llvmInt8Type, static_cast<int8_t>(0)));
    auto device = deviceCallBuilder.create(loc, rewriter, {name, ordinal});
    auto context =
        contextCallBuilder.create(loc, rewriter, device.getResult(0));
    auto intel_stream =
        streamCallBuilder.create(loc, rewriter, context.getResult(0));

    // End Code for Intel GPU ops
    auto module = moduleLoadCallBuilder.create(loc, rewriter, data);
    // Get the function from the module. The name corresponds to the name of
    // the kernel function.
    auto kernelName = generateKernelNameConstant(
        launchOp.getKernelModuleName().getValue(),
        launchOp.getKernelName().getValue(), loc, rewriter);
    auto function = moduleGetFunctionCallBuilder.create(
        loc, rewriter, {module.getResult(0), kernelName});
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(0));
    Value stream =
        adaptor.asyncDependencies().empty()
            ? streamCreateCallBuilder.create(loc, rewriter, {}).getResult(0)
            : adaptor.asyncDependencies().front();
    // Create array of pointers to kernel arguments.
    auto kernelParams = generateParamsArray(launchOp, adaptor, rewriter);
    auto nullpointer =
        rewriter.create<LLVM::NullOp>(loc, llvmPointerPointerType);
    Value dynamicSharedMemorySize = launchOp.dynamicSharedMemorySize()
                                        ? launchOp.dynamicSharedMemorySize()
                                        : zero;
    launchKernelCallBuilder.create(
        loc, rewriter,
        {function.getResult(0), adaptor.gridSizeX(), adaptor.gridSizeY(),
         adaptor.gridSizeZ(), adaptor.blockSizeX(), adaptor.blockSizeY(),
         adaptor.blockSizeZ(), dynamicSharedMemorySize, stream, kernelParams,
         /*extra=*/nullpointer});

    if (launchOp.asyncToken()) {
      // Async launch: make dependent ops use the same stream.
      rewriter.replaceOp(launchOp, {stream});
    } else {
      // Synchronize with host and destroy stream. This must be the stream
      // created above (with no other uses) because we check that the
      // synchronous version does not have any async dependencies.
      streamSynchronizeCallBuilder.create(loc, rewriter, stream);
      streamDestroyCallBuilder.create(loc, rewriter, stream);
      rewriter.eraseOp(launchOp);
    }
    moduleUnloadCallBuilder.create(loc, rewriter, module.getResult(0));

    return success();
  }
}; // namespace

struct IntelGPUToLLVMPass
    : public mlir::PassWrapper<IntelGPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
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

    // target.addIllegalDialect<mlir::gpu::GPUDialect>();
    target.addIllegalOp<
        // clang-format off
        intel_gpu::GetDeviceOp,
        intel_gpu::CreateContextOp,
        intel_gpu::CreateStreamOp
        // clang-format on
        >();

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

    patterns.add<ConvertAllocOpToGpuRuntimeCallPattern,
                 ConvertDeallocOpToGpuRuntimeCallPattern>(converter);

    patterns.add<ConvertLaunchFuncOpToGpuRuntimeCallPattern>(
        converter, mlir::gpu::getDefaultGpuBinaryAnnotation());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> intel_gpu::createIntelGPUToLLVMPass() {
  return std::make_unique<IntelGPUToLLVMPass>();
}
