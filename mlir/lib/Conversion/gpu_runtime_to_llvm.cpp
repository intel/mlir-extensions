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

#include "mlir-extensions/Conversion/gpu_runtime_to_llvm.hpp"

#include "mlir-extensions/Dialect/gpu_runtime/IR/gpu_runtime_ops.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"
#include "mlir-extensions/Transforms/func_utils.hpp"

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace {
struct LowerTakeContext
    : public mlir::ConvertOpToLLVMPattern<plier::TakeContextOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::TakeContextOp op,
                  plier::TakeContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcTypes = op.getResultTypes();
    auto count = static_cast<unsigned>(srcTypes.size());
    llvm::SmallVector<mlir::Type> newTypes(count);
    auto converter = getTypeConverter();
    assert(converter);
    for (auto i : llvm::seq(0u, count)) {
      auto oldType = srcTypes[i];
      auto newType = converter->convertType(oldType);
      newTypes[i] = (newType ? newType : oldType);
    }

    auto initFunc = adaptor.initFunc().getValueOr(mlir::SymbolRefAttr());
    auto releaseFunc = adaptor.releaseFunc().getValueOr(mlir::SymbolRefAttr());
    rewriter.replaceOpWithNewOp<plier::TakeContextOp>(op, newTypes, initFunc,
                                                      releaseFunc);
    return mlir::success();
  }
};

#if !defined(IMEX_ENABLE_NUMBA_HOTFIX)
struct LowerUndef : public mlir::ConvertOpToLLVMPattern<plier::UndefOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UndefOp op, plier::UndefOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto type = converter->convertType(op.getType());
    if (!type)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, type);
    return mlir::success();
  }
};
#endif

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

  mlir::Type llvmI32PtrType = mlir::LLVM::LLVMPointerType::get(llvmIndexType);

  mlir::Type llvmRangeType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmIndexType});
  mlir::Type llvmRangePointerType =
      mlir::LLVM::LLVMPointerType::get(llvmRangeType);
  mlir::Type llvmAllocResType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmPointerType, llvmPointerType});
  mlir::Type llvmAllocResPtrType =
      mlir::LLVM::LLVMPointerType::get(llvmAllocResType);

  FunctionCallBuilder streamCreateCallBuilder = {
      "dpcompGpuStreamCreate",
      llvmPointerType, // stream
      {
          llvmIndexType // events count
      }};

  FunctionCallBuilder streamDestroyCallBuilder = {"dpcompGpuStreamDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // stream
                                                  }};

  FunctionCallBuilder moduleLoadCallBuilder = {"dpcompGpuModuleLoad",
                                               llvmPointerType, // module
                                               {
                                                   llvmPointerType, // stream
                                                   llvmPointerType, // data ptr
                                                   llvmIndexType,   // data size
                                               }};

  FunctionCallBuilder moduleDestroyCallBuilder = {"dpcompGpuModuleDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // module
                                                  }};

  FunctionCallBuilder kernelGetCallBuilder = {"dpcompGpuKernelGet",
                                              llvmPointerType, // kernel
                                              {
                                                  llvmPointerType, // module
                                                  llvmPointerType, // name
                                              }};

  FunctionCallBuilder kernelDestroyCallBuilder = {"dpcompGpuKernelDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // kernel
                                                  }};

  FunctionCallBuilder launchKernelCallBuilder = {
      "dpcompGpuLaunchKernel",
      llvmPointerType, // dep
      {
          llvmPointerType,        // stream
          llvmPointerType,        // kernel
          llvmIndexType,          // gridXDim
          llvmIndexType,          // gridyDim
          llvmIndexType,          // gridZDim
          llvmIndexType,          // blockXDim
          llvmIndexType,          // blockYDim
          llvmIndexType,          // blockZDim
          llvmPointerPointerType, // deps (null-term)
          llvmRangePointerType,   // params (null-term)
          llvmIndexType,          // eventIndex
      }};

  FunctionCallBuilder waitEventCallBuilder = {"dpcompGpuWait",
                                              llvmVoidType,
                                              {
                                                  llvmPointerType // dep
                                              }};

  FunctionCallBuilder allocCallBuilder = {
      "dpcompGpuAlloc",
      llvmVoidType,
      {
          llvmPointerType,        // stream
          llvmIndexType,          // size
          llvmIndexType,          // alignment
          llvmInt32Type,          // shared
          llvmPointerPointerType, // deps (null-term)
          llvmIndexType,          // eventIndex
          llvmAllocResPtrType,    // result
      }};

  FunctionCallBuilder deallocCallBuilder = {
      "dpcompGpuDeAlloc",
      llvmVoidType,
      {
          llvmPointerType, // stream
          llvmPointerType, // memory pointer
      }};

  FunctionCallBuilder suggestBlockSizeBuilder = {
      "dpcompGpuSuggestBlockSize",
      llvmVoidType,
      {
          llvmPointerType, // stream
          llvmPointerType, // kernel
          llvmI32PtrType,  // grid sizes
          llvmI32PtrType,  // ret block sizes
          llvmIndexType,   // dim count
      }};

  mlir::Value createDepsArray(mlir::OpBuilder &rewriter, mlir::Location loc,
                              mlir::Operation *op,
                              mlir::ValueRange deps) const {
    auto depsArraySize = static_cast<unsigned>(deps.size());
    auto depsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmPointerType, depsArraySize + 1);
    mlir::Value depsArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, depsArrayType);
    for (auto i : llvm::seq(0u, depsArraySize)) {
      auto index = rewriter.getI64ArrayAttr(i);
      depsArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, depsArray,
                                                             deps[i], index);
    }
    auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    depsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, depsArray, nullPtr, rewriter.getI64ArrayAttr(depsArraySize));

    auto depsArrayPtrType = mlir::LLVM::LLVMPointerType::get(depsArrayType);
    plier::AllocaInsertionPoint allocaHelper(op);
    auto depsArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, depsArrayPtrType, size,
                                                   0);
    });

    rewriter.create<mlir::LLVM::StoreOp>(loc, depsArray, depsArrayPtr);

    return rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerPointerType,
                                                  depsArrayPtr);
  }

  mlir::Value createEventIndexVar(mlir::OpBuilder &rewriter, mlir::Location loc,
                                  mlir::Operation *op) const {
    auto eventIndex = [&]() -> int64_t {
      auto value = mlir::getConstantIntValue(op->getAttr(kEventIndexAttrName));
      if (!value)
        return -1;

      return *value;
    }();
    return rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, eventIndex));
  }
};

class ConvertGpuStreamCreatePattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::CreateGpuStreamOp> {
public:
  ConvertGpuStreamCreatePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::CreateGpuStreamOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::CreateGpuStreamOp op,
                  gpu_runtime::CreateGpuStreamOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto eventsCount =
        mlir::getConstantIntValue(mod->getAttr(kEventCountAttrName));
    if (!eventsCount)
      return mlir::failure();

    auto loc = op.getLoc();
    auto eventsCountVar =
        rewriter
            .create<mlir::LLVM::ConstantOp>(
                loc, llvmIndexType,
                rewriter.getIntegerAttr(llvmIndexType, *eventsCount))
            .getResult();
    auto res = streamCreateCallBuilder.create(loc, rewriter, eventsCountVar);
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuStreamDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuStreamOp> {
public:
  ConvertGpuStreamDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuStreamOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::DestroyGpuStreamOp op,
                  gpu_runtime::DestroyGpuStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = streamDestroyCallBuilder.create(loc, rewriter, adaptor.source());
    rewriter.replaceOp(op, res.getResults());
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

class ConvertGpuModuleLoadPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LoadGpuModuleOp> {
public:
  ConvertGpuModuleLoadPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LoadGpuModuleOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::LoadGpuModuleOp op,
                  gpu_runtime::LoadGpuModuleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto gpuMod = mod.lookupSymbol<mlir::gpu::GPUModuleOp>(op.module());
    if (!gpuMod)
      return mlir::failure();

    auto blobAttr = gpuMod->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (!blobAttr)
      return mlir::failure();

    auto blob = blobAttr.getValue();

    auto loc = op.getLoc();
    auto name = getUniqueLLVMGlobalName(mod, "gpu_blob");
    auto data = mlir::LLVM::createGlobalString(loc, rewriter, name, blob,
                                               mlir::LLVM::Linkage::Internal);
    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType,
        mlir::IntegerAttr::get(llvmIndexType,
                               static_cast<int64_t>(blob.size())));
    auto res = moduleLoadCallBuilder.create(loc, rewriter,
                                            {adaptor.stream(), data, size});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuModuleDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuModuleOp> {
public:
  ConvertGpuModuleDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuModuleOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::DestroyGpuModuleOp op,
                  gpu_runtime::DestroyGpuModuleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = moduleDestroyCallBuilder.create(loc, rewriter, adaptor.source());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelGetPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GetGpuKernelOp> {
public:
  ConvertGpuKernelGetPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GetGpuKernelOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GetGpuKernelOp op,
                  gpu_runtime::GetGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto loc = op.getLoc();
    llvm::SmallString<64> name = op.kernel().getLeafReference().getValue();

    auto varName = getUniqueLLVMGlobalName(mod, "kernel_name");
    name.push_back('\0');
    auto data = mlir::LLVM::createGlobalString(loc, rewriter, varName, name,
                                               mlir::LLVM::Linkage::Internal);
    auto res =
        kernelGetCallBuilder.create(loc, rewriter, {adaptor.module(), data});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuKernelOp> {
public:
  ConvertGpuKernelDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuKernelOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::DestroyGpuKernelOp op,
                  gpu_runtime::DestroyGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = kernelDestroyCallBuilder.create(loc, rewriter, adaptor.source());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelLaunchPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LaunchGpuKernelOp> {
public:
  ConvertGpuKernelLaunchPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LaunchGpuKernelOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::LaunchGpuKernelOp op,
                  gpu_runtime::LaunchGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto depsArrayPtr =
        createDepsArray(rewriter, loc, op, adaptor.asyncDependencies());

    plier::AllocaInsertionPoint allocaHelper(op);
    auto kernelParams = adaptor.operands();
    auto paramsCount = static_cast<unsigned>(kernelParams.size());
    auto paramsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmRangeType, paramsCount + 1);
    auto paramsArrayPtrType = mlir::LLVM::LLVMPointerType::get(paramsArrayType);

    auto getKernelParamType = [&](unsigned i) -> mlir::Type {
      if (op.operands()[i].getType().isa<mlir::MemRefType>()) {
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
    auto localMemStorageClass = gpu_runtime::StorageClassAttr::get(
        getContext(), gpu_runtime::StorageClass::local);
    auto computeTypeSize = [&](mlir::Type type) -> mlir::Value {
      // %Size = getelementptr %T* null, int 1
      // %SizeI = ptrtoint %T* %Size to i32
      auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, type);
      auto gep = rewriter.create<mlir::LLVM::GEPOp>(loc, type, nullPtr, one);
      return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, llvmIndexType, gep);
    };

    auto getKernelParam =
        [&](unsigned i) -> std::pair<mlir::Value, mlir::Value> {
      auto memrefType = op.operands()[i].getType().dyn_cast<mlir::MemRefType>();
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
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, ptr, rewriter.getI64ArrayAttr(0));
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, typeSize, rewriter.getI64ArrayAttr(1));

      paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, paramsArray, range, rewriter.getI64ArrayAttr(i));
    }

    auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    auto nullRange = [&]() {
      auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));
      mlir::Value range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, llvmRangeType);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, nullPtr, rewriter.getI64ArrayAttr(0));
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, zero, rewriter.getI64ArrayAttr(1));
      return range;
    }();
    paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, paramsArray, nullRange, rewriter.getI64ArrayAttr(paramsCount));
    rewriter.create<mlir::LLVM::StoreOp>(loc, paramsArray, paramsArrayPtr);

    auto eventIndexVar = createEventIndexVar(rewriter, loc, op);

    auto paramsArrayVoidPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, llvmRangePointerType, paramsArrayPtr);
    mlir::Value params[] = {
        // clang-format off
        adaptor.stream(),
        adaptor.kernel(),
        adaptor.gridSizeX(),
        adaptor.gridSizeY(),
        adaptor.gridSizeZ(),
        adaptor.blockSizeX(),
        adaptor.blockSizeY(),
        adaptor.blockSizeZ(),
        depsArrayPtr,
        paramsArrayVoidPtr,
        eventIndexVar,
        // clang-format on
    };
    auto event =
        launchKernelCallBuilder.create(loc, rewriter, params)->getResult(0);
    if (op.getNumResults() == 0) {
      waitEventCallBuilder.create(loc, rewriter, event);
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, event);
    }
    return mlir::success();
  }
};

class ConvertGpuAllocPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUAllocOp> {
public:
  ConvertGpuAllocPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUAllocOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUAllocOp op,
                  gpu_runtime::GPUAllocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.symbolOperands().empty())
      return mlir::failure();

    auto memrefType = op.getType();
    auto converter = getTypeConverter();
    auto dstType = converter->convertType(memrefType);
    if (!dstType)
      return mlir::failure();

    auto loc = op.getLoc();

    mlir::SmallVector<mlir::Value, 4> shape;
    mlir::SmallVector<mlir::Value, 4> strides;
    mlir::Value sizeBytes;
    getMemRefDescriptorSizes(loc, memrefType, adaptor.dynamicSizes(), rewriter,
                             shape, strides, sizeBytes);

    assert(shape.size() == strides.size());

    auto alignment = rewriter.getIntegerAttr(llvmIndexType, 64);
    auto alignmentVar =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmIndexType, alignment);

    bool shared = op->hasAttr(gpu_runtime::getAllocSharedAttrName());
    auto sharedVar = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type,
        rewriter.getI32IntegerAttr(static_cast<int>(shared)));

    auto depsArrayPtr =
        createDepsArray(rewriter, loc, op, adaptor.asyncDependencies());

    auto eventIndexVar = createEventIndexVar(rewriter, loc, op);

    plier::AllocaInsertionPoint allocaHelper(op);
    auto resultPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmAllocResPtrType,
                                                   size, 0);
    });

    mlir::Value params[] = {
        // clang-format off
        adaptor.stream(),
        sizeBytes,
        alignmentVar,
        sharedVar,
        depsArrayPtr,
        eventIndexVar,
        resultPtr,
        // clang-format on
    };
    allocCallBuilder.create(loc, rewriter, params);
    auto res = rewriter.create<mlir::LLVM::LoadOp>(loc, resultPtr);
    auto meminfo = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, rewriter.getI64ArrayAttr(0));
    auto dataPtr = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, rewriter.getI64ArrayAttr(1));

    auto memrefDesc = mlir::MemRefDescriptor::undef(rewriter, loc, dstType);
    auto elemPtrTye = memrefDesc.getElementPtrType();
    memrefDesc.setAllocatedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, meminfo));
    memrefDesc.setAlignedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, dataPtr));

    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));

    memrefDesc.setOffset(rewriter, loc, zero);
    for (auto i : llvm::seq(0u, static_cast<unsigned>(shape.size()))) {
      memrefDesc.setSize(rewriter, loc, i, shape[i]);
      memrefDesc.setStride(rewriter, loc, i, strides[i]);
    }

    mlir::Value resMemref = memrefDesc;
    mlir::Value event = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, rewriter.getI64ArrayAttr(2));
    if (op.getNumResults() == 1) {
      waitEventCallBuilder.create(loc, rewriter, event);
      rewriter.replaceOp(op, resMemref);
    } else {
      mlir::Value vals[] = {
          resMemref,
          event,
      };
      rewriter.replaceOp(op, vals);
    }
    return mlir::success();
  }
};

class ConvertGpuDeAllocPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUDeallocOp> {
public:
  ConvertGpuDeAllocPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUDeallocOp>(converter) {
  }

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUDeallocOp op,
                  gpu_runtime::GPUDeallocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    mlir::Value pointer =
        mlir::MemRefDescriptor(adaptor.memref()).allocatedPtr(rewriter, loc);
    auto casted =
        rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerType, pointer);
    mlir::Value params[] = {adaptor.stream(), casted};
    auto res = deallocCallBuilder.create(loc, rewriter, params);
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuSuggestBlockSizePattern
    : public ConvertOpToGpuRuntimeCallPattern<
          gpu_runtime::GPUSuggestBlockSizeOp> {
public:
  ConvertGpuSuggestBlockSizePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUSuggestBlockSizeOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUSuggestBlockSizeOp op,
                  gpu_runtime::GPUSuggestBlockSizeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto numDims = op.getNumResults();
    auto loc = op.getLoc();
    plier::AllocaInsertionPoint allocaHelper(op);
    auto gridArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(numDims));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmI32PtrType, size,
                                                   0);
    });
    auto blockArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(numDims));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmI32PtrType, size,
                                                   0);
    });

    auto sizesType = mlir::LLVM::LLVMArrayType::get(llvmInt32Type, numDims);
    auto sizesPtrType = mlir::LLVM::LLVMPointerType::get((sizesType));
    auto castToSizesPtrType = [&](mlir::Value val) {
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, sizesPtrType, val);
    };

    mlir::Value gridArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, sizesType);
    for (auto i : llvm::seq(0u, numDims)) {
      auto index = rewriter.getI64ArrayAttr(i);
      auto gridSize = rewriter.create<mlir::LLVM::TruncOp>(
          loc, llvmInt32Type, adaptor.gridSize()[i]);
      gridArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, gridArray,
                                                             gridSize, index);
    }

    rewriter.create<mlir::LLVM::StoreOp>(loc, gridArray,
                                         castToSizesPtrType(gridArrayPtr));
    mlir::Value numDimsVal = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, numDims));

    mlir::Value params[] = {
        // clang-format off
        adaptor.stream(),
        adaptor.kernel(),
        gridArrayPtr,
        blockArrayPtr,
        numDimsVal,
        // clang-format on
    };

    suggestBlockSizeBuilder.create(loc, rewriter, params);

    mlir::Value blockSizeArray = rewriter.create<mlir::LLVM::LoadOp>(
        loc, castToSizesPtrType(blockArrayPtr));
    llvm::SmallVector<mlir::Value, 3> result(numDims);
    for (auto i : llvm::seq(0u, numDims)) {
      auto ind = rewriter.getI64ArrayAttr(i);
      auto blockSize = rewriter.create<mlir::LLVM::ExtractValueOp>(
          loc, llvmInt32Type, blockSizeArray, ind);
      result[i] =
          rewriter.create<mlir::LLVM::ZExtOp>(loc, llvmIndexType, blockSize);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct EnumerateEventsPass
    : public mlir::PassWrapper<EnumerateEventsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto mod = getOperation();
    int64_t eventCount = 0;
    auto *ctx = &getContext();
    auto intType = mlir::IntegerType::get(ctx, 64);
    auto indexAttrName = mlir::StringAttr::get(ctx, kEventIndexAttrName);
    auto countAttrName = mlir::StringAttr::get(ctx, kEventCountAttrName);
    mod.walk([&](mlir::gpu::AsyncOpInterface op) {
      //      if (op.getAsyncToken())
      op->setAttr(indexAttrName, mlir::IntegerAttr::get(intType, eventCount));
      ++eventCount;
    });
    mod->setAttr(countAttrName, mlir::IntegerAttr::get(intType, eventCount));
  }
};

struct GPUToLLVMPass
    : public mlir::PassWrapper<GPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::LLVMTypeConverter converter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMConversionTarget target(getContext());

    auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));
    converter.addConversion(
        [llvmPointerType](gpu_runtime::OpaqueType) -> mlir::Type {
          return llvmPointerType;
        });

    target.addIllegalDialect<mlir::gpu::GPUDialect>();
    target.addIllegalOp<
        // clang-format off
        gpu_runtime::CreateGpuStreamOp,
        gpu_runtime::DestroyGpuStreamOp,
        gpu_runtime::LoadGpuModuleOp,
        gpu_runtime::DestroyGpuModuleOp,
        gpu_runtime::GetGpuKernelOp,
        gpu_runtime::DestroyGpuKernelOp,
        gpu_runtime::LaunchGpuKernelOp,
        gpu_runtime::GPUAllocOp,
        gpu_runtime::GPUDeallocOp,
        gpu_runtime::GPUSuggestBlockSizeOp
        // clang-format on
        >();

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());
#if defined(IMEX_ENABLE_NUMBA_HOTFIX)
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    mlir::populateReturnOpTypeConversionPattern(patterns, converter);
    mlir::populateCallOpTypeConversionPattern(patterns, converter);
#endif

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) -> llvm::Optional<bool> {
          if (converter.isSignatureLegal(op.getFunctionType()) &&
              converter.isLegal(&op.getBody()))
            return true;

          return llvm::None;
        });

    target.addDynamicallyLegalOp<mlir::func::ReturnOp, plier::TakeContextOp,
                                 mlir::func::CallOp>(
        [&](mlir::Operation *op) -> llvm::Optional<bool> {
          for (auto range : {mlir::TypeRange(op->getOperandTypes()),
                             mlir::TypeRange(op->getResultTypes())})
            for (auto type : range)
              if (converter.isLegal(type))
                return true;

          return llvm::None;
        });
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) -> llvm::Optional<bool> {
          auto type = op.getFunctionType();
          for (auto range : {type.getInputs(), type.getResults()})
            for (auto type : range)
              if (converter.isLegal(type))
                return true;

          return llvm::None;
        });

    patterns.insert<
        // clang-format off
        ConvertGpuStreamCreatePattern,
        ConvertGpuStreamDestroyPattern,
        ConvertGpuModuleLoadPattern,
        ConvertGpuModuleDestroyPattern,
        ConvertGpuKernelGetPattern,
        ConvertGpuKernelDestroyPattern,
        ConvertGpuKernelLaunchPattern,
        ConvertGpuAllocPattern,
        ConvertGpuDeAllocPattern,
        ConvertGpuSuggestBlockSizePattern,
#if !defined(IMEX_ENABLE_NUMBA_HOTFIX)
        LowerUndef,
#endif
        LowerTakeContext
        // clang-format on
        >(converter);

    auto mod = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// Expose the passes to the outside world
std::unique_ptr<mlir::Pass> gpu_runtime::createEnumerateEventsPass() {
  return std::make_unique<EnumerateEventsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUToLLVMPass() {
  return std::make_unique<GPUToLLVMPass>();
}
