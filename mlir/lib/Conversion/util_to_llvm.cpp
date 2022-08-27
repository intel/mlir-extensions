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

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

#include "mlir-extensions/Dialect/plier/dialect.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"

#include "mlir-extensions/Conversion/util_to_llvm.hpp"

static mlir::Type convertTupleTypes(mlir::MLIRContext &context,
                                    mlir::TypeConverter &converter,
                                    mlir::TypeRange types) {
  if (types.empty())
    return mlir::LLVM::LLVMStructType::getLiteral(&context, llvm::None);

  auto unitupleType = [&]() -> mlir::Type {
    assert(!types.empty());
    auto elemType = types.front();
    auto tail = types.drop_front();
    if (llvm::all_of(tail, [&](auto t) { return t == elemType; }))
      return elemType;
    return nullptr;
  }();

  auto count = static_cast<unsigned>(types.size());
  if (unitupleType) {
    auto newType = converter.convertType(unitupleType);
    if (!newType)
      return {};
    return mlir::LLVM::LLVMArrayType::get(newType, count);
  }
  llvm::SmallVector<mlir::Type> newTypes;
  newTypes.reserve(count);
  for (auto type : types) {
    auto newType = converter.convertType(type);
    if (!newType)
      return {};
    newTypes.emplace_back(newType);
  }

  return mlir::LLVM::LLVMStructType::getLiteral(&context, newTypes);
}

static mlir::Type convertTuple(mlir::MLIRContext &context,
                               mlir::TypeConverter &converter,
                               mlir::TupleType tuple) {
  return convertTupleTypes(context, converter, tuple.getTypes());
}

static void
populateToLLVMAdditionalTypeConversion(mlir::LLVMTypeConverter &converter) {
  converter.addConversion(
      [&converter](mlir::TupleType type) -> llvm::Optional<mlir::Type> {
        auto res = convertTuple(*type.getContext(), converter, type);
        if (!res)
          return llvm::None;
        return res;
      });
  auto voidPtrType = mlir::LLVM::LLVMPointerType::get(
      mlir::IntegerType::get(&converter.getContext(), 8));
  converter.addConversion(
      [voidPtrType](mlir::NoneType) -> llvm::Optional<mlir::Type> {
        return voidPtrType;
      });
  converter.addConversion(
      [voidPtrType](imex::util::OpaqueType) -> llvm::Optional<mlir::Type> {
        return voidPtrType;
      });
}

namespace {
struct LowerRetainOp : public mlir::ConvertOpToLLVMPattern<imex::util::RetainOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::RetainOp op, imex::util::RetainOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto arg = adaptor.source();
    if (!arg.getType().isa<mlir::LLVM::LLVMStructType>())
      return mlir::failure();

    auto llvmVoidPointerType = getVoidPtrType();
    auto incref_func = [&]() {
      auto mod = op->getParentOfType<mlir::ModuleOp>();
      assert(mod);
      auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>("NRT_incref");
      if (!func) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto llvmVoidType = getVoidType();
        func = rewriter.create<mlir::LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), "NRT_incref",
            mlir::LLVM::LLVMFunctionType::get(llvmVoidType,
                                              llvmVoidPointerType));
      }
      return func;
    }();

    mlir::MemRefDescriptor source(arg);

    auto loc = op.getLoc();
    mlir::Value ptr = source.allocatedPtr(rewriter, loc);
    ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmVoidPointerType, ptr);
    rewriter.create<mlir::LLVM::CallOp>(loc, incref_func, ptr);
    rewriter.replaceOp(op, arg);

    return mlir::success();
  }
};

struct LowerExtractMemrefMetadataOp
    : public mlir::ConvertOpToLLVMPattern<imex::util::ExtractMemrefMetadataOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::ExtractMemrefMetadataOp op,
                  imex::util::ExtractMemrefMetadataOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto arg = adaptor.source();
    if (!arg.getType().isa<mlir::LLVM::LLVMStructType>())
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::MemRefDescriptor src(arg);
    auto val = [&]() -> mlir::Value {
      auto index = op.dimIndex().getSExtValue();
      if (index == -1)
        return src.offset(rewriter, loc);

      return src.stride(rewriter, loc, static_cast<unsigned>(index));
    }();

    rewriter.replaceOp(op, static_cast<mlir::Value>(val));
    return mlir::success();
  }
};

struct LowerBuildTuple
    : public mlir::ConvertOpToLLVMPattern<plier::BuildTupleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildTupleOp op, plier::BuildTupleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto type = converter->convertType(op.getType());
    if (!type)
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::Value init = rewriter.create<mlir::LLVM::UndefOp>(loc, type);
    for (auto it : llvm::enumerate(adaptor.args())) {
      auto arg = it.value();
      auto newType = arg.getType();
      assert(newType);
      auto index = rewriter.getI64ArrayAttr(static_cast<int64_t>(it.index()));
      init = rewriter.create<mlir::LLVM::InsertValueOp>(loc, init, arg, index);
    }

    rewriter.replaceOp(op, init);
    return mlir::success();
  }
};

struct LowerUndef : public mlir::ConvertOpToLLVMPattern<imex::util::UndefOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::UndefOp op, imex::util::UndefOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto type = converter->convertType(op.getType());
    if (!type)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, type);
    return mlir::success();
  }
};

static void addToGlobalDtors(mlir::ConversionPatternRewriter &rewriter,
                             mlir::ModuleOp mod, mlir::SymbolRefAttr attr,
                             int32_t priority) {
  auto loc = mod->getLoc();
  auto dtorOps = mod.getOps<mlir::LLVM::GlobalDtorsOp>();
  auto prioAttr = rewriter.getI32IntegerAttr(priority);
  mlir::OpBuilder::InsertionGuard g(rewriter);
  if (dtorOps.empty()) {
    rewriter.setInsertionPoint(mod.getBody(), std::prev(mod.getBody()->end()));
    auto syms = rewriter.getArrayAttr(attr);
    auto priorities = rewriter.getArrayAttr(prioAttr);
    rewriter.create<mlir::LLVM::GlobalDtorsOp>(loc, syms, priorities);
    return;
  }
  assert(llvm::hasSingleElement(dtorOps));
  auto dtorOp = *dtorOps.begin();

  auto addpendArray = [&](mlir::ArrayAttr arr,
                          mlir::Attribute attr) -> mlir::ArrayAttr {
    auto vals = arr.getValue();
    llvm::SmallVector<mlir::Attribute> ret(vals.begin(), vals.end());
    ret.emplace_back(attr);
    return rewriter.getArrayAttr(ret);
  };
  auto newDtors = addpendArray(dtorOp.getDtors(), attr);
  auto newPrioritiess = addpendArray(dtorOp.getPriorities(), prioAttr);
  rewriter.setInsertionPoint(dtorOp);
  rewriter.create<mlir::LLVM::GlobalDtorsOp>(loc, newDtors, newPrioritiess);
  rewriter.eraseOp(dtorOp);
}

struct LowerTakeContextOp
    : public mlir::ConvertOpToLLVMPattern<imex::util::TakeContextOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::TakeContextOp op,
                  imex::util::TakeContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);
    auto ctx = op.context();
    auto ctxType = converter->convertType(ctx.getType());
    if (!ctxType)
      return mlir::failure();

    mlir::ValueRange results = op.results();
    auto resultsCount = static_cast<unsigned>(results.size());
    llvm::SmallVector<mlir::Type> resultTypes(resultsCount);
    for (auto i : llvm::seq(0u, resultsCount)) {
      auto type = converter->convertType(results[i].getType());
      if (!type)
        return mlir::failure();

      resultTypes[i] = type;
    }

    auto ctxStructType =
        mlir::LLVM::LLVMStructType::getLiteral(getContext(), resultTypes);
    auto ctxStructPtrType = mlir::LLVM::LLVMPointerType::get(ctxStructType);

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto unknownLoc = rewriter.getUnknownLoc();
    auto loc = op->getLoc();
    auto wrapperType =
        mlir::LLVM::LLVMFunctionType::get(getVoidType(), ctxType);
    auto wrapperPtrType = mlir::LLVM::LLVMPointerType::get(wrapperType);
    mlir::Value initFuncPtr;

    auto insertFunc = [&](mlir::StringRef name, mlir::Type type,
                          mlir::LLVM::Linkage linkage) {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      return rewriter.create<mlir::LLVM::LLVMFuncOp>(unknownLoc, name, type,
                                                     linkage);
    };

    auto lookupFunc = [&](mlir::StringRef name, mlir::Type type) {
      // TODO: fix and use lookupOrCreateFn
      if (auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
        return func;

      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      return rewriter.create<mlir::LLVM::LLVMFuncOp>(
          unknownLoc, name, type, mlir::LLVM::Linkage::External);
    };

    if (auto initFuncSym = adaptor.initFuncAttr()) {
      auto funcName = initFuncSym.getLeafReference().getValue();
      auto wrapperName = (funcName + "_wrapper").str();

      auto initFunc = [&]() {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        auto func =
            insertFunc(wrapperName, wrapperType, mlir::LLVM::Linkage::Private);
        auto block = rewriter.createBlock(
            &func.getBody(), mlir::Region::iterator{}, ctxType, unknownLoc);
        rewriter.setInsertionPointToStart(block);

        // Get init func declaration so we can check original return types.
        auto initFunc = mod.lookupSymbol<mlir::func::FuncOp>(initFuncSym);
        assert(initFunc && "Invalid init func");
        auto initFuncType = initFunc.getFunctionType();
        assert(initFuncType.getNumResults() == resultsCount &&
               "Invalid init func");

        auto innerResults =
            rewriter
                .create<mlir::func::CallOp>(unknownLoc, initFuncSym,
                                            initFuncType.getResults())
                ->getResults();

        mlir::Value ctxStruct =
            rewriter.create<mlir::LLVM::UndefOp>(unknownLoc, ctxStructType);
        for (auto i : llvm::seq(0u, resultsCount)) {
          auto pos = rewriter.getI64ArrayAttr(i);
          auto srcType = initFuncType.getResult(i);
          auto convertedType = converter->convertType(srcType);
          assert(convertedType && "Invalid init func result type");

          mlir::Value val = innerResults[i];
          // Init function may not be type-converted at this point, so insert
          // conversion casts.
          if (convertedType != srcType)
            val = converter->materializeSourceConversion(rewriter, unknownLoc,
                                                         convertedType, val);
          assert(val && "Invalid init func result type");

          ctxStruct = rewriter.create<mlir::LLVM::InsertValueOp>(
              unknownLoc, ctxStruct, val, pos);
        }
        auto ptr = rewriter.create<mlir::LLVM::BitcastOp>(
            unknownLoc, ctxStructPtrType, block->getArgument(0));
        rewriter.create<mlir::LLVM::StoreOp>(unknownLoc, ctxStruct, ptr);
        rewriter.create<mlir::LLVM::ReturnOp>(unknownLoc, llvm::None);
        return func;
      }();

      initFuncPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, initFunc);
    } else {
      initFuncPtr = rewriter.create<mlir::LLVM::NullOp>(loc, wrapperPtrType);
    }

    mlir::Value deinitFuncPtr;
    if (auto deinitFuncSym = adaptor.releaseFuncAttr()) {
      auto funcName = deinitFuncSym.getLeafReference().getValue();
      auto wrapperName = (funcName + "_wrapper").str();

      auto deinitFunc = [&]() {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        auto func =
            insertFunc(wrapperName, wrapperType, mlir::LLVM::Linkage::Private);
        auto block = rewriter.createBlock(
            &func.getBody(), mlir::Region::iterator{}, ctxType, unknownLoc);
        rewriter.setInsertionPointToStart(block);

        auto ptr = rewriter.create<mlir::LLVM::BitcastOp>(
            unknownLoc, ctxStructPtrType, block->getArgument(0));
        auto ctxStruct =
            rewriter.create<mlir::LLVM::LoadOp>(unknownLoc, ctxStructType, ptr);

        // Get deinit func declaration so we can check original arg types.
        auto deinitFunc = mod.lookupSymbol<mlir::func::FuncOp>(deinitFuncSym);
        assert(deinitFunc && "Invalid deinit func");
        auto deinitFuncType = deinitFunc.getFunctionType();
        assert(deinitFuncType.getNumInputs() == resultsCount);

        llvm::SmallVector<mlir::Value> args(resultsCount);
        for (auto i : llvm::seq(0u, resultsCount)) {
          auto pos = rewriter.getI64ArrayAttr(i);
          mlir::Value val = rewriter.create<mlir::LLVM::ExtractValueOp>(
              unknownLoc, resultTypes[i], ctxStruct, pos);
          auto resType = deinitFuncType.getInput(i);
          // Deinit function may not be type-converted at this point, so insert
          // conversion casts.
          if (resultTypes[i] != resType)
            val = converter->materializeTargetConversion(rewriter, unknownLoc,
                                                         resType, val);

          args[i] = val;
        }

        rewriter.create<mlir::func::CallOp>(unknownLoc, deinitFuncSym,
                                            llvm::None, args);
        rewriter.create<mlir::LLVM::ReturnOp>(unknownLoc, llvm::None);
        return func;
      }();

      deinitFuncPtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, deinitFunc);
    } else {
      deinitFuncPtr = rewriter.create<mlir::LLVM::NullOp>(loc, wrapperPtrType);
    }

    auto takeCtxFunc = [&]() -> mlir::LLVM::LLVMFuncOp {
      llvm::StringRef name("dpcompTakeContext");
      auto retType = getVoidPtrType();
      const mlir::Type argTypes[] = {
          mlir::LLVM::LLVMPointerType::get(getVoidPtrType()),
          getIndexType(),
          wrapperPtrType,
          wrapperPtrType,
      };
      auto funcType = mlir::LLVM::LLVMFunctionType::get(retType, argTypes);
      return lookupFunc(name, funcType);
    }();

    auto purgeCtxFunc = [&]() -> mlir::LLVM::LLVMFuncOp {
      llvm::StringRef name("dpcompPurgeContext");
      auto retType = getVoidType();
      auto argType = mlir::LLVM::LLVMPointerType::get(getVoidPtrType());
      auto funcType = mlir::LLVM::LLVMFunctionType::get(retType, argType);
      return lookupFunc(name, funcType);
    }();

    auto ctxHandle = [&]() {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      llvm::StringRef name("context_handle"); // TODO: unique name
      auto handle = rewriter.create<mlir::LLVM::GlobalOp>(
          unknownLoc, ctxType, /*isConstant*/ false,
          mlir::LLVM::Linkage::Internal, name, mlir::Attribute());

      llvm::StringRef cleanupFuncName(".dpcomp_context_cleanup");
      auto cleanupFunc =
          mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(cleanupFuncName);
      if (!cleanupFunc) {
        auto cleanupFuncType =
            mlir::LLVM::LLVMFunctionType::get(getVoidType(), llvm::None);
        cleanupFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
            unknownLoc, cleanupFuncName, cleanupFuncType);
        auto block = rewriter.createBlock(&cleanupFunc.getBody());
        rewriter.setInsertionPointToStart(block);
        rewriter.create<mlir::LLVM::ReturnOp>(unknownLoc, llvm::None);

        addToGlobalDtors(rewriter, mod, mlir::SymbolRefAttr::get(cleanupFunc),
                         0);
      }

      assert(llvm::hasSingleElement(cleanupFunc.getBody()));
      rewriter.setInsertionPointToStart(&cleanupFunc.getBody().front());
      mlir::Value addr =
          rewriter.create<mlir::LLVM::AddressOfOp>(unknownLoc, handle);
      rewriter.create<mlir::LLVM::CallOp>(unknownLoc, purgeCtxFunc, addr);

      return handle;
    }();

    auto ctxHandlePtr =
        rewriter.create<mlir::LLVM::AddressOfOp>(loc, ctxHandle);
    auto contextSize = getSizeInBytes(loc, ctxStructType, rewriter);

    const mlir::Value takeCtxArgs[] = {
        ctxHandlePtr,
        contextSize,
        initFuncPtr,
        deinitFuncPtr,
    };
    auto ctxPtr =
        rewriter.create<mlir::LLVM::CallOp>(loc, takeCtxFunc, takeCtxArgs)
            .getResult(0);

    llvm::SmallVector<mlir::Value> takeCtxResults;
    takeCtxResults.emplace_back(ctxPtr);

    auto ctxStructPtr =
        rewriter.create<mlir::LLVM::BitcastOp>(loc, ctxStructPtrType, ctxPtr);
    auto ctxStruct =
        rewriter.create<mlir::LLVM::LoadOp>(loc, ctxStructType, ctxStructPtr);

    for (auto i : llvm::seq(0u, resultsCount)) {
      auto pos = rewriter.getI64ArrayAttr(i);
      auto res = rewriter.create<mlir::LLVM::ExtractValueOp>(
          loc, resultTypes[i], ctxStruct, pos);
      takeCtxResults.emplace_back(res);
    }

    rewriter.replaceOp(op, takeCtxResults);
    return mlir::success();
  }
};

struct LowerReleaseContextOp
    : public mlir::ConvertOpToLLVMPattern<imex::util::ReleaseContextOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::ReleaseContextOp op,
                  imex::util::ReleaseContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto unknownLoc = rewriter.getUnknownLoc();
    auto loc = op->getLoc();

    auto lookupFunc = [&](mlir::StringRef name, mlir::Type type) {
      // TODO: fix and use lookupOrCreateFn
      if (auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
        return func;

      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      return rewriter.create<mlir::LLVM::LLVMFuncOp>(
          unknownLoc, name, type, mlir::LLVM::Linkage::External);
    };

    auto releaseCtxFunc = [&]() -> mlir::LLVM::LLVMFuncOp {
      llvm::StringRef name("dpcompReleaseContext");
      auto voidPtr = getVoidPtrType();
      auto funcType = mlir::LLVM::LLVMFunctionType::get(voidPtr, voidPtr);
      return lookupFunc(name, funcType);
    }();

    rewriter.create<mlir::LLVM::CallOp>(loc, releaseCtxFunc, adaptor.context());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// Convert operations from the plier_util dialect to the LLVM dialect.
struct PlierUtilToLLVMPass
    : public mlir::PassWrapper<PlierUtilToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlierUtilToLLVMPass)

  PlierUtilToLLVMPass(
      std::function<mlir::LowerToLLVMOptions(mlir::MLIRContext &)> &&getter)
      : optsGetter(std::move(getter)) {}

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    auto &context = getContext();
    auto options = optsGetter(context);

    mlir::LLVMTypeConverter typeConverter(&context, options);
    populateToLLVMAdditionalTypeConversion(typeConverter);
    mlir::RewritePatternSet patterns(&context);

    patterns.insert<
        // clang-format off
        LowerUndef,
        LowerBuildTuple,
        LowerRetainOp,
        LowerExtractMemrefMetadataOp,
        LowerTakeContextOp,
        LowerReleaseContextOp
        // clang-format on
        >(typeConverter);

    mlir::LLVMConversionTarget target(context);
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::CallOp>();
    target.addIllegalDialect<imex::util::PlierUtilDialect>();
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }

private:
  std::function<mlir::LowerToLLVMOptions(mlir::MLIRContext &)> optsGetter;
};

} // namespace

std::unique_ptr<mlir::Pass> imex::createUtilToLLVMPass(
    std::function<mlir::LowerToLLVMOptions(mlir::MLIRContext &)> optsGetter) {
  assert(optsGetter && "invalid optsGetter");
  return std::make_unique<PlierUtilToLLVMPass>(std::move(optsGetter));
}
