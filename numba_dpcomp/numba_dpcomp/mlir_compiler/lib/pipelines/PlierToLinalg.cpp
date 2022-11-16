// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipelines/PlierToLinalg.hpp"

#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/LoopInvariantCodeMotionUtils.h>
#include <mlir/Transforms/Passes.h>

#include "pipelines/PlierToScf.hpp"
#include "pipelines/PlierToStd.hpp"
#include "pipelines/PreLowSimplifications.hpp"

#include "imex/Compiler/PipelineRegistry.hpp"
#include "imex/Conversion/NtensorToLinalg.hpp"
#include "imex/Conversion/NtensorToMemref.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"
#include "imex/Dialect/ntensor/Transforms/PropagateEnvironment.hpp"
#include "imex/Dialect/ntensor/Transforms/ResolveArrayOps.hpp"
#include "imex/Dialect/plier/Dialect.hpp"
#include "imex/Transforms/CanonicalizeReductions.hpp"
#include "imex/Transforms/CommonOpts.hpp"
#include "imex/Transforms/Cse.hpp"
#include "imex/Transforms/InlineUtils.hpp"
#include "imex/Transforms/LoopUtils.hpp"
#include "imex/Transforms/MakeSignless.hpp"
#include "imex/Transforms/MemoryRewrites.hpp"
#include "imex/Transforms/PipelineUtils.hpp"
#include "imex/Transforms/PromoteBoolMemref.hpp"
#include "imex/Transforms/PromoteToParallel.hpp"
#include "imex/Transforms/RewriteWrapper.hpp"
#include "imex/Transforms/TypeConversion.hpp"
#include "imex/Transforms/UpliftMath.hpp"

#include "BasePipeline.hpp"
#include "LoopUtils.hpp"
#include "Mangle.hpp"
#include "NumpyResolver.hpp"
#include "PyFuncResolver.hpp"
#include "PyLinalgResolver.hpp"

#include <cctype>

namespace {
static int64_t getOptLevel(mlir::Operation *op) {
  assert(op);
  auto attr = op->getAttr(imex::util::attributes::getOptLevelName())
                  .dyn_cast_or_null<mlir::IntegerAttr>();
  if (!attr)
    return 0;

  return std::max(static_cast<int64_t>(0), attr.getInt());
}

static mlir::LogicalResult applyOptimizations(
    mlir::func::FuncOp op, const mlir::FrozenRewritePatternSet &patterns,
    mlir::AnalysisManager am,
    llvm::function_ref<mlir::LogicalResult(mlir::func::FuncOp)> additionalOpts =
        nullptr) {
  bool repeat = false;
  do {
    repeat = false;
    (void)mlir::applyPatternsAndFoldGreedily(op, patterns);
    if (mlir::succeeded(imex::applyCSE(op.getRegion(), false)))
      repeat = true;

    auto memOptRes = imex::optimizeMemoryOps(am);
    if (!memOptRes) {
      op.emitError() << "Failed to build memssa analysis";
      return mlir::failure();
    }
    if (mlir::succeeded(*memOptRes)) {
      repeat = true;
    }

    if (additionalOpts && mlir::succeeded(additionalOpts(op))) {
      repeat = true;
    }
    if (repeat) {
      am.invalidate({});
    }
  } while (repeat);
  return mlir::success();
}

static void rerunScfPipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToScfPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  imex::addPipelineJumpMarker(mod, marker);
}

static mlir::Value skipCasts(mlir::Value val) {
  auto getArg = [](mlir::Value arg) -> mlir::Value {
    auto cast = arg.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!cast)
      return {};

    auto inputs = cast.getInputs();
    if (inputs.size() != 1)
      return {};

    return inputs.front();
  };
  while (auto arg = getArg(val))
    val = arg;

  return val;
};

static mlir::LogicalResult
lowerPrange(plier::PyCallOp op, mlir::ValueRange operands,
            llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
            mlir::PatternRewriter &rewriter) {
  auto parent = op->getParentOp();
  auto setAttr = [](mlir::scf::ForOp op) {
    op->setAttr(imex::util::attributes::getParallelName(),
                mlir::UnitAttr::get(op->getContext()));
  };
  if (mlir::succeeded(
          imex::lowerRange(op, operands, kwargs, rewriter, setAttr))) {
    rerunScfPipeline(parent);
    return mlir::success();
  }
  return mlir::failure();
}

using kwargs_t = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;
using func_t = mlir::LogicalResult (*)(plier::PyCallOp, mlir::ValueRange,
                                       kwargs_t, mlir::PatternRewriter &);
static const std::pair<llvm::StringRef, func_t> builtinFuncsHandlers[] = {
    // clang-format off
    {"numba.prange", lowerPrange},
    // clang-format on
};

static llvm::Optional<mlir::Type> isUniTuple(mlir::TupleType type) {
  auto count = type.size();
  if (count == 0)
    return llvm::None;

  auto elemType = type.getType(0);
  for (auto i : llvm::seq<size_t>(1, count)) {
    if (type.getType(i) != elemType)
      return llvm::None;
  }
  return elemType;
}

static llvm::Optional<mlir::Type> isUniTuple(mlir::Type type) {
  auto tupleType = type.dyn_cast<mlir::TupleType>();
  if (!tupleType)
    return llvm::None;

  return isUniTuple(tupleType);
}

static void genCopy(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value src, mlir::Value dst) {
  auto srcType = src.getType().cast<mlir::ShapedType>();
  auto dstType = dst.getType().cast<mlir::ShapedType>();
  assert(srcType.getRank() == dstType.getRank());
  assert(srcType.getElementType() == dstType.getElementType());
  auto rank = srcType.getRank();

  auto affineMap =
      mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  const mlir::AffineMap maps[] = {
      affineMap,
      affineMap,
  };

  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      rank, mlir::utils::IteratorType::parallel);

  auto bodyBuilder = [](mlir::OpBuilder &b, mlir::Location l,
                        mlir::ValueRange args) {
    assert(args.size() == 2);
    b.create<mlir::linalg::YieldOp>(l, args.front());
  };
  builder.create<mlir::linalg::GenericOp>(loc, src, dst, maps, iterators,
                                          bodyBuilder);
}

struct CleanupLoads : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    auto it = mlir::Block::iterator(op);
    if (it == block->begin())
      return mlir::failure();

    --it;
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(*it);
    if (!store)
      return mlir::failure();

    if (store.getMemref() != op.getMemref() ||
        store.getIndices() != op.getIndices())
      return mlir::failure();

    rewriter.replaceOp(op, store.getValue());
    return mlir::success();
  }
};

struct ReshapeChangeLayout
    : public mlir::OpRewritePattern<mlir::memref::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getSource().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto src = cl.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    auto dstType = op.getSource().getType().cast<mlir::MemRefType>();
    if (srcType.getRank() != dstType.getRank())
      return mlir::failure();

    auto rank = static_cast<unsigned>(dstType.getRank());
    if (rank == 0)
      return mlir::failure();

    int64_t offset;
    llvm::SmallVector<int64_t> strides;
    if (mlir::failed(mlir::getStridesAndOffset(dstType, strides, offset)))
      return mlir::failure();

    auto loc = cl.getLoc();

    llvm::SmallVector<mlir::OpFoldResult> sizesVals(rank);
    for (auto i : llvm::seq(0u, rank))
      sizesVals[i] = rewriter.createOrFold<mlir::memref::DimOp>(loc, src, i);

    int64_t stride = 1;
    llvm::SmallVector<mlir::Value> expectedStrides(rank);
    mlir::Value runningStride =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    for (auto ii = rank; ii-- > 0;) {
      auto i = static_cast<unsigned>(ii);
      expectedStrides[i] = runningStride;

      int64_t size = dstType.getShape()[i];
      if (size == 0)
        continue;

      bool useSizeAsStride = (stride == 1);
      if (size == mlir::ShapedType::kDynamicSize)
        stride = mlir::ShapedType::kDynamicSize;
      if (stride != mlir::ShapedType::kDynamicSize)
        stride *= size;

      auto sizeVal = sizesVals[i].get<mlir::Value>();
      if (useSizeAsStride)
        runningStride = sizeVal;
      else if (stride == mlir::ShapedType::kDynamicSize)
        runningStride =
            rewriter.create<mlir::arith::MulIOp>(loc, runningStride, sizeVal);
      else
        runningStride = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    }

    mlir::OpFoldResult offsetVal = rewriter.getIndexAttr(offset);

    llvm::SmallVector<mlir::OpFoldResult> stridesVals(rank);
    auto offsetConst =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
    auto actualOffset =
        rewriter.createOrFold<imex::util::ExtractMemrefMetadataOp>(loc, src);

    mlir::Value cmp;
    for (auto i : llvm::seq(0u, rank)) {
      if (mlir::ShapedType::isDynamicStrideOrOffset(strides[i])) {
        stridesVals[i] = expectedStrides[i];
      } else {
        stridesVals[i] = rewriter.getIndexAttr(strides[i]);
      }
      auto actualStride =
          rewriter.createOrFold<imex::util::ExtractMemrefMetadataOp>(loc, src,
                                                                     i);

      auto cmpTemp = rewriter.createOrFold<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, expectedStrides[i],
          actualStride);

      if (i == 0) {
        cmp = cmpTemp;
      } else {
        cmp = rewriter.createOrFold<mlir::arith::AndIOp>(loc, cmp, cmpTemp);
      }
    }

    auto trueBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      mlir::Value flat = builder.create<imex::util::MemrefApplyOffsetOp>(
          loc, src.getType(), src);

      mlir::Value res = builder.create<mlir::memref::ReinterpretCastOp>(
          loc, dstType, flat, offsetVal, sizesVals, stridesVals);
      builder.create<mlir::scf::YieldOp>(loc, res);
    };
    auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      llvm::SmallVector<mlir::Value> sizes;
      sizes.reserve(rank);
      auto shape = dstType.getShape();
      for (auto i : llvm::seq(0u, rank))
        if (mlir::ShapedType::isDynamic(shape[i]))
          sizes.emplace_back(sizesVals[i].get<mlir::Value>());

      auto res = builder.create<mlir::memref::AllocOp>(loc, dstType, sizes)
                     .getResult();
      genCopy(rewriter, loc, src, res);
      builder.create<mlir::scf::YieldOp>(loc, res);
    };

    auto res =
        rewriter.create<mlir::scf::IfOp>(loc, dstType, cmp, trueBody, falseBody)
            .getResult(0);
    rewriter.replaceOpWithNewOp<mlir::memref::ReshapeOp>(op, op.getType(), res,
                                                         op.getShape());
    return mlir::success();
  }
};

static constexpr llvm::StringLiteral
    kContigiousArraysAttr("plier.contigious_arrays");

struct MakeStridedLayoutPass
    : public mlir::PassWrapper<MakeStridedLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MakeStridedLayoutPass)

  void runOnOperation() override;
};

void MakeStridedLayoutPass::runOnOperation() {
  auto context = &getContext();
  auto mod = getOperation();

  mlir::OpBuilder builder(mod);
  auto loc = builder.getUnknownLoc();
  auto attrStr = builder.getStringAttr(kContigiousArraysAttr);

  llvm::SmallVector<bool> contigiousArrayArg;

  auto isContigiousArrayArg = [&](unsigned i) {
    if (contigiousArrayArg.empty())
      return false;

    assert(i < contigiousArrayArg.size());
    return contigiousArrayArg[i];
  };

  llvm::SmallVector<mlir::Type> newArgTypes;
  llvm::SmallVector<mlir::Type> newResTypes;
  llvm::SmallVector<mlir::Value> newOperands;
  for (auto func : mod.getOps<mlir::func::FuncOp>()) {
    auto contAttr = func->getAttr(attrStr).dyn_cast_or_null<mlir::ArrayAttr>();
    if (contAttr) {
      auto contAttrRange = contAttr.getAsValueRange<mlir::BoolAttr>();
      contigiousArrayArg.assign(contAttrRange.begin(), contAttrRange.end());
    } else {
      contigiousArrayArg.clear();
    }

    auto funcType = func.getFunctionType();
    auto argTypes = funcType.getInputs();
    auto resTypes = funcType.getResults();
    newArgTypes.assign(argTypes.begin(), argTypes.end());
    newResTypes.assign(resTypes.begin(), resTypes.end());
    auto &body = func.getBody();
    bool hasBody = !body.empty();
    if (hasBody)
      builder.setInsertionPointToStart(&body.front());

    for (auto it : llvm::enumerate(argTypes)) {
      auto i = static_cast<unsigned>(it.index());
      auto type = it.value();
      auto memrefType = type.dyn_cast<mlir::MemRefType>();
      if (!memrefType || isContigiousArrayArg(i) ||
          !memrefType.getLayout().isIdentity())
        continue;

      auto rank = static_cast<unsigned>(memrefType.getRank());
      auto makeShape = [&](int64_t val) {
        return llvm::SmallVector<int64_t>(rank, val);
      };
      auto strideVal = mlir::ShapedType::kDynamicStrideOrOffset;
      auto layout = mlir::StridedLayoutAttr::get(context, strideVal,
                                                 makeShape(strideVal));
      auto newMemrefType =
          mlir::MemRefType::get(makeShape(mlir::ShapedType::kDynamicSize),
                                memrefType.getElementType(), layout);

      if (newMemrefType != memrefType) {
        newArgTypes[i] = newMemrefType;

        if (hasBody) {
          auto arg = body.front().getArgument(i);
          arg.setType(newMemrefType);
          auto dst =
              builder.create<imex::util::ChangeLayoutOp>(loc, memrefType, arg);
          arg.replaceAllUsesExcept(dst, dst);
        }
      }
    }

    for (auto it : llvm::enumerate(resTypes)) {
      auto type = it.value();
      auto memrefType = type.dyn_cast<mlir::MemRefType>();
      if (!memrefType || !memrefType.getLayout().isIdentity())
        continue;

      auto rank = static_cast<unsigned>(memrefType.getRank());
      auto makeShape = [&](int64_t val) {
        return llvm::SmallVector<int64_t>(rank, val);
      };
      auto strideVal = mlir::ShapedType::kDynamicStrideOrOffset;
      auto layout = mlir::StridedLayoutAttr::get(context, strideVal,
                                                 makeShape(strideVal));
      auto newmemrefType =
          mlir::MemRefType::get(makeShape(mlir::ShapedType::kDynamicSize),
                                memrefType.getElementType(), layout);
      newResTypes[it.index()] = newmemrefType;
    }

    auto newFuncType =
        mlir::FunctionType::get(&getContext(), newArgTypes, newResTypes);
    if (newFuncType != funcType) {
      func.setType(newFuncType);
      func.walk([&](mlir::func::ReturnOp ret) {
        builder.setInsertionPoint(ret);
        auto count = static_cast<unsigned>(newResTypes.size());
        for (auto i : llvm::seq(0u, count)) {
          auto arg = ret.getOperand(i);
          auto newType = newResTypes[i];
          if (arg.getType() != newType) {
            assert(arg.getType().isa<mlir::MemRefType>());
            assert(newType.isa<mlir::MemRefType>());
            auto newArg = builder.createOrFold<mlir::memref::CastOp>(
                ret.getLoc(), newType, arg);
            ret.setOperand(i, newArg);
          }
        }
      });
      auto funcUses = mlir::SymbolTable::getSymbolUses(func, mod);
      if (funcUses) {
        for (auto use : *funcUses) {
          auto call = mlir::cast<mlir::func::CallOp>(use.getUser());
          auto loc = call.getLoc();

          builder.setInsertionPoint(call);
          assert(newArgTypes.size() == call.operands().size());
          auto argsCount = static_cast<unsigned>(newArgTypes.size());
          newOperands.resize(argsCount);
          for (auto i : llvm::seq(0u, argsCount)) {
            auto arg = call.operands()[i];
            auto oldType = arg.getType();
            auto newType = newArgTypes[i];
            if (oldType != newType) {
              assert(oldType.isa<mlir::MemRefType>());
              assert(newType.isa<mlir::MemRefType>());
              auto newArg =
                  builder.create<imex::util::ChangeLayoutOp>(loc, newType, arg)
                      .getResult();
              newOperands[i] = newArg;
            } else {
              newOperands[i] = arg;
            }
          }
          call.operandsMutable().assign(newOperands);

          builder.setInsertionPointAfter(call);
          assert(newResTypes.size() == call.getNumResults());
          auto numResults = call.getNumResults();
          for (auto i : llvm::seq(0u, numResults)) {
            auto res = call.getResult(i);
            auto oldType = res.getType();
            auto newType = newResTypes[i];
            if (oldType != newType) {
              assert(oldType.isa<mlir::MemRefType>());
              assert(newType.isa<mlir::MemRefType>());
              res.setType(newType);
              auto newRes =
                  builder.create<imex::util::ChangeLayoutOp>(loc, oldType, res);
              res.replaceAllUsesExcept(newRes, newRes);
            }
          }
        }
      }
    }
  }
}

struct ChangeLayoutReturn
    : public mlir::OpRewritePattern<mlir::func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.operands().empty())
      return mlir::failure();

    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (!func || !func.isPrivate() || !llvm::hasSingleElement(func.getBody()))
      return mlir::failure();

    auto mod = func->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto funcUses = mlir::SymbolTable::getSymbolUses(func, mod);
    if (!funcUses)
      return mlir::failure();

    for (auto use : *funcUses)
      if (!mlir::isa<mlir::func::CallOp>(use.getUser()))
        return mlir::failure();

    auto loc = op->getLoc();
    auto args = op.operands();
    auto count = static_cast<unsigned>(args.size());
    llvm::SmallVector<mlir::Value> newArgs(args.begin(), args.end());
    llvm::SmallVector<int64_t> shape;

    bool changed = false;
    for (auto i : llvm::seq(0u, count)) {
      auto arg = args[i];
      auto retType = arg.getType().dyn_cast<mlir::MemRefType>();
      if (!retType)
        continue;

      auto cast = arg.getDefiningOp<mlir::memref::CastOp>();
      if (!cast)
        continue;

      auto src = cast.getSource();
      auto srcType = src.getType().cast<mlir::MemRefType>();
      assert(srcType.getElementType() == retType.getElementType());

      auto srcLayout = srcType.getLayout();
      auto srcShape = srcType.getShape();
      auto dstShape = retType.getShape();
      assert(srcShape.size() == dstShape.size());
      auto rank = static_cast<unsigned>(srcShape.size());
      shape.resize(rank);
      for (auto j : llvm::seq(0u, rank)) {
        if (!mlir::ShapedType::isDynamic(dstShape[j])) {
          shape[j] = dstShape[j];
        } else if (!mlir::ShapedType::isDynamic(srcShape[j])) {
          shape[j] = srcShape[j];
        } else {
          shape[j] = mlir::ShapedType::kDynamicSize;
        }
      }

      auto newType = mlir::MemRefType::get(shape, srcType.getElementType(),
                                           srcLayout, srcType.getMemorySpace());
      if (newType == retType)
        continue;

      auto newArg = rewriter.create<mlir::memref::CastOp>(loc, newType, src);
      newArgs[i] = newArg;
      changed = true;
    }

    if (!changed)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, newArgs);

    auto newFuncType = [&]() {
      auto origType = func.getFunctionType();
      mlir::ValueRange r(newArgs);
      return mlir::FunctionType::get(getContext(), origType.getInputs(),
                                     r.getTypes());
    }();

    rewriter.updateRootInPlace(func, [&]() {
      func.setFunctionTypeAttr(mlir::TypeAttr::get(newFuncType));
    });

    llvm::SmallVector<mlir::func::CallOp> calls;
    for (auto use : *funcUses) {
      auto call = mlir::cast<mlir::func::CallOp>(use.getUser());
      calls.emplace_back(call);
    }

    for (auto call : calls) {
      rewriter.setInsertionPoint(call);
      auto callLoc = call->getLoc();
      auto oldResults = call.getResults();
      auto newResults =
          rewriter.create<mlir::func::CallOp>(callLoc, func, call.operands())
              .getResults();
      newArgs.assign(newResults.begin(), newResults.end());
      for (auto i : llvm::seq(0u, count)) {
        auto oldType = oldResults[i].getType();
        auto newType = newArgs[i].getType();
        if (oldType != newType)
          newArgs[i] = rewriter.create<mlir::memref::CastOp>(callLoc, oldType,
                                                             newArgs[i]);
      }
      rewriter.replaceOp(call, newArgs);
    }

    return mlir::success();
  }
};

struct OptimizeStridedLayoutPass
    : public mlir::PassWrapper<OptimizeStridedLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeStridedLayoutPass)

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    imex::populateCanonicalizationPatterns(patterns);

    patterns.insert<ChangeLayoutReturn>(context);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct FinalizeStridedLayoutPass
    : public mlir::PassWrapper<FinalizeStridedLayoutPass,
                               mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeStridedLayoutPass)

  void runOnOperation() override;
};

void FinalizeStridedLayoutPass::runOnOperation() {
  auto *context = &getContext();
  auto op = getOperation();
  mlir::RewritePatternSet patterns(context);

  patterns.insert<ReshapeChangeLayout, CleanupLoads>(context);

  (void)mlir::applyPatternsAndFoldGreedily(op, std::move(patterns));

  op->walk([&](imex::util::ChangeLayoutOp cl) {
    cl.emitError("Layout change failed");
    signalPassFailure();
  });
}

static mlir::Value convertScalarType(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value val,
                                     mlir::Type dstType) {
  auto srcType = val.getType();
  if (srcType.isIntOrIndexOrFloat() && dstType.isIntOrIndexOrFloat()) {
    val = builder.create<plier::CastOp>(loc, dstType, val);
    rerunScfPipeline(val.getDefiningOp());
  }

  return val;
}

struct GetitemToNtensor : public mlir::OpConversionPattern<plier::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op, plier::GetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getValue();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto index = adaptor.getIndex();

    rewriter.replaceOpWithNewOp<imex::ntensor::GetitemOp>(op, resultType, src,
                                                          index);
    return mlir::success();
  }
};

struct SetitemToNtensor : public mlir::OpConversionPattern<plier::SetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SetItemOp op, plier::SetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getTarget();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    auto value = convertScalarType(rewriter, op->getLoc(), adaptor.getValue(),
                                   srcType.getElementType());

    rewriter.replaceOpWithNewOp<imex::ntensor::SetitemOp>(op, src, index,
                                                          value);
    return mlir::success();
  }
};

struct NtensorGetitemToNtensor
    : public mlir::OpConversionPattern<imex::ntensor::GetitemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::GetitemOp op,
                  imex::ntensor::GetitemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<imex::ntensor::GetitemOp>(
        op, resultType, adaptor.getSource(), adaptor.getIndex());
    return mlir::success();
  }
};

struct NtensorSetitemToNtensor
    : public mlir::OpConversionPattern<imex::ntensor::SetitemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SetitemOp op,
                  imex::ntensor::SetitemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<imex::ntensor::SetitemOp>(
        op, adaptor.getSource(), adaptor.getIndex(), adaptor.getValue());
    return mlir::success();
  }
};

struct BinopToNtensor : public mlir::OpConversionPattern<plier::BinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BinOp op, plier::BinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    if (!lhs.getType().isa<imex::ntensor::NTensorType>() &&
        !rhs.getType().isa<imex::ntensor::NTensorType>())
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto opName = op.getOp();
    rewriter.replaceOpWithNewOp<imex::ntensor::BinaryOp>(op, resultType, lhs,
                                                         rhs, opName);
    return mlir::success();
  }
};

struct BuildSliceToNtensor
    : public mlir::OpConversionPattern<plier::BuildSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildSliceOp op, plier::BuildSliceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto indexType = rewriter.getIndexType();

    auto loc = op->getLoc();
    auto doCast = [&](mlir::Value val) -> mlir::Value {
      if (val.getType() != indexType) {
        val = rewriter.create<plier::CastOp>(loc, indexType, val);
        rerunScfPipeline(op);
      }

      return val;
    };

    auto isNone = [](mlir::Value val) {
      return val.getType().isa<mlir::NoneType>();
    };

    auto getVal = [&](mlir::Value orig, mlir::Value converted) {
      return isNone(orig) ? mlir::Value() : doCast(converted);
    };

    auto begin = getVal(op.getBegin(), adaptor.getBegin());
    auto end = getVal(op.getEnd(), adaptor.getEnd());
    auto step = getVal(op.getStep(), adaptor.getStep());
    rewriter.replaceOpWithNewOp<imex::ntensor::BuildSliceOp>(op, begin, end,
                                                             step);
    return mlir::success();
  }
};

static bool isBoundFunc(mlir::Type type) {
  if (auto pyType = type.dyn_cast<plier::PyType>()) {
    auto name = pyType.getName().getValue();
    return name.consume_front("BoundFunction(") && name.consume_back(")");
  }
  return false;
}

struct NumpyCallsToNtensor : public mlir::OpConversionPattern<plier::PyCallOp> {
  NumpyCallsToNtensor(mlir::TypeConverter &converter, mlir::MLIRContext *ctx,
                      NumpyResolver &r)
      : OpConversionPattern(converter, ctx), resolver(r) {}

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op, plier::PyCallOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto funcName = op.getFuncName();
    if (!resolver.hasFunc(funcName))
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);

    llvm::SmallVector<mlir::Type> resTypes;
    if (mlir::failed(converter->convertTypes(op->getResultTypes(), resTypes)))
      return mlir::failure();

    auto func = adaptor.getFunc();
    auto getAttr = func.getDefiningOp<plier::GetattrOp>();
    bool isAttr = getAttr && isBoundFunc(func.getType());

    llvm::SmallVector<mlir::Value> args;
    llvm::SmallVector<mlir::Attribute> argNames;
    auto srcArgs = adaptor.getArgs();
    auto srcKwArgs = adaptor.getKwargs();
    auto srcKwNames = adaptor.getKwNames();
    auto totalCount =
        srcArgs.size() + srcKwArgs.size() + static_cast<size_t>(isAttr);
    args.reserve(totalCount);
    argNames.reserve(totalCount);

    auto emptyStrAttr = rewriter.getStringAttr("");
    if (isAttr) {
      auto val = rewriter.getRemappedValue(getAttr.getValue());
      args.emplace_back(val);
      argNames.emplace_back(emptyStrAttr);
    }

    args.append(srcArgs.begin(), srcArgs.end());
    argNames.append(srcArgs.size(), emptyStrAttr);

    args.append(srcKwArgs.begin(), srcKwArgs.end());
    argNames.append(srcKwNames.begin(), srcKwNames.end());

    auto argNamesAttr = rewriter.getArrayAttr(argNames);
    rewriter.replaceOpWithNewOp<imex::ntensor::CallOp>(op, resTypes, args,
                                                       argNamesAttr, funcName);
    return mlir::success();
  }

private:
  NumpyResolver &resolver;
};

struct NumpyAttrsToNtensor
    : public mlir::OpConversionPattern<plier::GetattrOp> {
  NumpyAttrsToNtensor(mlir::TypeConverter &converter, mlir::MLIRContext *ctx,
                      NumpyResolver &r)
      : OpConversionPattern(converter, ctx), resolver(r) {}

  mlir::LogicalResult
  matchAndRewrite(plier::GetattrOp op, plier::GetattrOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getValue();
    if (!src.getType().isa<imex::ntensor::NTensorType>())
      return mlir::failure();

    auto funcName = ("array." + op.getName()).str();
    if (!resolver.hasFunc(funcName))
      return mlir::failure();

    if (isBoundFunc(op.getType())) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    auto converter = getTypeConverter();
    assert(converter);

    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto argNamesAttr = rewriter.getArrayAttr(rewriter.getStringAttr(""));
    rewriter.replaceOpWithNewOp<imex::ntensor::CallOp>(op, resultType, src,
                                                       argNamesAttr, funcName);

    return mlir::success();
  }

private:
  NumpyResolver &resolver;
};

struct BuiltinCallsToNtensor
    : public mlir::OpConversionPattern<plier::PyCallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op, plier::PyCallOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto name = op.getFuncName();
    for (auto &handler : builtinFuncsHandlers)
      if (handler.first == name) {
        auto func = adaptor.getFunc();
        auto args = adaptor.getArgs();
        auto varArgs = adaptor.getVarargs();
        auto kwArgs = adaptor.getKwargs();
        auto kwNames = adaptor.getKwNames();
        assert(kwArgs.size() == kwNames.size() &&
               "Args and names size mismatch");

        auto converter = getTypeConverter();
        assert(converter);

        auto resType = converter->convertType(op.getType());
        if (!resType)
          return mlir::failure();

        rewriter.replaceOpWithNewOp<plier::PyCallOp>(
            op, resType, func, args, varArgs, kwArgs, name, kwNames);
        return mlir::success();
      }

    return mlir::failure();
  }
};

static mlir::Value addElementConversion(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value srcArray,
                                        mlir::Type dstType) {
  auto srcType = srcArray.getType().cast<imex::ntensor::NTensorType>();
  auto dstShaped = dstType.cast<mlir::ShapedType>();
  auto srcElementType = srcType.getElementType();
  auto dstElementType = dstShaped.getElementType();
  if (srcElementType == dstElementType)
    return srcArray;

  auto dstArrayTupe = imex::ntensor::NTensorType::get(
      dstShaped.getShape(), dstElementType, srcType.getEnvironment(),
      srcType.getLayout());

  rerunScfPipeline(srcArray.getParentRegion()->getParentOp());
  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                         mlir::Value val) {
    mlir::Value res = b.create<plier::CastOp>(l, dstElementType, val);
    b.create<imex::ntensor::ElementwiseYieldOp>(l, res);
  };

  return builder.create<imex::ntensor::ElementwiseOp>(loc, dstArrayTupe,
                                                      srcArray, bodyBuilder);
}

static mlir::Value castType(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src, mlir::Type dstType) {
  auto srcType = src.getType();
  if (srcType == dstType)
    return src;

  if (srcType.isa<mlir::MemRefType>())
    return builder.create<mlir::memref::CastOp>(loc, dstType, src);

  if (srcType.isa<mlir::RankedTensorType>())
    return builder.create<mlir::tensor::CastOp>(loc, dstType, src);

  if (srcType.isa<imex::ntensor::NTensorType>())
    return builder.create<imex::ntensor::CastOp>(loc, dstType, src);

  llvm_unreachable("Invalid shaped type");
}

static llvm::Optional<mlir::Value> doCast(mlir::OpBuilder &builder,
                                          mlir::Location loc, mlir::Value src,
                                          mlir::Type dstType) {
  auto srcType = src.getType();
  if (srcType == dstType)
    return src;

  if (auto srcArrayType = srcType.dyn_cast<imex::ntensor::NTensorType>()) {
    auto dstShapedType = dstType.dyn_cast<mlir::ShapedType>();
    if (!dstShapedType)
      return llvm::None;

    mlir::Value res = addElementConversion(builder, loc, src, dstShapedType);
    if (dstShapedType.isa<mlir::MemRefType>()) {
      auto dstMemrefType = mlir::MemRefType::get(
          srcArrayType.getShape(), dstShapedType.getElementType());
      res = builder.create<imex::ntensor::ToMemrefOp>(loc, dstMemrefType, res);
    } else if (dstShapedType.isa<mlir::RankedTensorType>()) {
      auto dstTensorType = mlir::RankedTensorType::get(
          srcArrayType.getShape(), dstShapedType.getElementType());
      res = builder.create<imex::ntensor::ToTensorOp>(loc, dstTensorType, res);
    }

    return castType(builder, loc, res, dstShapedType);
  } else {
    auto srcShapedType = srcType.dyn_cast<mlir::ShapedType>();
    if (!srcShapedType)
      return llvm::None;

    auto dstArrayType = dstType.dyn_cast<imex::ntensor::NTensorType>();
    if (!dstArrayType)
      return llvm::None;

    srcArrayType = imex::ntensor::NTensorType::get(
        dstArrayType.getShape(), srcShapedType.getElementType(),
        dstArrayType.getEnvironment(), dstArrayType.getLayout());

    mlir::Value res;
    if (srcShapedType.isa<mlir::MemRefType>()) {
      auto dstMemrefType = mlir::MemRefType::get(
          dstArrayType.getShape(), srcShapedType.getElementType());
      src = castType(builder, loc, src, dstMemrefType);
      res = builder.create<imex::ntensor::FromMemrefOp>(loc, srcArrayType, src);
    } else if (srcShapedType.isa<mlir::RankedTensorType>()) {
      auto dstTensorType = mlir::RankedTensorType::get(
          dstArrayType.getShape(), srcShapedType.getElementType());
      src = castType(builder, loc, src, dstTensorType);
      res = builder.create<imex::ntensor::FromTensorOp>(loc, srcArrayType, src);
    }
    assert(res && "Expected tensor or memref type.");
    return addElementConversion(builder, loc, res, dstArrayType);
  }
}

struct CastsToNtensor : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getValue();
    auto srcType = src.getType();

    auto converter = getTypeConverter();
    assert(converter);

    auto dstType = converter->convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    if (imex::ntensor::NTensorType::isValidElementType(srcType) &&
        dstType.isa<imex::ntensor::NTensorType>()) {
      auto ntensorType = dstType.cast<imex::ntensor::NTensorType>();
      if (srcType != ntensorType.getElementType() ||
          !ntensorType.hasStaticShape())
        return mlir::failure();

      rewriter.replaceOpWithNewOp<imex::ntensor::CreateArrayOp>(
          op, ntensorType, /*dynamicSizes*/ llvm::None, src);
      return mlir::success();
    }

    if (auto res = doCast(rewriter, op->getLoc(), src, dstType)) {
      rewriter.replaceOp(op, *res);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct UnitupleExtractToNtensor
    : public mlir::OpConversionPattern<imex::util::TupleExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::TupleExtractOp op,
                  imex::util::TupleExtractOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSource();
    auto elemType = isUniTuple(src.getType());
    if (!elemType || !imex::ntensor::NTensorType::isValidElementType(*elemType))
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);

    auto dstType = converter->convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    rewriter.replaceOpWithNewOp<imex::ntensor::GetitemOp>(op, dstType, src,
                                                          index);
    return mlir::success();
  }
};

static bool isNtensor(mlir::TypeConverter &converter, mlir::Type type) {
  return !!converter.convertType(type)
               .dyn_cast_or_null<imex::ntensor::NTensorType>();
}

struct PlierToNtensorPass
    : public mlir::PassWrapper<PlierToNtensorPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlierToNtensorPass)

  PlierToNtensorPass()
      : resolver(std::make_shared<NumpyResolver>(
            "numba_dpcomp.mlir.numpy.funcs", "_get_func")) {}

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<plier::PlierDialect>();
  }

  void runOnOperation() override {
    auto &context = getContext();

    mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](mlir::Type type) { return type; });

    imex::populateTupleTypeConverter(typeConverter);
    typeConverter.addConversion(
        [](plier::SliceType type) -> llvm::Optional<mlir::Type> {
          return imex::ntensor::SliceType::get(type.getContext());
        });

    auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs, mlir::Location loc) {
      auto cast =
          builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
      return llvm::Optional<mlir::Value>(cast.getResult(0));
    };
    typeConverter.addArgumentMaterialization(addUnrealizedCast);
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);

    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    imex::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                             patterns, target);
    imex::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                       target);

    target.addDynamicallyLegalOp<plier::GetItemOp>(
        [&typeConverter](plier::GetItemOp op) -> llvm::Optional<bool> {
          auto containerType = op.getValue().getType();
          if (isNtensor(typeConverter, containerType))
            return false;

          return llvm::None;
        });
    target.addDynamicallyLegalOp<plier::SetItemOp>(
        [&typeConverter](plier::SetItemOp op) -> llvm::Optional<bool> {
          auto containerType = op.getTarget().getType();
          if (isNtensor(typeConverter, containerType))
            return false;

          return llvm::None;
        });

    target.addDynamicallyLegalOp<imex::ntensor::GetitemOp,
                                 imex::ntensor::SetitemOp>(
        [&typeConverter](mlir::Operation *op) {
          return typeConverter.isLegal(op);
        });

    target.addDynamicallyLegalOp<plier::BinOp>(
        [&typeConverter](plier::BinOp op) -> llvm::Optional<bool> {
          auto lhs = op.getLhs().getType();
          auto rhs = op.getRhs().getType();
          if (isNtensor(typeConverter, lhs) || isNtensor(typeConverter, rhs))
            return false;

          return llvm::None;
        });

    target.addDynamicallyLegalOp<plier::PyCallOp>(
        [this, &typeConverter](plier::PyCallOp op) -> llvm::Optional<bool> {
          auto funcName = op.getFuncName();
          if (resolver->hasFunc(funcName))
            return false;

          for (auto &handler : builtinFuncsHandlers)
            if (handler.first == funcName)
              return typeConverter.isLegal(op);

          return llvm::None;
        });

    target.addDynamicallyLegalOp<plier::GetattrOp>(
        [&typeConverter](plier::GetattrOp op) -> llvm::Optional<bool> {
          auto containerType = op.getValue().getType();
          if (isNtensor(typeConverter, containerType))
            return false;

          return llvm::None;
        });

    target.addDynamicallyLegalOp<plier::CastOp>(
        [&typeConverter](plier::CastOp op) -> llvm::Optional<bool> {
          auto srcType = op.getValue().getType();
          auto dstType = op.getType();
          if (isNtensor(typeConverter, srcType) ||
              isNtensor(typeConverter, dstType))
            return false;

          return true;
        });

    target.addDynamicallyLegalOp<imex::util::TupleExtractOp>(
        [](imex::util::TupleExtractOp op) -> llvm::Optional<bool> {
          if (auto elemType = isUniTuple(op.getSource().getType()))
            if (imex::ntensor::NTensorType::isValidElementType(*elemType))
              return false;

          return llvm::None;
        });

    target.addIllegalOp<plier::BuildSliceOp>();

    target.addLegalDialect<imex::ntensor::NTensorDialect>();

    patterns.insert<
        // clang-format off
        GetitemToNtensor,
        SetitemToNtensor,
        NtensorGetitemToNtensor,
        NtensorSetitemToNtensor,
        BinopToNtensor,
        BuildSliceToNtensor,
        BuiltinCallsToNtensor,
        CastsToNtensor,
        UnitupleExtractToNtensor
        // clang-format on
        >(typeConverter, &context);

    patterns.insert<NumpyCallsToNtensor, NumpyAttrsToNtensor>(
        typeConverter, &context, *resolver);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }

private:
  std::shared_ptr<NumpyResolver> resolver;
};

struct GetitemArrayOpLowering
    : public mlir::OpRewritePattern<imex::ntensor::GetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::GetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    if (!src.getType().isa<imex::ntensor::NTensorType>())
      return mlir::failure();

    auto index = op.getIndex();
    if (!index.getType().isa<imex::ntensor::NTensorType>())
      return mlir::failure();

    mlir::StringRef opName = "array.__getitem__";
    auto resType = op.getType();
    auto args = {src, index};
    rewriter.replaceOpWithNewOp<imex::ntensor::PrimitiveOp>(op, resType, args,
                                                            opName);
    return mlir::success();
  }
};

static PyLinalgResolver::Values
castRetTypes(mlir::Location loc, mlir::PatternRewriter &rewriter,
             mlir::Operation *op,
             llvm::Optional<PyLinalgResolver::Values> vals) {
  auto results = std::move(vals).value();
  assert(results.size() == op->getNumResults());
  for (auto it : llvm::enumerate(results)) {
    auto i = it.index();
    auto ret = it.value();
    auto dstType = op->getResultTypes()[i];

    auto srcType = ret.getType();
    if (dstType != srcType)
      results[i] = rewriter.create<plier::CastOp>(loc, dstType, ret);
  }
  return results;
}

static mlir::FailureOr<mlir::Attribute> getEnvAttr(mlir::Operation *op) {
  assert(op && "Invalid op");

  mlir::Attribute env;
  for (auto types : {mlir::TypeRange(op->getOperandTypes()),
                     mlir::TypeRange(op->getResultTypes())}) {
    for (auto type : types) {
      auto tensor = type.dyn_cast<imex::ntensor::NTensorType>();
      if (!tensor)
        continue;

      if (!env) {
        env = tensor.getEnvironment();
      } else if (env != tensor.getEnvironment()) {
        return mlir::failure();
      }
    }
  }

  return env;
}

struct NtensorPrimitiveCallsLowering final
    : public mlir::OpRewritePattern<imex::ntensor::PrimitiveOp> {
  NtensorPrimitiveCallsLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context),
        resolver("numba_dpcomp.mlir.numpy.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::PrimitiveOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto env = getEnvAttr(op);
    if (mlir::failed(env))
      return mlir::failure();

    auto opName = op.getOp();

    auto loc = op->getLoc();
    auto getRes = [&]() -> llvm::Optional<PyLinalgResolver::Values> {
      auto args = op.getArgs();
      auto funcRes =
          resolver.rewriteFunc(opName, loc, rewriter, args, llvm::None);
      if (funcRes)
        return funcRes;

      if (opName.startswith("array.") && args.size() == 1)
        return resolver.rewriteAttr(opName, loc, rewriter, args.front());

      return llvm::None;
    };

    PyLinalgResolver::Values newRes;
    if (*env != nullptr) {
      auto regionOp = rewriter.create<imex::util::EnvironmentRegionOp>(
          loc, *env, /*args*/ llvm::None, op->getResultTypes());
      auto &newBody = regionOp.getRegion().front();
      rewriter.eraseOp(newBody.getTerminator());

      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&newBody);
      auto res = getRes();
      if (!res) {
        rewriter.eraseOp(regionOp);
        return mlir::failure();
      }

      auto results = castRetTypes(loc, rewriter, op, *res);
      rewriter.create<imex::util::EnvironmentRegionYieldOp>(loc, results);

      auto regResults = regionOp.getResults();
      newRes.assign(regResults.begin(), regResults.end());
    } else {
      auto res = getRes();
      if (!res)
        return mlir::failure();

      auto results = castRetTypes(loc, rewriter, op, *res);
      newRes.assign(results.begin(), results.end());
    }

    rerunScfPipeline(op);
    rewriter.replaceOp(op, newRes);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct NumpyCallsResolver
    : public mlir::OpRewritePattern<imex::ntensor::CallOp> {
  NumpyCallsResolver(mlir::MLIRContext *ctx, NumpyResolver &r)
      : OpRewritePattern(ctx), resolver(r) {}

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op.getOp();

    auto loc = op.getLoc();
    llvm::SmallVector<mlir::Value> args;
    llvm::SmallVector<mlir::Value> outResults;
    if (mlir::failed(resolver.resolveFuncArgs(rewriter, loc, funcName,
                                              op.getArgs(), op.getArgsNames(),
                                              args, outResults)))
      return mlir::failure();

    mlir::ValueRange results =
        rewriter
            .create<imex::ntensor::PrimitiveOp>(loc, op->getResultTypes(), args,
                                                funcName)
            .getResults();

    for (auto [dst, src] : llvm::zip(outResults, results))
      rewriter.create<imex::ntensor::CopyOp>(loc, src, dst);

    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  NumpyResolver &resolver;
};

struct ExternalCallsResolver final : mlir::OpRewritePattern<plier::PyCallOp> {
  using OpRewritePattern::OpRewritePattern;

protected:
  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getKwargs().empty())
      return mlir::failure(); // TODO: kwargs support

    auto name = op.getFuncName();
    auto args = op.getArgs();

    llvm::SmallVector<mlir::Type> types(args.size());
    for (auto [i, arg] : llvm::enumerate(args)) {
      arg = skipCasts(arg);
      types[i] = arg.getType();
    }

    auto mangledName = mangle(name, types);
    if (mangledName.empty())
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto externalFunc = mod.lookupSymbol<mlir::func::FuncOp>(mangledName);
    if (!externalFunc) {
      externalFunc = resolver.getFunc(name, types);
      if (externalFunc) {
        externalFunc.setPrivate();
        externalFunc.setName(mangledName);
      }
    }
    if (!externalFunc)
      return mlir::failure();

    assert(externalFunc.getFunctionType().getNumResults() ==
           op->getNumResults());

    auto loc = op->getLoc();

    llvm::SmallVector<mlir::Value> castedArgs(args.size());
    auto funcTypes = externalFunc.getFunctionType().getInputs();
    for (auto it : llvm::enumerate(args)) {
      auto arg = it.value();
      auto i = it.index();
      auto dstType = funcTypes[i];
      if (arg.getType() != dstType)
        castedArgs[i] = rewriter.createOrFold<plier::CastOp>(loc, dstType, arg);
      else
        castedArgs[i] = arg;
    }

    auto newFuncCall =
        rewriter.create<mlir::func::CallOp>(loc, externalFunc, castedArgs);

    auto results = newFuncCall.getResults();
    llvm::SmallVector<mlir::Value> castedResults(results.size());

    for (auto it : llvm::enumerate(results)) {
      auto i = static_cast<unsigned>(it.index());
      auto res = it.value();
      auto oldResType = op->getResult(i).getType();
      if (res.getType() != oldResType)
        castedResults[i] =
            rewriter.createOrFold<plier::CastOp>(loc, oldResType, res);
      else
        castedResults[i] = res;
    }

    rerunScfPipeline(op);
    rewriter.replaceOp(op, castedResults);
    return mlir::success();
  }

private:
  PyFuncResolver resolver;
};

struct BuiltinCallsLowering : public mlir::OpRewritePattern<plier::PyCallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto name = op.getFuncName();
    for (auto &handler : builtinFuncsHandlers)
      if (handler.first == name) {
        auto args = op.getArgs();
        auto kwArgs = op.getKwargs();
        auto kwNames = op.getKwNames();
        assert(kwArgs.size() == kwNames.size() &&
               "Args and names size mismatch");
        llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>> kwArgsArray;
        kwArgsArray.reserve(kwArgs.size());
        for (auto [arg, nameAttr] : llvm::zip(kwArgs, kwNames)) {
          auto argName = nameAttr.cast<mlir::StringAttr>().getValue();
          kwArgsArray.emplace_back(argName, arg);
        }

        return handler.second(op, args, kwArgsArray, rewriter);
      }

    return mlir::failure();
  }
};

struct BinOpsLowering : public mlir::OpRewritePattern<imex::ntensor::BinaryOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::BinaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto name = op.getOp();
    for (auto it : plier::getOperators()) {
      if (it.op == name) {
        auto newName = (llvm::Twine("operator.") + it.name).str();
        mlir::Value args[] = {op.getLhs(), op.getRhs()};
        rewriter.replaceOpWithNewOp<imex::ntensor::PrimitiveOp>(
            op, op->getResultTypes(), args, newName);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct ResolveNtensorPass
    : public mlir::PassWrapper<ResolveNtensorPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveNtensorPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    imex::ntensor::populateResolveArrayOpsPatterns(patterns);

    patterns
        .insert<GetitemArrayOpLowering, NtensorPrimitiveCallsLowering,
                BuiltinCallsLowering, BinOpsLowering, ExternalCallsResolver>(
            &ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct WrapParforRegions
    : public mlir::PassWrapper<WrapParforRegions, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapParforRegions)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();

    auto getOpEnv = [](mlir::Operation *op) -> llvm::Optional<mlir::Attribute> {
      if (auto load = mlir::dyn_cast<imex::ntensor::LoadOp>(op))
        return load.getArray().getType().getEnvironment();

      if (auto store = mlir::dyn_cast<imex::ntensor::LoadOp>(op))
        return store.getArray().getType().getEnvironment();

      return llvm::None;
    };

    mlir::OpBuilder builder(&getContext());
    auto attrName =
        builder.getStringAttr(imex::util::attributes::getParallelName());
    llvm::SmallVector<std::pair<mlir::scf::ForOp, mlir::Attribute>>
        opsToProcess;

    auto visitor = [&](mlir::scf::ForOp forOp) -> mlir::WalkResult {
      if (!forOp->hasAttr(attrName))
        return mlir::WalkResult::advance();

      llvm::Optional<mlir::Attribute> env;
      auto innerVisitor = [&](mlir::Operation *innerOp) -> mlir::WalkResult {
        auto opEnv = getOpEnv(innerOp);
        if (!opEnv)
          return mlir::WalkResult::advance();

        if (!env) {
          env = *opEnv;
        } else if (*env != *opEnv) {
          forOp->emitError("Incompatible envs: ") << *env << " and " << *opEnv;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      };
      if (forOp->walk(innerVisitor).wasInterrupted())
        return mlir::WalkResult::interrupt();

      if (env && *env)
        opsToProcess.emplace_back(forOp, *env);

      return mlir::WalkResult::advance();
    };
    if (op->walk(visitor).wasInterrupted())
      return signalPassFailure();

    if (opsToProcess.empty())
      return markAllAnalysesPreserved();

    for (auto [forOp, env] : opsToProcess) {
      auto resultTypes = forOp.getResultTypes();
      builder.setInsertionPoint(forOp);
      auto envRegion = builder.create<imex::util::EnvironmentRegionOp>(
          forOp.getLoc(), env, /*args*/ llvm::None, resultTypes);
      auto &envRegionBlock = envRegion.getRegion().front();
      auto term = envRegionBlock.getTerminator();
      forOp->moveBefore(term);
      forOp->replaceAllUsesWith(envRegion.getResults());
      builder.setInsertionPoint(term);
      builder.create<imex::util::EnvironmentRegionYieldOp>(term->getLoc(),
                                                           forOp.getResults());
      term->erase();
    }
  }
};

struct ResolveNumpyFuncsPass
    : public mlir::PassWrapper<ResolveNumpyFuncsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveNumpyFuncsPass)

  ResolveNumpyFuncsPass()
      : resolver(std::make_shared<NumpyResolver>(
            "numba_dpcomp.mlir.numpy.funcs", "_get_func")) {}

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<NumpyCallsResolver>(&ctx, *resolver);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }

private:
  std::shared_ptr<NumpyResolver> resolver;
};

struct SimplifyExpandDims
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using mlir::OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics())
      return mlir::failure();

    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return mlir::failure();

    auto context = op.getContext();
    auto parallelAttr = mlir::linalg::IteratorTypeAttr::get(
        context, mlir::utils::IteratorType::parallel);

    if (llvm::any_of(op.getIteratorTypes(),
                     [&](auto attr) { return attr != parallelAttr; }))
      return mlir::failure();

    auto maps = op.getIndexingMaps();
    assert(maps.size() == 2);
    auto outMap = maps[1].cast<mlir::AffineMapAttr>().getValue();
    if (!outMap.isIdentity())
      return mlir::failure();

    auto inMap = maps[0].cast<mlir::AffineMapAttr>().getValue();
    auto numDims = op.getNumLoops();
    if (inMap.getNumResults() != numDims)
      return mlir::failure();

    bool changed = false;
    auto outShape =
        op.getOutputs()[0].getType().cast<mlir::RankedTensorType>().getShape();
    llvm::SmallVector<mlir::AffineExpr> exprs(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      auto prevExpr = inMap.getResult(i);
      bool canConvert = [&]() {
        if (outShape[i] == 1) {
          auto constExpr = prevExpr.dyn_cast<mlir::AffineConstantExpr>();
          if (constExpr && constExpr.getValue() == 0)
            return true;
        }
        return false;
      }();
      if (canConvert) {
        changed = true;
        exprs[i] = mlir::getAffineDimExpr(i, context);
      } else {
        exprs[i] = prevExpr;
      }
    }

    if (changed) {
      const mlir::Attribute newMaps[] = {
          mlir::AffineMapAttr::get(
              mlir::AffineMap::get(numDims, 0, exprs, context)),
          maps[1]};
      auto newMapsAttr = mlir::ArrayAttr::get(context, newMaps);
      rewriter.updateRootInPlace(
          op, [&]() { op.setIndexingMapsAttr(newMapsAttr); });
    }

    return mlir::success(changed);
  }
};

struct LowerEnforceShape
    : public mlir::OpRewritePattern<imex::util::EnforceShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::EnforceShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto src = op.getValue();
    rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, type, src);
    return mlir::success();
  }
};

struct InsertSliceToPad
    : public mlir::OpRewritePattern<mlir::tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto gen = op.getDest().getDefiningOp<mlir::tensor::GenerateOp>();
    if (!gen)
      return mlir::failure();

    for (auto stride : op.getMixedStrides()) {
      auto val = mlir::getConstantIntValue(stride);
      if (!val || *val != 1)
        return mlir::failure();
    }

    auto src = op.getSource();
    auto srcType = src.getType().cast<mlir::RankedTensorType>();
    auto dstType = gen.getType().cast<mlir::RankedTensorType>();

    auto rank = static_cast<unsigned>(srcType.getRank());

    auto low = op.getMixedOffsets();
    llvm::SmallVector<mlir::OpFoldResult> high(rank);

    auto loc = op->getLoc();

    auto toVal = [&](mlir::OpFoldResult val) -> mlir::Value {
      if (val.is<mlir::Value>())
        return val.get<mlir::Value>();

      return rewriter.create<mlir::arith::ConstantOp>(
          loc, val.get<mlir::Attribute>());
    };

    for (auto i : llvm::seq(0u, rank)) {
      auto dstDim = rewriter.createOrFold<mlir::tensor::DimOp>(loc, gen, i);
      auto srcDim = rewriter.createOrFold<mlir::tensor::DimOp>(loc, src, i);
      auto offset = rewriter.createOrFold<mlir::arith::AddIOp>(
          loc, toVal(srcDim), toVal(low[i]));
      offset = rewriter.createOrFold<mlir::arith::SubIOp>(loc, toVal(dstDim),
                                                          offset);
      high[i] = mlir::getAsOpFoldResult(offset);
    }

    auto pad =
        rewriter.create<mlir::tensor::PadOp>(loc, dstType, src, low, high);
    rewriter.cloneRegionBefore(gen.getRegion(), pad.getRegion(),
                               pad.getRegion().end());
    rewriter.replaceOp(op, pad.getResult());
    return mlir::success();
  }
};

struct GenerateToFill
    : public mlir::OpRewritePattern<mlir::tensor::GenerateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::GenerateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto &body = op.getBody();
    if (!llvm::hasSingleElement(body))
      return mlir::failure();

    auto &block = body.getBlocks().front();
    if (!llvm::hasSingleElement(block))
      return mlir::failure();

    auto term = mlir::cast<mlir::tensor::YieldOp>(block.getTerminator());
    auto resType = op.getType().cast<mlir::ShapedType>();

    auto loc = op->getLoc();
    mlir::Value init = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resType.getShape(), resType.getElementType(),
        op.getDynamicExtents());

    rewriter.replaceOpWithNewOp<mlir::linalg::FillOp>(op, term.getValue(),
                                                      init);
    return mlir::success();
  }
};

struct SliceOfGeneric : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics() || op.hasIndexSemantics())
      return mlir::failure();

    if (op->getNumResults() != 1)
      return mlir::failure();

    auto res = op->getResult(0);
    if (!res.hasOneUse())
      return mlir::failure();

    mlir::Operation *user = *(res.getUsers().begin());
    if (!mlir::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::ExtractOp>(user))
      return mlir::failure();

    mlir::DominanceInfo dom;
    for (auto arg : user->getOperands())
      if (!dom.dominates(arg, op))
        return mlir::failure();

    auto output = op.getOutputs().front();

    auto resType = res.getType().cast<mlir::RankedTensorType>();
    auto resRank = static_cast<unsigned>(resType.getRank());
    auto maps = [&]() {
      auto mapsList =
          op.getIndexingMaps().getAsValueRange<mlir::AffineMapAttr>();
      return llvm::SmallVector<mlir::AffineMap>(mapsList.begin(),
                                                mapsList.end());
    }();
    assert(!maps.empty());
    for (auto m : maps)
      if (!m.isProjectedPermutation())
        return mlir::failure();

    auto resMap = maps.back();

    auto iters = op.getIteratorTypes();
    auto parallelIter = mlir::linalg::IteratorTypeAttr::get(
        rewriter.getContext(), mlir::utils::IteratorType::parallel);
    for (auto i : llvm::seq(0u, resRank)) {
      auto dim = resMap.getDimPosition(i);
      assert(dim < iters.size());
      if (iters[dim] != parallelIter)
        return mlir::failure();
    }

    bool extractElem = false;
    llvm::SmallBitVector droppedDims;
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;

    auto zero = rewriter.getIndexAttr(0);
    auto one = rewriter.getIndexAttr(1);

    auto assignArr = [](llvm::SmallVectorImpl<mlir::OpFoldResult> &arr,
                        const auto &range) {
      arr.reserve(range.size());
      arr.assign(range.begin(), range.end());
    };

    if (auto sliceOp = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
      offsets = sliceOp.getMixedOffsets();
      sizes = sliceOp.getMixedSizes();
      strides = sliceOp.getMixedStrides();
      droppedDims = sliceOp.getDroppedDims();
    } else if (auto extractOp = mlir::dyn_cast<mlir::tensor::ExtractOp>(user)) {
      if (extractOp.getIndices().empty())
        return mlir::failure();

      extractElem = true;
      assignArr(offsets, extractOp.getIndices());
      sizes.resize(offsets.size(), one);
      strides.resize(offsets.size(), one);
      droppedDims.resize(offsets.size(), true);
    } else {
      llvm_unreachable("Invalid op");
    }

    auto oldInputs = op.getInputs();
    llvm::SmallVector<mlir::Value, 4> newInputs(oldInputs.size());

    auto ctx = getContext();
    auto replaceAffineDim = [&](mlir::AffineExpr expr, unsigned srcDim,
                                unsigned dstDim) {
      auto src = mlir::getAffineDimExpr(srcDim, ctx);
      auto dst = mlir::getAffineDimExpr(dstDim, ctx);
      return expr.replace(src, dst);
    };
    auto findResDim = [&](unsigned inputDim) -> llvm::Optional<unsigned> {
      for (auto d : llvm::seq(0u, resRank)) {
        if (resMap.getDimPosition(d) == inputDim)
          return d;
      }
      return llvm::None;
    };
    auto isDroppedDim = [&](unsigned d) -> bool {
      if (auto indVal = findResDim(d)) {
        auto ind = *indVal;
        assert(ind < droppedDims.size());
        return droppedDims[ind];
      }
      return false;
    };

    auto numLoops = static_cast<unsigned>(iters.size());
    auto ErasedLoop = static_cast<unsigned>(-1);
    llvm::SmallVector<unsigned, 4> loopsMapping(numLoops, ErasedLoop);
    llvm::SmallVector<mlir::utils::IteratorType, 4> newIters;
    newIters.reserve(numLoops);
    for (auto d : llvm::seq(0u, numLoops)) {
      if (!isDroppedDim(d)) {
        auto i = newIters.size();
        assert(i != ErasedLoop);
        newIters.emplace_back(
            iters[d].cast<mlir::linalg::IteratorTypeAttr>().getValue());
        loopsMapping[d] = i;
      }
    }
    auto finalNumLoops = static_cast<unsigned>(newIters.size());

    llvm::SmallVector<mlir::AffineExpr, 4> tempExprs;
    tempExprs.reserve(numLoops);

    auto updateMap = [&](mlir::AffineMap srcMap) -> mlir::AffineMap {
      if (finalNumLoops == numLoops)
        return srcMap;

      tempExprs.clear();
      auto mapResults = srcMap.getResults();
      for (auto i : llvm::seq<size_t>(0, mapResults.size())) {
        auto origLoop = srcMap.getDimPosition(i);
        assert(origLoop < loopsMapping.size());
        auto newLoop = loopsMapping[origLoop];
        if (newLoop != ErasedLoop) {
          auto expr = mapResults[i];
          tempExprs.emplace_back(replaceAffineDim(expr, origLoop, newLoop));
        }
      }
      return mlir::AffineMap::get(finalNumLoops, 0, tempExprs, ctx);
    };

    maps.back() = updateMap(resMap);

    auto loc = op->getLoc();
    llvm::SmallVector<mlir::OpFoldResult, 4> tempOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> tempSizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> tempStrides;
    for (auto i : llvm::seq<size_t>(0, oldInputs.size())) {
      assert(i < maps.size());
      auto input = oldInputs[i];
      auto inputType = input.getType().cast<mlir::RankedTensorType>();
      auto inputRank = static_cast<unsigned>(inputType.getRank());
      auto inputMap = maps[i];

      bool needView = false;
      tempOffsets.resize(inputRank);
      tempSizes.resize(inputRank);
      tempStrides.resize(inputRank);

      unsigned inputResultRank = 0;
      for (auto r : llvm::seq(0u, inputRank)) {
        auto inputDim = inputMap.getDimPosition(r);
        if (auto indVal = findResDim(inputDim)) {
          auto ind = *indVal;
          tempOffsets[r] = offsets[ind];
          tempSizes[r] = sizes[ind];
          tempStrides[r] = strides[ind];
          needView = true;
          assert(ind < droppedDims.size());
          if (!droppedDims[ind])
            ++inputResultRank;
        } else {
          tempOffsets[r] = zero;
          tempSizes[r] =
              rewriter.createOrFold<mlir::tensor::DimOp>(loc, input, r);
          tempStrides[r] = one;
          ++inputResultRank;
        }
      }

      if (needView) {
        mlir::RankedTensorType viewType;
        if (inputResultRank < inputRank) {
          viewType =
              mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  inputResultRank, inputType, tempOffsets, tempSizes,
                  tempStrides);
        } else {
          viewType = mlir::tensor::ExtractSliceOp::inferResultType(
              inputType, tempOffsets, tempSizes, tempStrides);
        }
        newInputs[i] = rewriter.createOrFold<mlir::tensor::ExtractSliceOp>(
            loc, viewType, input, tempOffsets, tempSizes, tempStrides);
      } else {
        newInputs[i] = input;
      }

      maps[i] = updateMap(inputMap);
    }

    auto outputType = output.getType().cast<mlir::RankedTensorType>();
    mlir::RankedTensorType newInitType;
    if (droppedDims.any()) {
      auto initRank = droppedDims.size() - droppedDims.count();
      newInitType =
          mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
              initRank, outputType, offsets, sizes, strides);
    } else {
      newInitType = mlir::tensor::ExtractSliceOp::inferResultType(
          outputType, offsets, sizes, strides);
    }

    mlir::Value newInit = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, newInitType, output, offsets, sizes, strides);

    auto newOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, newInit.getType(), newInputs, newInit, maps, newIters);
    auto &newRegion = newOp.getRegion();

    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());

    mlir::Value result = newOp.getResult(0);
    if (extractElem)
      result = rewriter.create<mlir::tensor::ExtractOp>(loc, result);

    rewriter.replaceOp(user, result);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct OptimizeGlobalsConstsLoad
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // We access data outside function, but doesnt change it, lets hope it
    // is safe.
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod) {
      return mlir::failure();
    }
    mlir::SymbolTable symbolTable(mod);

    llvm::SmallVector<uint64_t> indices(op.getIndices().size());
    for (auto it : llvm::enumerate(op.getIndices())) {
      auto constIndex =
          it.value().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!constIndex)
        return mlir::failure();

      auto val = constIndex.value();
      if (val < 0)
        return mlir::failure();

      indices[it.index()] = static_cast<uint64_t>(val);
    }
    auto getGlobal = op.getMemref().getDefiningOp<mlir::memref::GetGlobalOp>();
    if (!getGlobal)
      return mlir::failure();

    auto sym = symbolTable.lookup<mlir::memref::GlobalOp>(getGlobal.getName());
    if (!sym)
      return mlir::failure();

    if (!sym.getConstant())
      return mlir::failure();

    auto initAttr = sym.getInitialValue();
    if (!initAttr)
      return mlir::failure();

    auto elements = initAttr->dyn_cast<mlir::ElementsAttr>();
    if (!elements)
      return mlir::failure();

    if (elements.getType().getElementType() != op.getType() ||
        !elements.isValidIndex(indices))
      return mlir::failure();

    auto vals = elements.tryGetValues<mlir::Attribute>();
    if (!vals)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, (*vals)[indices]);
    return mlir::success();
  }
};

struct OptimizeSingleElemCopy
    : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto dst = op.getTarget();
    auto dstType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    if (srcType.getRank() != 1 ||
        (srcType.getShape()[0] != 1 && dstType.getShape()[0] != 1))
      return mlir::failure();

    auto loc = op->getLoc();
    mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value val = rewriter.create<mlir::memref::LoadOp>(loc, src, idx);
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, val, dst, idx);
    return mlir::success();
  }
};

struct PostPlierToLinalgPass
    : public mlir::PassWrapper<PostPlierToLinalgPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostPlierToLinalgPass)

  void runOnOperation() override;
};

void PostPlierToLinalgPass::runOnOperation() {
  auto &context = getContext();
  mlir::RewritePatternSet patterns(&context);

  imex::populateCommonOptsPatterns(patterns);

  patterns.insert<SimplifyExpandDims>(&context);

  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

template <typename F>
static void visitTypeRecursive(mlir::Type type, F &&visitor) {
  if (auto tupleType = type.dyn_cast<mlir::TupleType>()) {
    for (auto t : tupleType.getTypes())
      visitTypeRecursive(t, std::forward<F>(visitor));
  } else {
    visitor(type);
  }
}

static bool isContigiousArray(mlir::Type type) {
  auto tensor = type.dyn_cast<imex::ntensor::NTensorType>();
  if (!tensor)
    return false;

  auto layout = tensor.getLayout();
  if (!layout)
    return false;

  return layout.getValue() == "C";
}

struct MarkContigiousArraysPass
    : public mlir::PassWrapper<MarkContigiousArraysPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkContigiousArraysPass)

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isPrivate())
      return;

    auto funcType = func.getFunctionType();

    mlir::OpBuilder builder(&getContext());
    auto attrStr = builder.getStringAttr(kContigiousArraysAttr);
    if (func->hasAttr(attrStr)) {
      markAllAnalysesPreserved();
      return;
    }

    bool needAttr = false;
    llvm::SmallVector<bool> result;
    result.reserve(funcType.getNumInputs());

    auto visitor = [&](mlir::Type type) {
      auto res = isContigiousArray(type);
      result.emplace_back(res);
      needAttr = needAttr || res;
    };

    for (auto type : (func.getFunctionType().getInputs()))
      visitTypeRecursive(type, visitor);

    if (needAttr)
      func->setAttr(attrStr, builder.getBoolArrayAttr(result));

    markAllAnalysesPreserved();
  }
};

struct LinalgOptPass
    : public mlir::PassWrapper<LinalgOptPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgOptPass)

  void runOnOperation() override;
};

static bool defaultControlFusionFn(mlir::OpOperand * /*fusedOperand*/) {
  return true;
}

void LinalgOptPass::runOnOperation() {
  auto &context = getContext();
  mlir::RewritePatternSet patterns(&context);

  imex::populateCommonOptsPatterns(patterns);

  patterns.insert<
      // clang-format off
      SimplifyExpandDims,
      LowerEnforceShape,
      GenerateToFill,
      // InsertSliceToPad,
      SliceOfGeneric
      // clang-format on
      >(&context);

  mlir::linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                     defaultControlFusionFn);
  mlir::linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct BufferizeReshape
    : public mlir::OpConversionPattern<mlir::tensor::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ReshapeOp op,
                  mlir::tensor::ReshapeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto getType = [&](mlir::Type type) {
      auto shapedType = type.cast<mlir::ShapedType>();
      return mlir::MemRefType::get(shapedType.getShape(),
                                   shapedType.getElementType());
    };
    auto source = adaptor.getSource();
    auto shape = adaptor.getShape();
    auto resType = getType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::memref::ReshapeOp>(op, resType, source,
                                                         shape);
    return mlir::success();
  }
};

struct BufferizeExtractSlice
    : public mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                  mlir::tensor::ExtractSliceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);
    auto dstType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    auto src = adaptor.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();

    auto dstRank = dstType.getRank();
    auto offsets = mlir::getMixedStridesOrOffsets(adaptor.getStaticOffsets(),
                                                  adaptor.getOffsets());
    auto sizes =
        mlir::getMixedSizes(adaptor.getStaticSizes(), adaptor.getSizes());
    auto strides = mlir::getMixedStridesOrOffsets(adaptor.getStaticStrides(),
                                                  adaptor.getStrides());

    auto viewType = [&]() {
      if (srcType.getRank() == dstRank)
        return mlir::memref::SubViewOp::inferResultType(srcType, offsets, sizes,
                                                        strides)
            .cast<mlir::MemRefType>();

      return mlir::memref::SubViewOp::inferRankReducedResultType(
                 dstType.getShape(), srcType, offsets, sizes, strides)
          .cast<mlir::MemRefType>();
    }();
    auto loc = op->getLoc();
    mlir::Value view = rewriter.create<mlir::memref::SubViewOp>(
        loc, viewType, src, offsets, sizes, strides);

    if (viewType != dstType)
      view = rewriter.create<imex::util::ChangeLayoutOp>(loc, dstType, view);

    rewriter.replaceOp(op, view);
    return mlir::success();
  }
};

struct BufferizeForceCopy
    : public mlir::OpConversionPattern<imex::util::ForceCopyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::ForceCopyOp op,
                  imex::util::ForceCopyOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);
    auto dstType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();

    if (!dstType)
      return mlir::failure();

    auto src = adaptor.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    auto rank = static_cast<unsigned>(srcType.getRank());

    auto loc = op->getLoc();
    llvm::SmallVector<mlir::Value> sizes(rank);
    for (auto i : llvm::seq(0u, rank))
      sizes[i] = rewriter.create<mlir::memref::DimOp>(loc, src, i);

    auto dst = rewriter.create<mlir::memref::AllocOp>(loc, dstType, sizes);
    genCopy(rewriter, loc, src, dst);
    rewriter.replaceOp(op, dst.getResult());
    return mlir::success();
  }
};

struct BufferizeMixedGeneric
    : public mlir::OpConversionPattern<mlir::linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::linalg::GenericOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics() || op.hasBufferSemantics())
      return mlir::failure();

    bool changed = false;
    llvm::SmallVector<mlir::Value> newInputs(adaptor.getInputs().size());
    for (auto [i, input] : llvm::enumerate(adaptor.getInputs())) {
      auto orig = op.getInputs()[i];
      if (orig.getType().isa<mlir::RankedTensorType>()) {
        newInputs[i] = input;
        changed = true;
      } else {
        newInputs[i] = orig;
      }
    }
    if (!changed)
      return mlir::failure();

    rewriter.updateRootInPlace(
        op, [&]() { op.getInputsMutable().assign(newInputs); });
    return mlir::success();
  }
};

struct FixDeallocPlacement
    : public mlir::OpRewritePattern<mlir::memref::DeallocOp> {
  using mlir::OpRewritePattern<mlir::memref::DeallocOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    auto blockIt = mlir::Block::iterator(op);
    mlir::Operation *newPos = op;
    ++blockIt;
    auto memref = op.getMemref();
    mlir::BufferViewFlowAnalysis analysis(
        op->getParentOfType<mlir::func::FuncOp>());
    auto aliases = analysis.resolve(memref);
    auto blockEnd = block->without_terminator().end();
    for (auto &it : llvm::make_range(blockIt, blockEnd)) {
      auto visitor = [&](mlir::Operation *inner) {
        for (auto arg : inner->getOperands()) {
          if (aliases.count(arg)) {
            return mlir::WalkResult::interrupt();
          }
        }
        return mlir::WalkResult::advance();
      };
      if (it.walk(visitor).wasInterrupted()) {
        newPos = &it;
      }
    }

    if (newPos != op) {
      rewriter.setInsertionPointAfter(newPos);
      rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), memref);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct AdditionalBufferize
    : public mlir::PassWrapper<AdditionalBufferize, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AdditionalBufferize)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    mlir::bufferization::BufferizeTypeConverter typeConverter;
    imex::populateTupleTypeConverter(typeConverter);

    auto materializeTupleCast =
        [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
           mlir::Location loc) -> llvm::Optional<mlir::Value> {
      if (inputs.size() != 1)
        return llvm::None;

      auto input = inputs.front();
      if (input.getType().isa<mlir::TupleType>() || type.isa<mlir::TupleType>())
        return builder.createOrFold<plier::CastOp>(loc, type, input);

      return llvm::None;
    };
    typeConverter.addArgumentMaterialization(materializeTupleCast);
    typeConverter.addSourceMaterialization(materializeTupleCast);
    typeConverter.addTargetMaterialization(materializeTupleCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    imex::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                             patterns, target);
    imex::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                       target);
    target
        .addIllegalOp<mlir::tensor::ReshapeOp, mlir::tensor::ExtractSliceOp>();
    target.addIllegalOp<imex::util::ForceCopyOp>();
    target.addLegalOp<mlir::memref::ReshapeOp>();

    target.addDynamicallyLegalOp<mlir::linalg::GenericOp>(
        [](mlir::linalg::GenericOp op) {
          return op.hasTensorSemantics() || op.hasBufferSemantics();
        });

    patterns.insert<BufferizeReshape, BufferizeExtractSlice, BufferizeForceCopy,
                    BufferizeMixedGeneric>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct CloneArgsPass
    : public mlir::PassWrapper<CloneArgsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CloneArgsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
  }

  void runOnOperation() override;
};

void CloneArgsPass::runOnOperation() {
  auto func = getOperation();
  if (func.isPrivate() || func.isDeclaration() || func.getBody().empty())
    return;

  mlir::OpBuilder builder(&getContext());

  for (auto &block : func.getBody()) {
    auto ret =
        mlir::dyn_cast_or_null<mlir::func::ReturnOp>(block.getTerminator());
    if (!ret)
      continue;

    auto loc = ret.getLoc();
    bool needReplace = false;
    llvm::SmallVector<mlir::Value> newArgs(ret.operands().size());
    builder.setInsertionPoint(ret);
    for (auto it : llvm::enumerate(ret.operands())) {
      auto i = it.index();
      auto arg = it.value();
      if (arg.getType().isa<mlir::MemRefType>()) {
        newArgs[i] = builder.create<mlir::bufferization::CloneOp>(loc, arg);
        needReplace = true;
      } else {
        newArgs[i] = arg;
      }
    }

    if (needReplace) {
      builder.create<mlir::func::ReturnOp>(loc, newArgs);
      ret.erase();
    }
  }
}

struct ReplaceClones
    : public mlir::OpRewritePattern<mlir::bufferization::CloneOp> {
  using mlir::OpRewritePattern<mlir::bufferization::CloneOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<imex::util::RetainOp>(op, op.getType(),
                                                      op.getSource());
    return mlir::success();
  }
};

struct LowerCloneOpsPass
    : public imex::RewriteWrapperPass<LowerCloneOpsPass, mlir::func::FuncOp,
                                      void, ReplaceClones> {};

struct PostLinalgOptPass
    : public mlir::PassWrapper<PostLinalgOptPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostLinalgOptPass)

  void runOnOperation() override;
};

void PostLinalgOptPass::runOnOperation() {
  auto func = getOperation();
  auto optLevel = getOptLevel(func);
  if (0 == optLevel)
    return;

  auto &context = getContext();
  mlir::RewritePatternSet patterns(&context);

  imex::populateCommonOptsPatterns(patterns);

  patterns.insert<OptimizeGlobalsConstsLoad, OptimizeSingleElemCopy,
                  imex::CanonicalizeReduction, imex::PromoteToParallel,
                  imex::MergeNestedForIntoParallel>(&context);

  auto additionalOpt = [](mlir::func::FuncOp op) {
    (void)imex::prepareForFusion(op.getRegion());
    return imex::naivelyFuseParallelOps(op.getRegion());
  };

  if (mlir::failed(applyOptimizations(func, std::move(patterns),
                                      getAnalysisManager(), additionalOpt)))
    signalPassFailure();
}

struct FixDeallocPlacementPass
    : public imex::RewriteWrapperPass<FixDeallocPlacementPass,
                                      mlir::func::FuncOp, void,
                                      FixDeallocPlacement> {};

static void populatePlierToLinalgGenPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<MarkContigiousArraysPass>());
  pm.addPass(std::make_unique<PlierToNtensorPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<ResolveNumpyFuncsPass>());
  pm.addPass(imex::ntensor::createPropagateEnvironmentPass());
  pm.addPass(std::make_unique<ResolveNtensorPass>());
  pm.addPass(std::make_unique<WrapParforRegions>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(imex::createNtensorAliasAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(imex::createNtensorToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(imex::createForceInlinePass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<PostPlierToLinalgPass>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}

static void populatePlierToLinalgOptPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<LinalgOptPass>());

  pm.addPass(imex::createNtensorToMemrefPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(imex::createMakeSignlessPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.addPass(mlir::arith::createConstantBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<AdditionalBufferize>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createTensorBufferizePass());
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());

  pm.addPass(mlir::createCanonicalizerPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferHoistingPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferLoopHoistingPass());

  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<CloneArgsPass>());
  pm.addPass(std::make_unique<MakeStridedLayoutPass>());
  pm.addPass(std::make_unique<OptimizeStridedLayoutPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<FinalizeStridedLayoutPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<LowerCloneOpsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createPromoteBuffersToStackPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToParallelLoopsPass());
  pm.addPass(imex::createForceInlinePass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(imex::createPromoteBoolMemrefPass());
  pm.addNestedPass<mlir::func::FuncOp>(imex::createUpliftMathPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLoopInvariantCodeMotionPass());

  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  // ToDo: This pass also tries to do some simple fusion, whic should be split
  // in separate pass
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<PostLinalgOptPass>());

  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<FixDeallocPlacementPass>());

  pm.addPass(mlir::createSymbolDCEPass());
}
} // namespace

// ToDo: how does this sink stuff actually works?
void registerPlierToLinalgPipeline(imex::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToLinalgGenPipelineName(), {plierToStdPipelineName()},
         {plierToLinalgOptPipelineName(), untuplePipelineName()},
         {plierToScfPipelineName()}, &populatePlierToLinalgGenPipeline);
    sink(plierToLinalgOptPipelineName(),
         {plierToLinalgGenPipelineName(), untuplePipelineName()},
         {removeSignPipelineName(), stage.end}, {},
         &populatePlierToLinalgOptPipeline);
  });
}

llvm::StringRef plierToLinalgGenPipelineName() { return "plier_to_linalg_gen"; }

llvm::StringRef plierToLinalgOptPipelineName() { return "plier_to_linalg_opt"; }
