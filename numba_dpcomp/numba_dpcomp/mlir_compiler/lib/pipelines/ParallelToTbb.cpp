// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipelines/ParallelToTbb.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "pipelines/BasePipeline.hpp"
#include "pipelines/LowerToLlvm.hpp"

#include "imex/Compiler/PipelineRegistry.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Transforms/ConstUtils.hpp"
#include "imex/Transforms/FuncUtils.hpp"
#include "imex/Transforms/RewriteWrapper.hpp"

namespace {
static mlir::MemRefType getReduceType(mlir::Type type, int64_t count) {
  if (type.isIntOrFloat())
    return mlir::MemRefType::get(count, type);

  return {};
}

static mlir::Attribute getReduceInitVal(mlir::Type type,
                                        mlir::Block &reduceBlock) {
  if (!llvm::hasSingleElement(reduceBlock.without_terminator()))
    return {};

  auto &reduceOp = reduceBlock.front();
  double reduceInit;
  if (mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp, mlir::arith::SubFOp,
                mlir::arith::SubIOp>(reduceOp)) {
    reduceInit = 0.0;
  } else if (mlir::isa<mlir::arith::MulFOp, mlir::arith::MulIOp>(reduceOp)) {
    reduceInit = 1.0;
  } else {
    return {};
  }
  return imex::getConstAttr(type, reduceInit);
}

struct ParallelToTbb : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (mlir::isa<imex::util::ParallelOp>(op->getParentOp()))
      return mlir::failure();

    bool needParallel =
        op->hasAttr(imex::util::attributes::getParallelName()) ||
        !op->getParentOfType<mlir::scf::ParallelOp>();
    if (!needParallel)
      return mlir::failure();

    int64_t maxConcurrency = 0;
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();
    if (auto mc = mod->getAttrOfType<mlir::IntegerAttr>(
            imex::util::attributes::getMaxConcurrencyName()))
      maxConcurrency = mc.getInt();

    if (maxConcurrency <= 1)
      return mlir::failure();

    for (auto type : op.getResultTypes())
      if (!getReduceType(type, maxConcurrency))
        return mlir::failure();

    llvm::SmallVector<mlir::Attribute> initVals;
    initVals.reserve(op.getNumResults());
    for (auto &nestedOp : op.getLoopBody().front().without_terminator()) {
      if (auto reduce = mlir::dyn_cast<mlir::scf::ReduceOp>(nestedOp)) {
        auto ind = static_cast<unsigned>(initVals.size());
        if (ind >= op.getNumResults())
          return mlir::failure();

        auto &region = reduce.getReductionOperator();
        if (!llvm::hasSingleElement(region))
          return mlir::failure();

        auto reduceInitVal =
            getReduceInitVal(op.getResult(ind).getType(), region.front());
        if (!reduceInitVal)
          return mlir::failure();

        initVals.emplace_back(reduceInitVal);
      }
    }

    if (initVals.size() != op.getNumResults())
      return mlir::failure();

    imex::AllocaInsertionPoint allocaIP(op);

    auto loc = op.getLoc();
    mlir::BlockAndValueMapping mapping;
    llvm::SmallVector<mlir::Value> reduceVars(op.getNumResults());
    for (auto it : llvm::enumerate(op.getResultTypes())) {
      auto type = it.value();
      auto reduceType = getReduceType(type, maxConcurrency);
      assert(reduceType);
      auto reduce = allocaIP.insert(rewriter, [&]() {
        return rewriter.create<mlir::memref::AllocaOp>(loc, reduceType);
      });
      auto index = static_cast<unsigned>(it.index());
      reduceVars[index] = reduce;
    }

    auto reduceInitBodyBuilder = [&](mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value index,
                                     mlir::ValueRange args) {
      assert(args.empty());
      (void)args;
      for (auto it : llvm::enumerate(reduceVars)) {
        auto reduce = it.value();
        auto initVal = initVals[it.index()];
        auto init = builder.create<mlir::arith::ConstantOp>(loc, initVal);
        builder.create<mlir::memref::StoreOp>(loc, init, reduce, index);
      }
      builder.create<mlir::scf::YieldOp>(loc);
    };

    auto reduceLowerBound =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto reduceUpperBound =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, maxConcurrency);
    auto reduceStep = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    rewriter.create<mlir::scf::ForOp>(loc, reduceLowerBound, reduceUpperBound,
                                      reduceStep, llvm::None,
                                      reduceInitBodyBuilder);

    auto &oldBody = op.getLoopBody().front();
    auto origLowerBound = op.getLowerBound();
    auto origUpperBound = op.getUpperBound();
    auto origStep = op.getStep();
    auto bodyBuilder = [&](mlir::OpBuilder &builder, ::mlir::Location loc,
                           mlir::ValueRange lowerBound,
                           mlir::ValueRange upperBound,
                           mlir::Value threadIndex) {
      llvm::SmallVector<mlir::Value> initVals(op.getInitVals().size());
      for (auto it : llvm::enumerate(op.getInitVals())) {
        auto reduceVar = reduceVars[it.index()];
        auto val =
            builder.create<mlir::memref::LoadOp>(loc, reduceVar, threadIndex);
        initVals[it.index()] = val;
      }
      auto newOp =
          mlir::cast<mlir::scf::ParallelOp>(builder.clone(*op, mapping));
      newOp->removeAttr(imex::util::attributes::getParallelName());
      assert(newOp->getNumResults() == reduceVars.size());
      newOp.getLowerBoundMutable().assign(lowerBound);
      newOp.getUpperBoundMutable().assign(upperBound);
      newOp.getInitValsMutable().assign(initVals);
      for (auto it : llvm::enumerate(newOp->getResults())) {
        auto reduce_var = reduceVars[it.index()];
        builder.create<mlir::memref::StoreOp>(loc, it.value(), reduce_var,
                                              threadIndex);
      }
    };

    rewriter.create<imex::util::ParallelOp>(loc, origLowerBound, origUpperBound,
                                            origStep, bodyBuilder);

    auto reduceBodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value index, mlir::ValueRange args) {
      assert(args.size() == reduceVars.size());
      mapping.clear();
      auto reduceOps =
          llvm::make_filter_range(oldBody.without_terminator(), [](auto &op) {
            return mlir::isa<mlir::scf::ReduceOp>(op);
          });
      llvm::SmallVector<mlir::Value> yieldArgs;
      yieldArgs.reserve(args.size());
      for (auto it : llvm::enumerate(reduceOps)) {
        auto &reduceVar = reduceVars[it.index()];
        auto arg = args[static_cast<unsigned>(it.index())];
        auto reduceOp = mlir::cast<mlir::scf::ReduceOp>(it.value());
        auto &reduceOpBody = reduceOp.getReductionOperator().front();
        assert(reduceOpBody.getNumArguments() == 2);
        auto prevVal =
            builder.create<mlir::memref::LoadOp>(loc, reduceVar, index);
        mapping.map(reduceOpBody.getArgument(0), arg);
        mapping.map(reduceOpBody.getArgument(1), prevVal);
        for (auto &oldReduceOp : reduceOpBody.without_terminator())
          builder.clone(oldReduceOp, mapping);

        auto result =
            mlir::cast<mlir::scf::ReduceReturnOp>(reduceOpBody.getTerminator())
                .getResult();
        result = mapping.lookupOrNull(result);
        assert(result);
        yieldArgs.emplace_back(result);
      }
      builder.create<mlir::scf::YieldOp>(loc, yieldArgs);
    };

    auto reduceLoop = rewriter.create<mlir::scf::ForOp>(
        loc, reduceLowerBound, reduceUpperBound, reduceStep, op.getInitVals(),
        reduceBodyBuilder);
    rewriter.replaceOp(op, reduceLoop.getResults());

    return mlir::success();
  }
};

static bool
isAnyArgDefinedInsideRegions(llvm::MutableArrayRef<mlir::Region> regs,
                             mlir::Operation *op) {
  assert(op);
  for (auto arg : op->getOperands())
    for (auto &reg : regs)
      if (reg.isAncestor(arg.getParentRegion()))
        return true;

  return false;
}

struct LoopInfo {
  mlir::Operation *outermostLoop = nullptr;
  imex::util::ParallelOp innermostParallel;
};

static llvm::Optional<LoopInfo> getLoopInfo(mlir::Operation *op) {
  assert(op);

  LoopInfo ret;
  auto parent = op->getParentOp();
  while (parent) {
    if (isAnyArgDefinedInsideRegions(parent->getRegions(), op))
      break;

    if (mlir::isa<mlir::scf::WhileOp, mlir::scf::ForOp, mlir::scf::ParallelOp,
                  imex::util::ParallelOp>(parent))
      ret.outermostLoop = parent;

    if (!ret.innermostParallel && mlir::isa<imex::util::ParallelOp>(parent))
      ret.innermostParallel = mlir::cast<imex::util::ParallelOp>(parent);

    parent = parent->getParentOp();
  }

  if (!ret.outermostLoop)
    return llvm::None;

  return ret;
}

static bool canResultEscape(mlir::Operation *op, bool original = true) {
  for (auto user : op->getUsers()) {
    if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user))
      continue;

    if (original && mlir::isa<mlir::memref::DeallocOp>(user))
      continue;

    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user)) {
      if (canResultEscape(user, false))
        return true;
    }
  }

  return false;
}

struct HoistBufferAllocs
    : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getSymbolOperands().empty())
      return mlir::failure();

    if (canResultEscape(op))
      return mlir::failure();

    auto loopInfo = getLoopInfo(op);
    if (!loopInfo)
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto mc = mod->getAttrOfType<mlir::IntegerAttr>(
        imex::util::attributes::getMaxConcurrencyName());
    if (loopInfo->innermostParallel && !mc)
      return mlir::failure();

    bool needParallel =
        loopInfo->innermostParallel && mc.getValue().getSExtValue() > 0;

    auto oldType = op.getType().cast<mlir::MemRefType>();
    auto memrefType = [&]() -> mlir::MemRefType {
      if (needParallel) {
        llvm::SmallVector<int64_t> newShape;
        newShape.emplace_back(mc.getValue().getSExtValue());
        auto oldShape = oldType.getShape();
        newShape.append(oldShape.begin(), oldShape.end());
        return mlir::MemRefType::get(newShape, oldType.getElementType(),
                                     oldType.getMemorySpace());
      }
      return oldType;
    }();

    for (auto user : llvm::make_early_inc_range(op->getUsers()))
      if (mlir::isa<mlir::memref::DeallocOp>(user))
        rewriter.eraseOp(user);

    auto loc = op.getLoc();
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(loopInfo->outermostLoop);
    mlir::Value newMemref = rewriter.create<mlir::memref::AllocOp>(
        loc, memrefType, op.getDynamicSizes(), op.getAlignmentAttr());
    mlir::Value view = newMemref;
    if (needParallel) {
      auto zero = rewriter.getIndexAttr(0);
      auto one = rewriter.getIndexAttr(1);

      auto rank = static_cast<unsigned>(memrefType.getRank());
      rewriter.setInsertionPointToStart(
          loopInfo->innermostParallel.getBodyBlock());
      llvm::SmallVector<mlir::OpFoldResult> offsets(rank, zero);
      llvm::SmallVector<mlir::OpFoldResult> sizes(rank, one);
      llvm::SmallVector<mlir::OpFoldResult> strides(rank, one);

      auto threadIndex = loopInfo->innermostParallel.getBodyThreadIndex();
      offsets[0] = threadIndex;
      for (auto i : llvm::seq(0u, rank - 1))
        sizes[i + 1] =
            rewriter.createOrFold<mlir::memref::DimOp>(loc, newMemref, i + 1);

      auto newType =
          mlir::memref::SubViewOp::inferRankReducedResultType(
              oldType.getShape(), memrefType, offsets, sizes, strides)
              .cast<mlir::MemRefType>();
      view = rewriter.create<mlir::memref::SubViewOp>(loc, newType, newMemref,
                                                      offsets, sizes, strides);
      if (view.getType() != oldType) {
        view = rewriter.create<imex::util::MemrefApplyOffsetOp>(loc, oldType,
                                                                view);
        view = rewriter.create<mlir::memref::CastOp>(loc, oldType, view);
      }
    }

    rewriter.replaceOp(op, view);

    rewriter.setInsertionPointAfter(loopInfo->outermostLoop);
    rewriter.create<mlir::memref::DeallocOp>(loc, newMemref);
    return mlir::success();
  }
};

struct ParallelToTbbPass
    : public imex::RewriteWrapperPass<
          ParallelToTbbPass, mlir::func::FuncOp,
          imex::DependentDialectsList<imex::util::ImexUtilDialect,
                                      mlir::arith::ArithDialect,
                                      mlir::scf::SCFDialect>,
          ParallelToTbb> {};

struct HoistBufferAllocsPass
    : public imex::RewriteWrapperPass<
          HoistBufferAllocsPass, mlir::func::FuncOp,
          imex::DependentDialectsList<imex::util::ImexUtilDialect,
                                      mlir::scf::SCFDialect,
                                      mlir::memref::MemRefDialect>,
          HoistBufferAllocs> {};

static void populateParallelToTbbPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<ParallelToTbbPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<HoistBufferAllocsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
}
} // namespace

void registerParallelToTBBPipeline(imex::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getLowerLoweringStage();
    auto llvm_pipeline = lowerToLLVMPipelineName();
    sink(parallelToTBBPipelineName(), {stage.begin}, {llvm_pipeline}, {},
         &populateParallelToTbbPipeline);
  });
}

llvm::StringRef parallelToTBBPipelineName() { return "parallel_to_tbb"; }
