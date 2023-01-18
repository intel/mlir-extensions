// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/PromoteToParallel.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static bool hasSideEffects(mlir::Operation *op) {
  assert(op);
  for (auto &region : op->getRegions()) {
    auto visitor = [](mlir::Operation *bodyOp) -> mlir::WalkResult {
      if (mlir::isa<mlir::scf::ReduceOp>(bodyOp) ||
          bodyOp->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>() ||
          bodyOp->hasTrait<mlir::OpTrait::IsTerminator>())
        return mlir::WalkResult::advance();

      if (mlir::isa<mlir::CallOpInterface>(bodyOp))
        return mlir::WalkResult::interrupt();

      auto memEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(bodyOp);
      if (!memEffects || memEffects.hasEffect<mlir::MemoryEffects::Write>())
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::advance();
    };
    if (region.walk(visitor).wasInterrupted())
      return true;
  }
  return false;
}

static bool canParallelizeLoop(mlir::Operation *op, bool hasParallelAttr) {
  return hasParallelAttr || !hasSideEffects(op);
}

using CheckFunc = bool (*)(mlir::Operation *, mlir::Value);
using LowerFunc = void (*)(mlir::OpBuilder &, mlir::Location, mlir::Value,
                           mlir::Operation *);

template <typename Op>
static bool simpleCheck(mlir::Operation *op, mlir::Value /*iterVar*/) {
  return mlir::isa<Op>(op);
}

template <typename Op>
static bool lhsArgCheck(mlir::Operation *op, mlir::Value iterVar) {
  auto casted = mlir::dyn_cast<Op>(op);
  if (!casted)
    return false;

  return casted.getLhs() == iterVar;
}

template <typename Op>
static void simpleLower(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value val, mlir::Operation *origOp) {
  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value lhs,
                         mlir::Value rhs) {
    auto casted = mlir::cast<Op>(origOp);
    mlir::BlockAndValueMapping mapper;
    mapper.map(casted.getLhs(), lhs);
    mapper.map(casted.getRhs(), rhs);
    mlir::Value res = b.clone(*origOp, mapper)->getResult(0);
    b.create<mlir::scf::ReduceReturnOp>(l, res);
  };
  builder.create<mlir::scf::ReduceOp>(loc, val, bodyBuilder);
}

template <typename Op>
static constexpr std::pair<CheckFunc, LowerFunc> getSimpleHandler() {
  return {&simpleCheck<Op>, &simpleLower<Op>};
}

static const constexpr std::pair<CheckFunc, LowerFunc> promoteHandlers[] = {
    // clang-format off
    getSimpleHandler<mlir::arith::AddIOp>(),
    getSimpleHandler<mlir::arith::AddFOp>(),

    getSimpleHandler<mlir::arith::MulIOp>(),
    getSimpleHandler<mlir::arith::MulFOp>(),

    getSimpleHandler<mlir::arith::MinSIOp>(),
    getSimpleHandler<mlir::arith::MinUIOp>(),
    getSimpleHandler<mlir::arith::MinFOp>(),

    getSimpleHandler<mlir::arith::MaxSIOp>(),
    getSimpleHandler<mlir::arith::MaxUIOp>(),
    getSimpleHandler<mlir::arith::MaxFOp>(),
    // clang-format on
};

static LowerFunc getLowerer(mlir::Operation *op, mlir::Value iterVar) {
  assert(op);
  for (auto [checker, lowerer] : promoteHandlers)
    if (checker(op, iterVar))
      return lowerer;

  return nullptr;
}

namespace {
struct PromoteToParallel : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto hasParallelAttr =
        op->hasAttr(imex::util::attributes::getParallelName());
    if (!canParallelizeLoop(op, hasParallelAttr))
      return mlir::failure();

    mlir::Block &loopBody = op.getLoopBody().front();
    auto term = mlir::cast<mlir::scf::YieldOp>(loopBody.getTerminator());
    auto iterVars = op.getRegionIterArgs();
    assert(iterVars.size() == term.getResults().size());

    using ReductionDesc = std::tuple<mlir::Operation *, LowerFunc, mlir::Value>;
    llvm::SmallVector<ReductionDesc> reductionOps;
    llvm::SmallDenseSet<mlir::Operation *> reductionOpsSet;
    for (auto [iterVar, result] : llvm::zip(iterVars, term.getResults())) {
      auto reductionOp = result.getDefiningOp();
      if (!reductionOp || reductionOp->getNumResults() != 1 ||
          reductionOp->getNumOperands() != 2 ||
          !llvm::hasSingleElement(reductionOp->getUses()))
        return mlir::failure();

      mlir::Value reductionArg;
      if (reductionOp->getOperand(0) == iterVar) {
        reductionArg = reductionOp->getOperand(1);
      } else if (reductionOp->getOperand(1) == iterVar) {
        reductionArg = reductionOp->getOperand(0);
      } else {
        return mlir::failure();
      }

      auto lowerer = getLowerer(reductionOp, iterVar);
      if (!lowerer)
        return mlir::failure();

      reductionOps.emplace_back(reductionOp, lowerer, reductionArg);
      reductionOpsSet.insert(reductionOp);
    }

    auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange iterVals, mlir::ValueRange) {
      assert(1 == iterVals.size());
      mlir::BlockAndValueMapping mapping;
      mapping.map(op.getInductionVar(), iterVals.front());
      for (auto &oldOp : loopBody.without_terminator())
        if (0 == reductionOpsSet.count(&oldOp))
          builder.clone(oldOp, mapping);

      for (auto [reductionOp, lowerer, reductionArg] : reductionOps) {
        auto arg = mapping.lookupOrDefault(reductionArg);
        lowerer(builder, loc, arg, reductionOp);
      }
      builder.create<mlir::scf::YieldOp>(loc);
    };

    auto parallelOp = rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
        op, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        op.getInitArgs(), bodyBuilder);
    if (hasParallelAttr)
      parallelOp->setAttr(imex::util::attributes::getParallelName(),
                          rewriter.getUnitAttr());

    return mlir::success();
  }
};

struct MergeNestedForIntoParallel
    : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = mlir::dyn_cast<mlir::scf::ForOp>(op->getParentOp());
    if (!parent)
      return mlir::failure();

    auto &block = parent.getLoopBody().front();
    if (!llvm::hasSingleElement(block.without_terminator()))
      return mlir::failure();

    if (parent.getInitArgs().size() != op.getInitVals().size())
      return mlir::failure();

    auto yield = mlir::cast<mlir::scf::YieldOp>(block.getTerminator());
    assert(yield.getNumOperands() == op.getNumResults());
    for (auto [arg, initVal, result, yieldOp] :
         llvm::zip(block.getArguments().drop_front(), op.getInitVals(),
                   op.getResults(), yield.getOperands())) {
      if (!arg.hasOneUse() || arg != initVal || result != yieldOp)
        return mlir::failure();
    }
    auto checkVals = [&](auto vals) {
      for (auto val : vals)
        if (val.getParentBlock() == &block)
          return true;

      return false;
    };
    if (checkVals(op.getLowerBound()) || checkVals(op.getUpperBound()) ||
        checkVals(op.getStep()))
      return mlir::failure();

    auto hasParallelAttr =
        op->hasAttr(imex::util::attributes::getParallelName());
    if (!canParallelizeLoop(op, hasParallelAttr))
      return mlir::failure();

    auto makeValueList = [](auto op, auto ops) {
      llvm::SmallVector<mlir::Value> ret;
      ret.reserve(ops.size() + 1);
      ret.emplace_back(op);
      ret.append(ops.begin(), ops.end());
      return ret;
    };

    auto lowerBounds =
        makeValueList(parent.getLowerBound(), op.getLowerBound());
    auto upperBounds =
        makeValueList(parent.getUpperBound(), op.getUpperBound());
    auto steps = makeValueList(parent.getStep(), op.getStep());

    auto &oldBody = op.getLoopBody().front();
    auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location /*loc*/,
                           mlir::ValueRange iter_vals, mlir::ValueRange temp) {
      assert(iter_vals.size() == lowerBounds.size());
      assert(temp.empty());
      mlir::BlockAndValueMapping mapping;
      assert((oldBody.getNumArguments() + 1) == iter_vals.size());
      mapping.map(block.getArgument(0), iter_vals.front());
      mapping.map(oldBody.getArguments(), iter_vals.drop_front());
      for (auto &op : oldBody.without_terminator())
        builder.clone(op, mapping);
    };

    rewriter.setInsertionPoint(parent);
    auto newOp = rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
        parent, lowerBounds, upperBounds, steps, parent.getInitArgs(),
        bodyBuilder);
    if (hasParallelAttr)
      newOp->setAttr(imex::util::attributes::getParallelName(),
                     rewriter.getUnitAttr());

    return mlir::success();
  }
};

struct PromoteToParallelPass
    : public mlir::PassWrapper<PromoteToParallelPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteToParallelPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto context = &getContext();

    mlir::RewritePatternSet patterns(context);
    imex::populatePromoteToParallelPatterns(patterns);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void imex::populatePromoteToParallelPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.insert<PromoteToParallel, MergeNestedForIntoParallel>(
      patterns.getContext());
}

std::unique_ptr<mlir::Pass> imex::createPromoteToParallelPass() {
  return std::make_unique<PromoteToParallelPass>();
}
