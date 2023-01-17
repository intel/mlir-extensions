// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/PromoteToParallel.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Interfaces/CallInterfaces.h>

static bool hasSideEffects(mlir::Operation *op) {
  return op
      ->walk([&](mlir::Operation *op) {
        if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
          if (effects.hasEffect<mlir::MemoryEffects::Write>()) {
            return mlir::WalkResult::interrupt();
          }
        }
        //        if (op->hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
        //        {
        //            return mlir::WalkResult::interrupt();
        //        }
        if (mlir::isa<mlir::CallOpInterface>(op)) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

static bool canParallelizeLoop(mlir::Operation *op, bool hasParallelAttr) {
  return hasParallelAttr || !hasSideEffects(op);
}

static mlir::Operation *getSingleUser(mlir::Value val) {
  if (!val.hasOneUse())
    return nullptr;

  return *val.user_begin();
}

mlir::LogicalResult imex::PromoteToParallel::matchAndRewrite(
    mlir::scf::ForOp op, mlir::PatternRewriter &rewriter) const {
  auto hasParallelAttr = op->hasAttr(imex::util::attributes::getParallelName());
  if (!canParallelizeLoop(op, hasParallelAttr))
    return mlir::failure();

  auto &oldBody = op.getLoopBody().front();
  auto oldYield = mlir::cast<mlir::scf::YieldOp>(oldBody.getTerminator());
  auto reduceArgs = oldBody.getArguments().drop_front();
  llvm::SmallVector<llvm::SmallVector<mlir::Operation *, 1>> reduce_bodies(
      reduceArgs.size());
  llvm::DenseSet<mlir::Operation *> reduceOps;
  for (auto [reduceIndex, reduceArg] : llvm::enumerate(reduceArgs)) {
    auto reduceOp = getSingleUser(reduceArg);
    if (!reduceOp)
      return mlir::failure();

    if (reduceOp->getNumOperands() != 2 || reduceOp->getNumResults() != 1)
      return mlir::failure();

    auto &reduceBody = reduce_bodies[reduceIndex];
    while (true) {
      auto nextOp = getSingleUser(reduceOp->getResult(0));
      if (!nextOp)
        return mlir::failure();

      reduceBody.push_back(reduceOp);
      reduceOps.insert(reduceOp);
      if (nextOp == oldYield) {
        auto yieldOperand =
            oldYield.getOperand(static_cast<unsigned>(reduceIndex));
        if (yieldOperand != reduceOp->getResult(0))
          return mlir::failure();

        break;
      }
      for (auto operand : nextOp->getOperands()) {
        if (operand.getDefiningOp() != reduceOp &&
            operand.getParentBlock() == &oldBody)
          return mlir::failure();
      }
      reduceOp = nextOp;
    }
  }

  auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange iterVals, mlir::ValueRange temp) {
    assert(1 == iterVals.size());
    assert(temp.empty());
    mlir::BlockAndValueMapping mapping;
    mapping.map(oldBody.getArguments().front(), iterVals.front());
    for (auto &old_op : oldBody.without_terminator()) {
      if (0 == reduceOps.count(&old_op))
        builder.clone(old_op, mapping);
    }
    mlir::BlockAndValueMapping reduceNapping;
    for (auto it : llvm::enumerate(reduce_bodies)) {
      auto &reduceBody = it.value();
      assert(!reduceBody.empty());
      reduceNapping = mapping;
      auto firstOp = reduceBody.front();
      assert(firstOp->getNumOperands() == 2);
      auto reduceBodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value val0, mlir::Value val1) {
        reduceNapping.map(firstOp->getOperand(0), val0);
        reduceNapping.map(firstOp->getOperand(1), val1);
        mlir::Operation *last_op = nullptr;
        for (auto reduceOp : reduceBody) {
          last_op = builder.clone(*reduceOp, reduceNapping);
          assert(1 == last_op->getNumResults());
        }
        builder.create<mlir::scf::ReduceReturnOp>(loc, last_op->getResult(0));
      };
      auto reduceArg = reduceArgs[it.index()];
      auto firstOpOperands = firstOp->getOperands();
      auto reduceOperand =
          (firstOpOperands[0] == reduceArg ? firstOpOperands[1]
                                           : firstOpOperands[0]);
      assert(reduceOperand != reduceArg);
      reduceOperand = mapping.lookupOrDefault(reduceOperand);
      assert(reduceOperand);
      builder.create<mlir::scf::ReduceOp>(loc, reduceOperand,
                                          reduceBodyBuilder);
    }
  };

  auto parallelOp = rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
      op, op.getLowerBound(), op.getUpperBound(), op.getStep(),
      op.getInitArgs(), bodyBuilder);
  if (hasParallelAttr)
    parallelOp->setAttr(imex::util::attributes::getParallelName(),
                        rewriter.getUnitAttr());

  return mlir::success();
}

mlir::LogicalResult imex::MergeNestedForIntoParallel::matchAndRewrite(
    mlir::scf::ParallelOp op, mlir::PatternRewriter &rewriter) const {
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

  auto hasParallelAttr = op->hasAttr(imex::util::attributes::getParallelName());
  if (!canParallelizeLoop(op, hasParallelAttr))
    return mlir::failure();

  auto makeValueList = [](auto op, auto ops) {
    llvm::SmallVector<mlir::Value> ret;
    ret.reserve(ops.size() + 1);
    ret.emplace_back(op);
    ret.append(ops.begin(), ops.end());
    return ret;
  };

  auto lowerBounds = makeValueList(parent.getLowerBound(), op.getLowerBound());
  auto upperBounds = makeValueList(parent.getUpperBound(), op.getUpperBound());
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
