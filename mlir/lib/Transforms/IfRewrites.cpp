// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/IfRewrites.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>

namespace {
struct IfOpConstCond : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  IfOpConstCond(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::scf::IfOp>(context, /*benefit*/ 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace

mlir::LogicalResult
IfOpConstCond::matchAndRewrite(mlir::scf::IfOp op,
                               mlir::PatternRewriter &rewriter) const {
  auto cond = op.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!cond)
    return mlir::failure();

  auto isConst = [](mlir::Value val) {
    if (auto parent = val.getDefiningOp())
      return parent->hasTrait<mlir::OpTrait::ConstantLike>();

    return false;
  };

  auto replace = [&](mlir::Block &block, mlir::Value toReplace,
                     mlir::Value newVal) {
    for (auto &use : llvm::make_early_inc_range(toReplace.getUses())) {
      auto owner = use.getOwner();
      if (block.findAncestorOpInBlock(*owner))
        rewriter.updateRootInPlace(owner, [&]() { use.set(newVal); });
    }
  };

  auto lhs = cond.getLhs();
  auto rhs = cond.getRhs();
  auto pred = cond.getPredicate();
  mlir::Value constVal;
  mlir::Value toReplace;
  if (isConst(lhs)) {
    constVal = lhs;
    toReplace = rhs;
  } else if (isConst(rhs)) {
    constVal = rhs;
    toReplace = lhs;
  } else {
    return mlir::failure();
  }

  if (pred == mlir::arith::CmpIPredicate::eq) {
    replace(*op.thenBlock(), toReplace, constVal);
  } else if (pred == mlir::arith::CmpIPredicate::ne && op.elseBlock()) {
    replace(*op.elseBlock(), toReplace, constVal);
  } else {
    return mlir::failure();
  }

  return mlir::success();
}

void imex::populateIfRewritesPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<IfOpConstCond>(patterns.getContext());
}
