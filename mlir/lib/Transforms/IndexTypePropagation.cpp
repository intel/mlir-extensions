// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/IndexTypePropagation.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>

static bool isIndexCompatible(mlir::Type lhsType, mlir::Type rhsType) {
  if (!lhsType.isa<mlir::IntegerType>() || lhsType != rhsType)
    return false;

  if (lhsType.cast<mlir::IntegerType>().getWidth() < 64)
    return false;

  return true;
}

namespace {
template <typename Op>
struct ArithIndexCastSimplify : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    if (!isIndexCompatible(lhsType, rhsType))
      return mlir::failure();

    auto getCast = [](mlir::Value val) -> mlir::Value {
      if (auto op = val.getDefiningOp<mlir::arith::IndexCastOp>())
        return op.getIn();

      return {};
    };

    auto getConst = [](mlir::Value val) -> mlir::IntegerAttr {
      if (auto op = val.getDefiningOp<mlir::arith::ConstantOp>())
        return op.getValue().cast<mlir::IntegerAttr>();

      return {};
    };

    auto lhs = getCast(op.getLhs());
    auto rhs = getCast(op.getRhs());
    auto lhsConst = getConst(op.getLhs());
    auto rhsConst = getConst(op.getRhs());
    auto loc = op.getLoc();
    if (lhs && rhs) {
      auto newOp = rewriter.create<Op>(op.getLoc(), lhs, rhs);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          loc, lhsType, newOp.getResult());
      rewriter.replaceOp(op, result.getResult());
      return mlir::success();
    }
    if (lhs && rhsConst) {
      auto newConst =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, rhsConst.getInt());
      auto newOp = rewriter.create<Op>(op.getLoc(), lhs, newConst);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          loc, lhsType, newOp.getResult());
      rewriter.replaceOp(op, result.getResult());
      return mlir::success();
    }
    if (lhsConst && rhs) {
      auto newConst =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, lhsConst.getInt());
      auto newOp = rewriter.create<Op>(op.getLoc(), newConst, rhs);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          loc, lhsType, newOp.getResult());
      rewriter.replaceOp(op, result.getResult());
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct CmpIndexCastSimplify
    : public mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    if (!isIndexCompatible(lhsType, rhsType))
      return mlir::failure();

    auto getCast = [](mlir::Value val) -> mlir::Value {
      if (auto op = val.getDefiningOp<mlir::arith::IndexCastOp>())
        return op.getOperand();

      return {};
    };

    auto getConst = [](mlir::Value val) -> mlir::IntegerAttr {
      if (auto op = val.getDefiningOp<mlir::arith::ConstantOp>())
        return op.getValue().cast<mlir::IntegerAttr>();

      return {};
    };

    auto lhs = getCast(op.getLhs());
    auto rhs = getCast(op.getRhs());
    auto lhsConst = getConst(op.getLhs());
    auto rhsConst = getConst(op.getRhs());
    auto pred = op.getPredicate();
    auto loc = op.getLoc();
    if (lhs && rhs) {
      auto newCmp = rewriter.create<mlir::arith::CmpIOp>(loc, pred, lhs, rhs);
      rewriter.replaceOp(op, newCmp.getResult());
      return mlir::success();
    }
    if (lhs && rhsConst) {
      auto newConst =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, rhsConst.getInt());
      auto newCmp =
          rewriter.create<mlir::arith::CmpIOp>(loc, pred, lhs, newConst);
      rewriter.replaceOp(op, newCmp.getResult());
      return mlir::success();
    }
    if (lhsConst && rhs) {
      auto newConst =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, lhsConst.getInt());
      auto newCmp =
          rewriter.create<mlir::arith::CmpIOp>(loc, pred, newConst, rhs);
      rewriter.replaceOp(op, newCmp.getResult());
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

void imex::populateIndexPropagatePatterns(mlir::MLIRContext &context,
                                          mlir::RewritePatternSet &patterns) {
  patterns
      .insert<CmpIndexCastSimplify, ArithIndexCastSimplify<mlir::arith::SubIOp>,
              ArithIndexCastSimplify<mlir::arith::AddIOp>,
              ArithIndexCastSimplify<mlir::arith::MulIOp>,
              ArithIndexCastSimplify<mlir::arith::DivSIOp>,
              ArithIndexCastSimplify<mlir::arith::DivUIOp>,
              ArithIndexCastSimplify<mlir::arith::RemSIOp>,
              ArithIndexCastSimplify<mlir::arith::RemUIOp>>(&context);
}
