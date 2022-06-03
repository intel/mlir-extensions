// Copyright 2021 Intel Corporation
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

#include "mlir-extensions/Transforms/index_type_propagation.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/PatternMatch.h>

namespace {
bool is_index_compatible(mlir::Type lhsType, mlir::Type rhsType) {
  if (!lhsType.isa<mlir::IntegerType>() || lhsType != rhsType)
    return false;

  if (lhsType.cast<mlir::IntegerType>().getWidth() < 64)
    return false;

  return true;
}

template <typename Op>
struct ArithIndexCastSimplify : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    if (!is_index_compatible(lhsType, rhsType))
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
    if (!is_index_compatible(lhsType, rhsType))
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

void plier::populateIndexPropagatePatterns(mlir::MLIRContext &context,
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
