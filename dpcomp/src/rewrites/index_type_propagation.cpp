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

#include "plier/rewrites/index_type_propagation.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>

namespace {
bool is_index_compatible(mlir::Type lhs_type, mlir::Type rhs_type) {
  if (!lhs_type.isa<mlir::IntegerType>() || lhs_type != rhs_type) {
    return false;
  }

  if (lhs_type.cast<mlir::IntegerType>().getWidth() < 64) {
    return false;
  }
  return true;
}

template <typename Op>
struct ArithIndexCastSimplify : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto lhsType = op.lhs().getType();
    auto rhsType = op.rhs().getType();
    if (!is_index_compatible(lhsType, rhsType))
      return mlir::failure();

    auto getCast = [](mlir::Value val) -> mlir::Value {
      if (auto op = val.getDefiningOp<mlir::arith::IndexCastOp>())
        return op.getOperand();

      return {};
    };

    auto getConst = [](mlir::Value val) -> mlir::IntegerAttr {
      if (auto op = val.getDefiningOp<mlir::arith::ConstantOp>())
        return op.value().cast<mlir::IntegerAttr>();

      return {};
    };

    auto lhs = getCast(op.lhs());
    auto rhs = getCast(op.rhs());
    auto lhsConst = getConst(op.lhs());
    auto rhsConst = getConst(op.rhs());
    if (lhs && rhs) {
      auto newOp = rewriter.create<Op>(op.getLoc(), lhs, rhs);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), newOp.getResult(), lhsType);
      rewriter.replaceOp(op, result.getResult());
      return mlir::success();
    }
    if (lhs && rhsConst) {
      auto newConst = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), rhsConst.getInt());
      auto newOp = rewriter.create<Op>(op.getLoc(), lhs, newConst);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), newOp.getResult(), lhsType);
      rewriter.replaceOp(op, result.getResult());
      return mlir::success();
    }
    if (lhsConst && rhs) {
      auto newConst = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), lhsConst.getInt());
      auto newOp = rewriter.create<Op>(op.getLoc(), newConst, rhs);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), newOp.getResult(), lhsType);
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
    auto lhsType = op.lhs().getType();
    auto rhsType = op.rhs().getType();
    if (!is_index_compatible(lhsType, rhsType))
      return mlir::failure();

    auto getCast = [](mlir::Value val) -> mlir::Value {
      if (auto op = val.getDefiningOp<mlir::arith::IndexCastOp>())
        return op.getOperand();

      return {};
    };

    auto getConst = [](mlir::Value val) -> mlir::IntegerAttr {
      if (auto op = val.getDefiningOp<mlir::arith::ConstantOp>())
        return op.value().cast<mlir::IntegerAttr>();

      return {};
    };

    auto lhs = getCast(op.lhs());
    auto rhs = getCast(op.rhs());
    auto lhsConst = getConst(op.lhs());
    auto rhsConst = getConst(op.rhs());
    if (lhs && rhs) {
      auto newCmp = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), op.predicate(), lhs, rhs);
      rewriter.replaceOp(op, newCmp.getResult());
      return mlir::success();
    }
    if (lhs && rhsConst) {
      auto newConst = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), rhsConst.getInt());
      auto newCmp = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), op.predicate(), lhs, newConst);
      rewriter.replaceOp(op, newCmp.getResult());
      return mlir::success();
    }
    if (lhsConst && rhs) {
      auto newConst = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), lhsConst.getInt());
      auto newCmp = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), op.predicate(), newConst, rhs);
      rewriter.replaceOp(op, newCmp.getResult());
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

void plier::populate_index_propagate_patterns(
    mlir::MLIRContext &context, mlir::RewritePatternSet &patterns) {
  patterns
      .insert<CmpIndexCastSimplify, ArithIndexCastSimplify<mlir::arith::SubIOp>,
              ArithIndexCastSimplify<mlir::arith::AddIOp>,
              ArithIndexCastSimplify<mlir::arith::MulIOp>,
              ArithIndexCastSimplify<mlir::arith::DivSIOp>,
              ArithIndexCastSimplify<mlir::arith::DivUIOp>,
              ArithIndexCastSimplify<mlir::arith::RemSIOp>,
              ArithIndexCastSimplify<mlir::arith::RemUIOp>>(&context);
}
