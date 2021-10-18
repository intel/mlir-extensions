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
    auto lhs_type = op.lhs().getType();
    auto rhs_type = op.rhs().getType();
    if (!is_index_compatible(lhs_type, rhs_type)) {
      return mlir::failure();
    }

    auto get_cast = [](mlir::Value val) -> mlir::Value {
      if (auto op = mlir::dyn_cast_or_null<mlir::arith::IndexCastOp>(
              val.getDefiningOp())) {
        return op.getOperand();
      }
      return {};
    };

    auto get_const = [](mlir::Value val) -> mlir::IntegerAttr {
      if (auto op =
              mlir::dyn_cast_or_null<mlir::ConstantOp>(val.getDefiningOp())) {
        return op.getValue().cast<mlir::IntegerAttr>();
      }
      return {};
    };

    auto lhs = get_cast(op.lhs());
    auto rhs = get_cast(op.rhs());
    auto lhs_const = get_const(op.lhs());
    auto rhs_const = get_const(op.rhs());
    if (lhs && rhs) {
      auto new_op = rewriter.create<Op>(op.getLoc(), lhs, rhs);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), new_op.getResult(), lhs_type);
      rewriter.replaceOp(op, result.getResult());
      return mlir::success();
    }
    if (lhs && rhs_const) {
      auto new_const = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), rhs_const.getInt());
      auto new_op = rewriter.create<Op>(op.getLoc(), lhs, new_const);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), new_op.getResult(), lhs_type);
      rewriter.replaceOp(op, result.getResult());
      return mlir::success();
    }
    if (lhs_const && rhs) {
      auto new_const = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), lhs_const.getInt());
      auto new_op = rewriter.create<Op>(op.getLoc(), new_const, rhs);
      auto result = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), new_op.getResult(), lhs_type);
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
    auto lhs_type = op.lhs().getType();
    auto rhs_type = op.rhs().getType();
    if (!is_index_compatible(lhs_type, rhs_type)) {
      return mlir::failure();
    }

    auto get_cast = [](mlir::Value val) -> mlir::Value {
      if (auto op = mlir::dyn_cast_or_null<mlir::arith::IndexCastOp>(
              val.getDefiningOp())) {
        return op.getOperand();
      }
      return {};
    };

    auto get_const = [](mlir::Value val) -> mlir::IntegerAttr {
      if (auto op =
              mlir::dyn_cast_or_null<mlir::ConstantOp>(val.getDefiningOp())) {
        return op.getValue().cast<mlir::IntegerAttr>();
      }
      return {};
    };

    auto lhs = get_cast(op.lhs());
    auto rhs = get_cast(op.rhs());
    auto lhs_const = get_const(op.lhs());
    auto rhs_const = get_const(op.rhs());
    if (lhs && rhs) {
      auto new_cmp = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), op.predicate(), lhs, rhs);
      rewriter.replaceOp(op, new_cmp.getResult());
      return mlir::success();
    }
    if (lhs && rhs_const) {
      auto new_const = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), rhs_const.getInt());
      auto new_cmp = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), op.predicate(), lhs, new_const);
      rewriter.replaceOp(op, new_cmp.getResult());
      return mlir::success();
    }
    if (lhs_const && rhs) {
      auto new_const = rewriter.create<mlir::arith::ConstantIndexOp>(
          op.getLoc(), lhs_const.getInt());
      auto new_cmp = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), op.predicate(), new_const, rhs);
      rewriter.replaceOp(op, new_cmp.getResult());
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
