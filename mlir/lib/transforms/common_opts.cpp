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

#include "mlir-extensions/transforms/common_opts.hpp"

#include "mlir-extensions/transforms/cse.hpp"
#include "mlir-extensions/transforms/if_rewrites.hpp"
#include "mlir-extensions/transforms/index_type_propagation.hpp"
#include "mlir-extensions/transforms/loop_rewrites.hpp"
#include "mlir-extensions/transforms/memory_rewrites.hpp"

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

namespace {
static bool isSameRank(mlir::Type type1, mlir::Type type2) {
  auto shaped1 = type1.dyn_cast<mlir::ShapedType>();
  if (!shaped1)
    return false;

  auto shaped2 = type2.dyn_cast<mlir::ShapedType>();
  if (!shaped2)
    return false;

  if (!shaped1.hasRank() || !shaped2.hasRank())
    return false;

  return shaped1.getRank() == shaped2.getRank();
}

static bool isMixedValuesEqual(llvm::ArrayRef<mlir::OpFoldResult> values,
                               int64_t expectedVal) {
  for (auto val : values) {
    auto intVal = mlir::getConstantIntValue(val);
    if (!intVal || *intVal != expectedVal)
      return false;
  }
  return true;
}

struct SubviewLoadPropagate
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.memref().getDefiningOp<mlir::memref::SubViewOp>();
    if (!src)
      return mlir::failure();

    if (!isSameRank(src.source().getType(), src.getType()))
      return mlir::failure();

    if (!isMixedValuesEqual(src.getMixedOffsets(), 0) ||
        !isMixedValuesEqual(src.getMixedStrides(), 1))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, src.source(),
                                                      op.indices());
    return mlir::success();
  }
};

struct SubviewStorePropagate
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.memref().getDefiningOp<mlir::memref::SubViewOp>();
    if (!src)
      return mlir::failure();

    if (!isSameRank(src.source().getType(), src.getType()))
      return mlir::failure();

    if (!isMixedValuesEqual(src.getMixedOffsets(), 0) ||
        !isMixedValuesEqual(src.getMixedStrides(), 1))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, op.value(), src.source(), op.indices());
    return mlir::success();
  }
};

struct PowSimplify : public mlir::OpRewritePattern<mlir::math::PowFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::math::PowFOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    mlir::FloatAttr constValue;
    if (!mlir::matchPattern(rhs, mlir::m_Constant(&constValue)))
      return mlir::failure();

    assert(constValue);
    auto val = constValue.getValueAsDouble();
    if (val == 1.0) {
      rewriter.replaceOp(op, lhs);
      return mlir::success();
    }
    if (val == 2.0) {
      rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, lhs, lhs);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

void plier::populateCanonicalizationPatterns(
    mlir::MLIRContext &context, mlir::RewritePatternSet &patterns) {
  for (auto *dialect : context.getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (auto op : context.getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, &context);
}

void plier::populateCommonOptsPatterns(mlir::MLIRContext &context,
                                       mlir::RewritePatternSet &patterns) {
  populateCanonicalizationPatterns(context, patterns);

  patterns.insert<
      // clang-format off
//      LoopInvariantCodeMotion, TODO
      plier::CmpLoopBoundsSimplify,
      plier::IfOpConstCond,
      plier::CSERewrite<mlir::FuncOp, /*recusive*/ false>,
      SubviewLoadPropagate,
      SubviewStorePropagate,
      PowSimplify
      // clang-format on
      >(&context);

  plier::populate_index_propagate_patterns(context, patterns);
}
