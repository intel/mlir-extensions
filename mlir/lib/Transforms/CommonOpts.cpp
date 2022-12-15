// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CommonOpts.hpp"

#include "imex/Transforms/IfRewrites.hpp"
#include "imex/Transforms/IndexTypePropagation.hpp"
#include "imex/Transforms/LoopRewrites.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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
    auto src = op.getMemref().getDefiningOp<mlir::memref::SubViewOp>();
    if (!src)
      return mlir::failure();

    if (!isSameRank(src.getSource().getType(), src.getType()))
      return mlir::failure();

    if (!isMixedValuesEqual(src.getMixedOffsets(), 0) ||
        !isMixedValuesEqual(src.getMixedStrides(), 1))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, src.getSource(),
                                                      op.getIndices());
    return mlir::success();
  }
};

struct SubviewStorePropagate
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getMemref().getDefiningOp<mlir::memref::SubViewOp>();
    if (!src)
      return mlir::failure();

    if (!isSameRank(src.getSource().getType(), src.getType()))
      return mlir::failure();

    if (!isMixedValuesEqual(src.getMixedOffsets(), 0) ||
        !isMixedValuesEqual(src.getMixedStrides(), 1))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, op.getValue(), src.getSource(), op.getIndices());
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

struct AndConflictSimplify
    : public mlir::OpRewritePattern<mlir::arith::AndIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AndIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<mlir::arith::CmpIOp>();
    if (!lhs)
      return mlir::failure();

    auto rhs = op.getRhs().getDefiningOp<mlir::arith::CmpIOp>();
    if (!rhs)
      return mlir::failure();

    if (lhs.getLhs() != rhs.getLhs() || lhs.getRhs() != rhs.getRhs())
      return mlir::failure();

    using Pred = mlir::arith::CmpIPredicate;
    std::array<Pred, mlir::arith::getMaxEnumValForCmpIPredicate() + 1>
        handlers{};
    handlers[static_cast<size_t>(Pred::eq)] = Pred::ne;
    handlers[static_cast<size_t>(Pred::ne)] = Pred::eq;
    handlers[static_cast<size_t>(Pred::slt)] = Pred::sge;
    handlers[static_cast<size_t>(Pred::sle)] = Pred::sgt;
    handlers[static_cast<size_t>(Pred::sgt)] = Pred::sle;
    handlers[static_cast<size_t>(Pred::sge)] = Pred::slt;
    handlers[static_cast<size_t>(Pred::ult)] = Pred::uge;
    handlers[static_cast<size_t>(Pred::ule)] = Pred::ugt;
    handlers[static_cast<size_t>(Pred::ugt)] = Pred::ule;
    handlers[static_cast<size_t>(Pred::uge)] = Pred::ult;
    if (handlers[static_cast<size_t>(lhs.getPredicate())] != rhs.getPredicate())
      return mlir::failure();

    auto val = rewriter.getIntegerAttr(op.getType(), 0);
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, val);
    return mlir::success();
  }
};

struct CmpiOfSelect : public mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    using Pred = mlir::arith::CmpIPredicate;
    auto predicate = op.getPredicate();
    if (predicate != Pred::eq && predicate != Pred::ne)
      return mlir::failure();

    for (bool reverse1 : {false, true}) {
      auto select = (reverse1 ? op.getRhs() : op.getLhs())
                        .getDefiningOp<mlir::arith::SelectOp>();
      if (!select)
        continue;

      auto other = (reverse1 ? op.getLhs() : op.getRhs());
      for (bool reverse2 : {false, true}) {
        auto selectArg(reverse2 ? select.getFalseValue()
                                : select.getTrueValue());
        if (other != selectArg)
          continue;

        bool res = static_cast<int64_t>(reverse2 != (predicate == Pred::ne));
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(op, res, 1);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

// TODO: upstream
struct ExtractStridedMetadataConstStrides
    : public mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp> {
  // Set benefit higher than ExtractStridedMetadataCast
  ExtractStridedMetadataConstStrides(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp>(
            context, /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExtractStridedMetadataOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto srcType = op.getSource().getType().cast<mlir::MemRefType>();

    int64_t offset;
    llvm::SmallVector<int64_t> strides;
    if (mlir::failed(mlir::getStridesAndOffset(srcType, strides, offset)))
      return mlir::failure();

    bool changed = false;
    auto loc = op.getLoc();
    auto replaceUses = [&](mlir::Value res, int64_t val) {
      if (mlir::ShapedType::isDynamic(val) || res.use_empty())
        return;

      changed = true;
      mlir::Value constVal = rewriter.create<mlir::arith::ConstantIndexOp>(loc, val);
      for (auto &use : llvm::make_early_inc_range(res.getUses())) {
        mlir::Operation *owner = use.getOwner();
        rewriter.updateRootInPlace(owner, [&] {
          use.set(constVal);
        });
      }
    };

    replaceUses(op.getOffset(), offset);
    for (auto [strideRes, strideVal] : llvm::zip(op.getStrides(), strides))
      replaceUses(strideRes, strideVal);

    return mlir::success(changed);
  }
};

// TODO: upstream
struct ExtractStridedMetadataCast
    : public mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExtractStridedMetadataOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cast = op.getSource().getDefiningOp<mlir::memref::CastOp>();
    if (!cast)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::ExtractStridedMetadataOp>(
        op, cast.getSource());
    return mlir::success();
  }
};

struct CommonOptsPass
    : public mlir::PassWrapper<CommonOptsPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonOptsPass)

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    imex::populateCommonOptsPatterns(patterns);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};
} // namespace

void imex::populateCanonicalizationPatterns(mlir::RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);
}

void imex::populateCommonOptsPatterns(mlir::RewritePatternSet &patterns) {
  populateCanonicalizationPatterns(patterns);

  patterns.insert<
      // clang-format off
      SubviewLoadPropagate,
      SubviewStorePropagate,
      PowSimplify,
      AndConflictSimplify,
      CmpiOfSelect,
      ExtractStridedMetadataConstStrides,
      ExtractStridedMetadataCast
      // clang-format on
      >(patterns.getContext());

  imex::populateIfRewritesPatterns(patterns);
  imex::populateLoopRewritesPatterns(patterns);
  imex::populateIndexPropagatePatterns(patterns);
}

std::unique_ptr<mlir::Pass> imex::createCommonOptsPass() {
  return std::make_unique<CommonOptsPass>();
}
