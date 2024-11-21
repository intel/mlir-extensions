//====-- Canonicalization.cpp - XeTile Canonicalization Pass  ----*- C++-*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass performs a set of canonicalization steps on XeTile ops that
/// are expected by the downstream passes. First, this will convert certain
/// vector ops (transpose, broadcast, multi_reduction) to equivalent XeTile
/// ops. Next, it will convert all XeTile ops consuming or producing
/// col-major tiles to one with row-major tiles. Finally, it will perform
/// cleanup to remove redundant ops that maybe produced by the previous
/// steps.
///
//===----------------------------------------------------------------------===//

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>
#include <math.h>

namespace imex {
#define GEN_PASS_DEF_XETILECANONICALIZATION
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace canonicalization {
template <typename T>
llvm::SmallVector<T> swapLastTwoElems(llvm::ArrayRef<T> arr) {
  assert(arr.size() >= 2 && "Shape must have at least 2 elements.");
  llvm::SmallVector<T> newArr(arr.begin(), arr.end());
  std::swap(newArr[newArr.size() - 1], newArr[newArr.size() - 2]);
  return newArr;
}

// This pattern convertes InitTileOps producing col-major tiles to equivalent
// InitTileOps producing row-major tiles. In the process, this also create a
// row-major view of the source memrefs.
struct InitTileOpPattern final
    : public mlir::OpConversionPattern<imex::xetile::InitTileOp> {
  using OpConversionPattern<imex::xetile::InitTileOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::InitTileOp initOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sourceTy = llvm::cast<mlir::MemRefType>(initOp.getSourceType());
    auto tileTy = initOp.getType();
    // Result type should have row-major `order` attribute.
    auto newTileTy = llvm::cast<imex::xetile::TileType>(
        getTypeConverter()->convertType(tileTy));
    // If source memref is statically shaped, we create a new view of the
    // memref by create a ReinterpretCastOp.
    if (sourceTy && sourceTy.hasStaticShape()) {
      auto sourceShape = sourceTy.getShape();
      llvm::SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(mlir::getStridesAndOffset(sourceTy, strides, offset)))
        return rewriter.notifyMatchFailure(initOp, "unexpected memref type.");
      // Swap the last 2 dimensions of the strides to make it row-major.
      auto newStrides = swapLastTwoElems<int64_t>(strides);
      // Create a ReinterpretCastOp
      auto newLayout =
          mlir::StridedLayoutAttr::get(getContext(), 0, newStrides);
      auto newSourceTy = mlir::MemRefType::get(
          swapLastTwoElems(sourceShape), sourceTy.getElementType(), newLayout,
          sourceTy.getMemorySpace());
      auto castOp = rewriter.create<mlir::memref::ReinterpretCastOp>(
          initOp.getLoc(), newSourceTy, initOp.getSource(), offset,
          swapLastTwoElems(sourceShape), newStrides);
      // Create a new InitTileOp with swapped offsets
      rewriter.replaceOpWithNewOp<imex::xetile::InitTileOp>(
          initOp, newTileTy, castOp,
          swapLastTwoElems<mlir::OpFoldResult>(initOp.getMixedOffsets()));
      return mlir::success();
    }
    // If the source is dynamic shaped memref, we create new InitTileOp with
    // swapped offsets, shape and strides arguments
    rewriter.replaceOpWithNewOp<imex::xetile::InitTileOp>(
        initOp, newTileTy, initOp.getSource(),
        swapLastTwoElems<mlir::OpFoldResult>(initOp.getMixedOffsets()),
        swapLastTwoElems<mlir::OpFoldResult>(initOp.getMixedSizes()),
        swapLastTwoElems<mlir::OpFoldResult>(initOp.getMixedStrides()));

    return mlir::success();
  }
};

// Pattern for rewriting UpdateTileOffsetOp to consume row-major tiles instead
// of col-major tiles.
struct UpdateTileOffsetOpPattern final
    : public mlir::OpConversionPattern<imex::xetile::UpdateTileOffsetOp> {
  using OpConversionPattern<
      imex::xetile::UpdateTileOffsetOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::UpdateTileOffsetOp updateOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tileTy = adaptor.getTile().getType();
    // Create a new update offset op with swapped offsets.
    rewriter.replaceOpWithNewOp<imex::xetile::UpdateTileOffsetOp>(
        updateOp, tileTy, adaptor.getTile(), updateOp.getOffsetY(),
        updateOp.getOffsetX(), updateOp.getIndices());
    return mlir::success();
  }
};

struct PrefetchTilePattern final
    : public mlir::OpConversionPattern<imex::xetile::PrefetchTileOp> {
  using OpConversionPattern<imex::xetile::PrefetchTileOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::PrefetchTileOp prefetchOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Create a new prefetch op.
    rewriter.replaceOpWithNewOp<imex::xetile::PrefetchTileOp>(
        prefetchOp, adaptor.getTile(), prefetchOp.getL1HintAttr(),
        prefetchOp.getL2HintAttr(), prefetchOp.getL3HintAttr());
    return mlir::success();
  }
};

// Pattern for rewriting LoadTileOp to consume row-major tiles.
struct LoadTileOpPattern final
    : public mlir::OpConversionPattern<imex::xetile::LoadTileOp> {
  using OpConversionPattern<imex::xetile::LoadTileOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::LoadTileOp loadOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // if the source type is unchanged, keep the loadOp
    if (loadOp.getSource().getType() == adaptor.getSource().getType())
      return mlir::failure();
    auto newTile = adaptor.getSource();
    auto newTileTy = llvm::cast<imex::xetile::TileType>(newTile.getType());
    mlir::VectorType newVecTy =
        mlir::VectorType::get(newTileTy.getShape(), newTileTy.getElementType());
    // Create a new loadOp.
    mlir::Value newOp = rewriter.create<imex::xetile::LoadTileOp>(
        loadOp.getLoc(), newVecTy, newTile);
    // Transpose the output of the load so that we get the return type of the
    // original loadOp
    rewriter.replaceOpWithNewOp<imex::xetile::TransposeOp>(
        loadOp, loadOp.getType(), newOp, mlir::ArrayRef<int64_t>({1, 0}));
    return mlir::success();
  }
};

// If ScfForOp has any col-major tiles as iterArgs, we rewrite it to use
// row-major tiles.
struct ScfForOpPattern final
    : public mlir::OpConversionPattern<mlir::scf::ForOp> {
  using OpConversionPattern<mlir::scf::ForOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp forOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> convertedArgs;

    convertedArgs.append(adaptor.getInitArgs().begin(),
                         adaptor.getInitArgs().end());
    auto newOp = rewriter.create<mlir::scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), convertedArgs);

    mlir::TypeConverter::SignatureConversion signatureConverter(
        forOp.getRegion().getNumArguments());
    for (size_t i = 0; i < forOp.getRegion().getNumArguments(); i++) {
      signatureConverter.addInputs(i,
                                   newOp.getRegion().getArgument(i).getType());
    }
    rewriter.applySignatureConversion(&forOp.getRegion().getBlocks().front(),
                                      signatureConverter, getTypeConverter());
    rewriter.eraseBlock(newOp.getBody());
    rewriter.inlineRegionBefore(forOp.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(forOp, newOp.getResults());
    return mlir::success();
  }
};

struct ScfYieldOpPattern final
    : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
  using OpConversionPattern<mlir::scf::YieldOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp yieldOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> convertedArgs;
    convertedArgs.append(adaptor.getOperands().begin(),
                         adaptor.getOperands().end());
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yieldOp, convertedArgs);
    return mlir::success();
  }
};

// This is a canonicalization pattern that rewrites vector TransposeOp to an
// equivalent xetile TransposeOp.
struct VectorTransposeToXetileTransposeOpPattern
    : public mlir::OpRewritePattern<mlir::vector::TransposeOp> {
  using mlir::OpRewritePattern<mlir::vector::TransposeOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getVector().getType().getRank() != 2)
      return mlir::failure();
    // Retain discardable attributes if any.
    llvm::SmallVector<mlir::NamedAttribute> discardableAttrs(
        op->getDiscardableAttrs().begin(), op->getDiscardableAttrs().end());
    // Create an equivalent XeTileTransposeOp
    auto newOp = rewriter.replaceOpWithNewOp<imex::xetile::TransposeOp>(
        op, op.getType(), op.getVector(), op.getPermutation());
    newOp->setDiscardableAttrs(discardableAttrs);
    return mlir::success();
  }
};

// Canonicalization pattern that rewrites vector BroadCastOp to an equivalent
// xetile BroadcastOp.
struct VectorBroadcastToXetileBroadcastOpPattern
    : public mlir::OpRewritePattern<mlir::vector::BroadcastOp> {
  using mlir::OpRewritePattern<mlir::vector::BroadcastOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::BroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy = op.getResultVectorType();
    auto resultRank = resultTy.getRank();
    // If result is not 2D, keep the vector op.
    if (resultRank != 2)
      return mlir::failure();
    // If source is scalar, keep the vector op.
    if (!llvm::isa<mlir::VectorType>(op.getSourceType()))
      return mlir::failure();
    auto sourceVectorTy = llvm::cast<mlir::VectorType>(op.getSourceType());
    auto sourceRank = sourceVectorTy.getRank();
    auto sourceShape = sourceVectorTy.getShape();
    // Retain the discardable attributes if any.
    llvm::SmallVector<mlir::NamedAttribute> discardableAttrs(
        op->getDiscardableAttrs().begin(), op->getDiscardableAttrs().end());
    // If the source rank is 1 and result rank is 2, we need to create a shape
    // cast to convert source to 2D and then create a xetile.broadcast. In this
    // case, broadcast dimension is 0 according to vector.broadcast definition.
    if (sourceRank < resultRank) {
      auto source2DTy =
          mlir::VectorType::get(llvm::ArrayRef<int64_t>({1, sourceShape[0]}),
                                resultTy.getElementType());
      auto source2D = rewriter.create<mlir::vector::ShapeCastOp>(
          op.getLoc(), source2DTy, op.getSource());
      source2D->setDiscardableAttrs(discardableAttrs);
      auto newOp = rewriter.replaceOpWithNewOp<imex::xetile::BroadcastOp>(
          op, resultTy, source2D, llvm::ArrayRef<int64_t>({0}));
      newOp->setDiscardableAttrs(discardableAttrs);
      return mlir::success();
    }
    // If ranks are same, decide the broadcast dimension based on the source
    // vector shape.
    auto broadcastDim = (sourceShape[0] == 1) ? 0 : 1;
    auto newOp = rewriter.replaceOpWithNewOp<imex::xetile::BroadcastOp>(
        op, resultTy, op.getSource(), llvm::ArrayRef<int64_t>({broadcastDim}));
    newOp->setDiscardableAttrs(discardableAttrs);
    return mlir::success();
  }
};

// Canonicalization pattern that rewrites vector MultiDimReduction ops to an
// equivalent xetile ReduceOp.
struct VectorMultiReductionToXeTileReduce
    : public mlir::OpRewritePattern<mlir::vector::MultiDimReductionOp> {
  using mlir::OpRewritePattern<
      mlir::vector::MultiDimReductionOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::MultiDimReductionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // If the source is not 2D we can not convert it to xetile.reduce.
    auto sourceTy = op.getSourceVectorType();
    if (sourceTy.getRank() != 2)
      return mlir::failure();
    // If result is not 1D, we can not convert it to xetile.reduce. This
    // requires that the reduction dimensions has rank 1.
    auto reductionDims = op.getReductionDims();
    if (reductionDims.size() != 1)
      return mlir::failure();
    // Retain discardable attributes if any.
    llvm::SmallVector<mlir::NamedAttribute> discardableAttrs(
        op->getDiscardableAttrs().begin(), op->getDiscardableAttrs().end());
    // Create an equivalent XeTileReduceOp
    int64_t reduceDim = reductionDims[0];
    auto resultTy = llvm::cast<mlir::VectorType>(op.getType());
    auto xetileResultTy = mlir::VectorType::get(
        (reduceDim == 0 ? llvm::ArrayRef<int64_t>({1, resultTy.getDimSize(0)})
                        : llvm::ArrayRef<int64_t>({resultTy.getDimSize(0), 1})),
        resultTy.getElementType());
    auto reduceOp = rewriter.create<imex::xetile::ReductionOp>(
        op->getLoc(), xetileResultTy, op.getKind(), op.getSource(),
        mlir::ArrayRef<int64_t>({reduceDim}));
    reduceOp->setDiscardableAttrs(discardableAttrs);
    // Shape cast the result back to original shape.
    auto shapeCastOp = rewriter.create<mlir::vector::ShapeCastOp>(
        op->getLoc(), resultTy, reduceOp.getResult());
    shapeCastOp->setDiscardableAttrs(discardableAttrs);
    // Finally add the result to the accumulator.
    if (llvm::isa<mlir::IntegerType>(sourceTy.getElementType())) {
      auto accOp = rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
          op, shapeCastOp, op.getAcc());
      accOp->setDiscardableAttrs(discardableAttrs);
    } else {
      auto accOp = rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(
          op, shapeCastOp, op.getAcc());
      accOp->setDiscardableAttrs(discardableAttrs);
    }
    return mlir::success();
  }
};

// XeTile canonicalization can result in back to back transpose operations that
// are redundant. This pattern detects such cases and removes the redundant
// transpose operations.
struct RemoveRedundantTransposeOpPattern
    : public mlir::OpRewritePattern<imex::xetile::TransposeOp> {
  using mlir::OpRewritePattern<imex::xetile::TransposeOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto sourceTransposeOp =
        llvm::dyn_cast_if_present<imex::xetile::TransposeOp>(
            op.getVector().getDefiningOp());
    // If the source is not another transpose we do not care.
    if (!sourceTransposeOp)
      return mlir::failure();
    // Check if the two back to back transpose ops are 2D true transpose ops.
    if (op.getPermutation() != llvm::ArrayRef<int64_t>({1, 0}))
      return mlir::failure();
    if (sourceTransposeOp.getPermutation() != llvm::ArrayRef<int64_t>({1, 0}))
      return mlir::failure();
    // Replace the current transpose op with the source of the source transpose
    // and remove it.
    op.getResult().replaceAllUsesWith(sourceTransposeOp.getVector());
    rewriter.eraseOp(op);
    // This makes the source transpose op dead. Remove it.
    if (sourceTransposeOp->getUsers().empty())
      rewriter.eraseOp(sourceTransposeOp);
    return mlir::success();
  }
};

struct XeTileCanonicalizationPass final
    : public imex::impl::XeTileCanonicalizationBase<
          XeTileCanonicalizationPass> {

  void runOnOperation() override {
    auto *context = &getContext();
    // First convert any 2D vector.transpose ops into equivalent
    // xetile.transpose operations. This makes the subsequent transformations
    // simpler.
    {
      mlir::RewritePatternSet patterns(context);
      mlir::GreedyRewriteConfig config;
      config.enableRegionSimplification =
          mlir::GreedySimplifyRegionLevel::Disabled;
      config.useTopDownTraversal = true;
      config.strictMode = mlir::GreedyRewriteStrictness::ExistingAndNewOps;
      patterns.add<VectorTransposeToXetileTransposeOpPattern,
                   VectorBroadcastToXetileBroadcastOpPattern,
                   VectorMultiReductionToXeTileReduce>(context);

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns), config))) {
        return signalPassFailure();
      }
    }
    {
      mlir::TypeConverter typeConverter;
      mlir::RewritePatternSet patterns(context);
      mlir::ConversionTarget target(*context);
      auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                  mlir::ValueRange inputs, mlir::Location loc) {
        auto cast =
            builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
        return cast.getResult(0);
      };
      typeConverter.addConversion([](mlir::Type type) { return type; });
      typeConverter.addConversion([](imex::xetile::TileType tileTy) {
        if (tileTy.getOrder().asArrayRef() == mlir::ArrayRef({0, 1})) {
          auto newAttr = imex::xetile::XeTileAttr::get(
              tileTy.getContext(), tileTy.getSgMap(), tileTy.getWgMap(),
              mlir::DenseI32ArrayAttr::get(tileTy.getContext(), {1, 0}),
              tileTy.getInnerBlocks(), tileTy.getMemorySpace(),
              tileTy.getScatterAttr());

          return imex::xetile::TileType::get(
              swapLastTwoElems(tileTy.getShape()), tileTy.getElementType(),
              newAttr);
        }
        return tileTy;
      });

      typeConverter.addArgumentMaterialization(addUnrealizedCast);
      typeConverter.addSourceMaterialization(addUnrealizedCast);
      typeConverter.addTargetMaterialization(addUnrealizedCast);

      target.addLegalOp<mlir::memref::ReinterpretCastOp>();
      target.addLegalOp<imex::xetile::TransposeOp>();
      // Col-major tile creattion is not allowed.
      target.addDynamicallyLegalOp<
          imex::xetile::InitTileOp>([&](imex::xetile::InitTileOp op) {
        return op.getType().getOrder().asArrayRef() != mlir::ArrayRef({0, 1});
      });
      // UpdateTileOffsetOp is legal if it does not consume col-major tiles.
      target.addDynamicallyLegalOp<
          imex::xetile::UpdateTileOffsetOp>([&](imex::xetile::UpdateTileOffsetOp
                                                    op) {
        return op.getType().getOrder().asArrayRef() != mlir::ArrayRef({0, 1});
      });
      // PrefetchTileOp is legal if it does not consume col-major tiles.
      target.addDynamicallyLegalOp<imex::xetile::PrefetchTileOp>(
          [&](imex::xetile::PrefetchTileOp op) {
            return op.getTile().getType().getOrder().asArrayRef() !=
                   mlir::ArrayRef({0, 1});
          });
      // LoadTileOp is legal if it does not consume col-major tiles.
      target.addDynamicallyLegalOp<imex::xetile::LoadTileOp>(
          [&](imex::xetile::LoadTileOp op) {
            return op.getSource().getType().getOrder().asArrayRef() !=
                   mlir::ArrayRef({0, 1});
          });
      // If any iterArg of the forOp is a col-major tile, it is illegal.
      target.addDynamicallyLegalOp<mlir::scf::ForOp>([&](mlir::scf::ForOp op) {
        for (auto arg : op.getRegionIterArgs()) {
          auto tileTy =
              llvm::dyn_cast_if_present<imex::xetile::TileType>(arg.getType());
          if (tileTy &&
              tileTy.getOrder().asArrayRef() == mlir::ArrayRef({0, 1}))
            return false;
        }
        return true;
      });
      // yieldOp can not return col-major tiles
      target.addDynamicallyLegalOp<mlir::scf::YieldOp>(
          [&](mlir::scf::YieldOp op) {
            for (auto arg : op.getOperands()) {
              auto tileTy = llvm::dyn_cast_if_present<imex::xetile::TileType>(
                  arg.getType());
              if (tileTy &&
                  tileTy.getOrder().asArrayRef() == mlir::ArrayRef({0, 1}))
                return false;
            }
            return true;
          });
      patterns
          .add<InitTileOpPattern, LoadTileOpPattern, UpdateTileOffsetOpPattern,
               PrefetchTilePattern, ScfForOpPattern, ScfYieldOpPattern>(
              typeConverter, context);

      if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                    std::move(patterns))))
        return signalPassFailure();
    }

    // This canonicalization can result in back to back transpose operation that
    // cancels out. These can be eliminated.
    {
      mlir::RewritePatternSet patterns(context);
      mlir::GreedyRewriteConfig config;
      config.enableRegionSimplification =
          mlir::GreedySimplifyRegionLevel::Disabled;
      config.useTopDownTraversal = true;
      config.strictMode = mlir::GreedyRewriteStrictness::ExistingAndNewOps;
      patterns.add<RemoveRedundantTransposeOpPattern>(context);

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns), config))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace canonicalization

namespace imex {
std::unique_ptr<mlir::Pass> createXeTileCanonicalizationPass() {
  return std::make_unique<canonicalization::XeTileCanonicalizationPass>();
}
} // namespace imex
