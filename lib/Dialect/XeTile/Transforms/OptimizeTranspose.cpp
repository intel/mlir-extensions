//===--------- OptimizeTranspose.cpp - OptimizeLoads Pass  --------------*- C++-
//*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains OptimizeTranspose pass. This pass detects and optimizes
/// the xetile loads that are transposed and then used in a MMA operation as the
/// B operand. These op patterns are rewritten to use the order attribute to
/// merge the transpose and load operations in a single tile load operation.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/AddDiscriminators.h"

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Transforms/Passes.h"
#include "imex/Utils/XeCommon.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <math.h>
#include <numeric>

namespace imex {
#define GEN_PASS_DECL_XETILEOPTIMIZETRANSPOSE
#define GEN_PASS_DEF_XETILEOPTIMIZETRANSPOSE
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace optimizetranspose {

void addBatchStrides(llvm::SmallVectorImpl<int64_t> &strides,
                     llvm::ArrayRef<int64_t> shape) {
  auto innerDims = shape.take_back(strides.size());
  int64_t stride = std::accumulate(innerDims.begin(), innerDims.end(), 1,
                                   std::multiplies<int64_t>());
  for (int64_t dimSize : shape.drop_back(strides.size())) {
    strides.insert(strides.begin(), stride);
    stride *= dimSize;
  }
}

// This patterns rewrites the InitTileOp to traverse the source memref in a
// transposed view instead of the originla view. This eliminates the need for
// vector transpose.
// clang-format off
// Example:
// %0 = xetile.init_tile %src [%c64, %c128] ... : tile<32x16xf16>
// becomes:
// %1 = memref.reinterpret_cast %src : memref<1024x512xf16> to memref<512x1024xf16, strided<1, 512>>
// %2 = xetile.init_tile %1 [%c128, %c64] ... : tile<32x16xf16, order=[0, 1]>
// clang-format on
struct InitTileOpPattern final
    : public mlir::OpConversionPattern<imex::xetile::InitTileOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::InitTileOp initOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(initOp.isSourceMemRef() && initOp.sourceMemRefHasStaticShape() &&
           "source must be a static memref");
    auto sourceTy = llvm::cast<mlir::MemRefType>(initOp.getSourceType());
    auto sourceShape = sourceTy.getShape();
    auto sourceRank = sourceShape.size();
    auto tileTy = initOp.getType();
    // If the source memref is row-major, we need to convert it to col-major
    // else, we convert it to row-major
    llvm::SmallVector<int64_t, 2> newStrides = {sourceShape[sourceRank - 2],
                                                1}; // to row-major
    bool sourceIsRowMajor = sourceTy.getLayout().isIdentity();
    if (sourceIsRowMajor)
      newStrides = {1, sourceShape[sourceRank - 1]}; // to col-major
    addBatchStrides(newStrides, sourceShape);

    // Convert the view of the source memref by creating a ReinterpretCastOp
    auto newLayout = mlir::StridedLayoutAttr::get(getContext(), 0, newStrides);
    auto newSourceTy = mlir::MemRefType::get(
        imex::swapLastTwoElements(sourceShape), sourceTy.getElementType(),
        newLayout, sourceTy.getMemorySpace());
    auto castOp = rewriter.create<mlir::memref::ReinterpretCastOp>(
        initOp.getLoc(), newSourceTy, initOp.getSource(), (int64_t)(0),
        imex::swapLastTwoElements(sourceShape), newStrides);

    // Create a new initTileOp with the new source by using the order attribute
    auto newTileAttr = imex::xetile::XeTileAttr::get(
        getContext(), tileTy.getSgMap(), tileTy.getWgMap(),
        (sourceIsRowMajor ? mlir::DenseI32ArrayAttr::get(getContext(), {0, 1})
                          : mlir::DenseI32ArrayAttr::get(getContext(), {1, 0})),
        tileTy.getInnerBlocks(), tileTy.getWgData());
    auto transposedTileTy = imex::xetile::TileType::get(
        imex::swapLastTwoElements(initOp.getType().getShape()),
        initOp.getElementType(), newTileAttr);
    auto offsets = llvm::SmallVector<mlir::OpFoldResult>(
        initOp.getOffsets().begin(), initOp.getOffsets().end());
    // Offsets of the original InitTileOp must be reversed.
    std::swap(offsets[sourceRank - 1], offsets[sourceRank - 2]);
    mlir::Value newOp = rewriter.create<imex::xetile::InitTileOp>(
        initOp.getLoc(), transposedTileTy, castOp, offsets);

    // Replace all uses.
    rewriter.replaceAllUsesWith(initOp.getResult(), newOp);
    rewriter.eraseOp(initOp);
    return mlir::success();
  }
};

struct LoadTileOpPattern final
    : public mlir::OpConversionPattern<imex::xetile::LoadTileOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::LoadTileOp loadOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sourceOp = adaptor.getSource().getDefiningOp();
    mlir::Value updatedSource;
    imex::xetile::TileType updatedTileTy;
    mlir::VectorType outVecTy;
    // LoadTileOp is inside the ForOp. Therefore the source is
    // UnrealizedConversionCastOp produced by the ForOp signature conversion.
    // TODO: direct uses of the InitTileOp should be supported i.e. more
    // patterns.
    if (llvm::isa<mlir::UnrealizedConversionCastOp>(sourceOp)) {
      auto castOp = llvm::cast<mlir::UnrealizedConversionCastOp>(sourceOp);
      updatedSource = castOp.getInputs()[0];
    } else {
      assert(false && "unsupported source op");
    }
    updatedTileTy = llvm::cast<imex::xetile::TileType>(updatedSource.getType());
    outVecTy = mlir::VectorType::get(updatedTileTy.getShape(),
                                     updatedTileTy.getElementType());
    // Create a new LoadOp that uses the source of UnrealizedConversionCastOp.
    auto newOp = rewriter.create<imex::xetile::LoadTileOp>(
        loadOp.getLoc(), outVecTy, updatedSource);
    rewriter.replaceAllUsesWith(loadOp.getResult(), newOp.getResult());
    rewriter.eraseOp(loadOp);
    return mlir::success();
  }
};

struct VectorTransposeOpPattern final
    : public mlir::OpConversionPattern<mlir::vector::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::TransposeOp transposeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(transposeOp.getSourceVectorType() ==
               transposeOp.getResultVectorType() &&
           "transpose must be a noop");
    // Transpose is a noop. Replace all uses of the transpose with the vector.
    rewriter.replaceAllUsesWith(transposeOp.getResult(),
                                transposeOp.getVector());
    rewriter.eraseOp(transposeOp);
    return mlir::success();
  }
};

// This pattern is used to convert the ForOp signature to accomodate the new
// TileType introduced by the InitTileOp pattern. UnrealizedConversionCast ops
// are inserted by the signature conversion inside the body of the ForOp to
// reconcile the new types.
// clang-format off
// Example:
// %0 = xetile.init_tile ... : tile<32x16xf16>
// scf.for %arg0 = %c0 to %c512 step %c16 iter_args(%arg0 : %0, ...) -> (tile<32x16xf16>,
// Becomes:
// %0 = xetile.init_tile ... : tile<16x32xf16, order=[0, 1]>
// scf.for %arg0 = %c0 to %c512 step %c16 iter_args(%arg0 : tile<16x32xf16, order=[0, 1]>, ...)
//  -> (tile<16x32xf16, order=[0, 1]>, ... {
//   %1 = unrealized_conversion_cast %0 : tile<16x32xf16, order=[0, 1]> to tile<32x16xf16>
// clang-format on
struct ScfForOpPattern final
    : public mlir::OpConversionPattern<mlir::scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
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
    // Signature conversion will insert UnrealizedConversionCastOp inside the
    // body of ForOp to convert the new InitTileOp type to the original
    // InitTileOp type. This is cleaned up later.
    rewriter.applySignatureConversion(&forOp.getRegion(), signatureConverter,
                                      getTypeConverter());
    rewriter.eraseBlock(newOp.getBody());
    rewriter.inlineRegionBefore(forOp.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(forOp, newOp.getResults());
    return mlir::success();
  }
};

// UdapteTileOffsetOp is converted to use the new TileType introduced by the
// InitTileOp pattern. Because the traversal order of the source memref is
// changed now, we need to swap the offsets.
// clang-format off
// Example:
// %0 = xetile.update_tile_offset %in [%c0, %c16] ... : tile<32x16xf16>
// becomes:
// %1 = xetile.update_tile_offset %in [%c16, %c0] ... : tile<32x16xf16, order=[0, 1]>
// clang-format on
struct UpdateTileOffsetOpPattern final
    : public mlir::OpConversionPattern<imex::xetile::UpdateTileOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(imex::xetile::UpdateTileOffsetOp updateOffsetOp,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sourceOp = adaptor.getTile().getDefiningOp();
    mlir::Value updatedSource;
    // UpdateTileOffsetOp is inside the ForOp. Therefore the source is an
    // UnrealizedConversionCastOp produced by the ForOp signature conversion.
    // TODO: direct uses of the InitTileOp should be supported.
    if (auto castOp =
            llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(sourceOp))
      updatedSource = castOp.getInputs()[0];
    else
      assert(false && "unsupported source op");

    auto newOp = rewriter.create<imex::xetile::UpdateTileOffsetOp>(
        updateOffsetOp.getLoc(), updatedSource.getType(), updatedSource,
        adaptor.getOffsetY(), adaptor.getOffsetX());
    rewriter.replaceAllUsesWith(updateOffsetOp.getResult(), newOp.getResult());
    rewriter.eraseOp(updateOffsetOp);
    return mlir::success();
  }
};

struct ScfYieldOpPattern final
    : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp yieldOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<mlir::scf::YieldOp>(yieldOp.getLoc(),
                                                     adaptor.getResults());
    rewriter.replaceOp(yieldOp, newOp);
    return mlir::success();
  }
};

// Progslice represents a slice of the program that is a candidate for
// optimization.
struct ProgSlice {
private:
  llvm::DenseSet<mlir::Operation *> ops;
  ProgSlice() = default;

public:
  static ProgSlice create(llvm::ArrayRef<mlir::Operation *> ops) {
    ProgSlice slice;
    slice.ops.insert(ops.begin(), ops.end());
    return slice;
  }
  bool contains(mlir::Operation *op) { return ops.count(op); }
};

// Helper function to analyze the def-use chain of initTileOps. Currently we
// pattern match the following def-use chain as a candidate for transformation.
// init_tile -> scf.for -> load_tile -> vector.transpose -> tile_mma
//                |
//                -> update_tile_offset -> scf.yield
void analyzeInitTileOps(mlir::Operation *op,
                        llvm::SmallVector<ProgSlice> &candidates) {

  op->walk([&](imex::xetile::InitTileOp initOp) -> mlir::WalkResult {
    llvm::SmallVector<mlir::Operation *> ops;
    // TODO: Add support for initTileOps using sources other than static memrefs
    if (!initOp.isSourceMemRef())
      return mlir::WalkResult::skip();
    if (!initOp.sourceMemRefHasStaticShape())
      return mlir::WalkResult::skip();

    // Ignore initTileOps with more than one use
    if (!initOp->hasOneUse())
      return mlir::WalkResult::skip();
    ops.push_back(initOp);
    auto user = *initOp->user_begin();
    // InitTileOp must be consumed by a ForOp
    mlir::Operation *loadUser = nullptr, *updateOffsetUser = nullptr;
    if (auto scfFor = llvm::dyn_cast_if_present<mlir::scf::ForOp>(user)) {
      auto argument = imex::getArgForOperand(scfFor, initOp.getResult());
      int userCount = 0;
      for (auto user : argument.getUsers()) {
        userCount++;
        if (llvm::isa<imex::xetile::LoadTileOp>(user)) {
          loadUser = user;
          ops.push_back(scfFor);
          ops.push_back(user);
        } else if (llvm::isa<imex::xetile::UpdateTileOffsetOp>(user)) {
          updateOffsetUser = user;
          ops.push_back(scfFor);
          ops.push_back(user);
        }
      }
      // ForOp argument should have only two users, a load and an update offset
      if (userCount != 2 || !(loadUser && updateOffsetUser))
        return mlir::WalkResult::skip();
    } else
      return mlir::WalkResult::skip();

    // LoadOp must be consumed by a transpose
    if (!(loadUser->hasOneUse() &&
          llvm::isa<mlir::vector::TransposeOp>(*loadUser->user_begin())))
      return mlir::WalkResult::skip();
    auto transposeOp =
        llvm::cast<mlir::vector::TransposeOp>(*loadUser->user_begin());
    ops.push_back(transposeOp);
    // Check if the transpose has only one user and that user is a TileMMAOp
    if (!(transposeOp->hasOneUse() &&
          llvm::isa<imex::xetile::TileMMAOp>(*transposeOp->user_begin())))
      return mlir::WalkResult::skip();
    auto mmaOp =
        llvm::cast<imex::xetile::TileMMAOp>(*transposeOp->user_begin());
    // Make sure the transpose is the second operand of the mma
    if (mmaOp.getB().getDefiningOp() != transposeOp)
      return mlir::WalkResult::skip();

    // Check if update offset is consumed by a yield
    if (!updateOffsetUser->hasOneUse())
      return mlir::WalkResult::skip();
    auto yieldOp = *updateOffsetUser->user_begin();
    if (!llvm::isa<mlir::scf::YieldOp>(yieldOp))
      return mlir::WalkResult::skip();
    ops.push_back(yieldOp);

    // At this point, we have a candidate def-use chain for optimization.
    auto slice = ProgSlice::create(ops);
    candidates.push_back(slice);
    return mlir::WalkResult::advance();
  });
}

struct XeTileOptimizeTransposePass final
    : public imex::impl::XeTileOptimizeTransposeBase<
          XeTileOptimizeTransposePass> {

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::Operation *op = getOperation();
    llvm::SmallVector<ProgSlice> candidates;
    // Run the analysis to find the candidates for the transformation
    analyzeInitTileOps(op, candidates);

    mlir::TypeConverter typeConverter;
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);
    auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs, mlir::Location loc) {
      auto cast =
          builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<mlir::Value>(cast.getResult(0));
    };
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addArgumentMaterialization(addUnrealizedCast);
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<mlir::memref::ReinterpretCastOp>();
    target.addLegalOp<imex::xetile::TileMMAOp>();

    // An op is transformed if it is part of candidate prog slices
    target.addDynamicallyLegalOp<
        imex::xetile::InitTileOp, imex::xetile::LoadTileOp, mlir::scf::ForOp,
        mlir::vector::TransposeOp, imex::xetile::UpdateTileOffsetOp,
        mlir::scf::YieldOp>([&](mlir::Operation *op) {
      for (auto candidate : candidates) {
        if (candidate.contains(op))
          return false;
      }
      return true;
    });

    patterns.add<InitTileOpPattern, LoadTileOpPattern, VectorTransposeOpPattern,
                 ScfForOpPattern, UpdateTileOffsetOpPattern, ScfYieldOpPattern>(
        typeConverter, context);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace optimizetranspose

namespace imex {
std::unique_ptr<mlir::Pass> createXeTileOptimizeTransposePass() {
  return std::make_unique<optimizetranspose::XeTileOptimizeTransposePass>();
}
} // namespace imex
