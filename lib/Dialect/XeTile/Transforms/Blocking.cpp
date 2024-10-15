//===-------------- Blocking.cpp --------- Blocking Pass  -------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains lowering transformation for determing the problem size
/// that can be handled by an XeGPU operator (hardware instruction). XeTile
/// program can work one bigger problem size that cannot be handled by a
/// hardware instruction. But it needs to be decomposed into smaller pieces
/// such that each pieces can be handled by a hardware instruction.
///
//===----------------------------------------------------------------------===//
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <optional>
#include <set>
#include <tuple>

#include "imex/Dialect/XeTile/Transforms/BlockingAnalysis.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/XeArch.h"

using namespace mlir;
using namespace llvm;
using namespace imex;
namespace imex {
#define GEN_PASS_DEF_XETILEBLOCKING
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {
namespace Blocking {

static xetile::TileUnpackOp
addUnpackOp(mlir::Value src, mlir::ConversionPatternRewriter &rewriter) {
  auto srcTy = llvm::dyn_cast_if_present<mlir::VectorType>(src.getType());
  assert(srcTy && srcTy.getRank() == 4);
  auto shape = srcTy.getShape();
  auto grids = shape.take_front(2);
  auto innerBlocks = shape.take_back(2);
  llvm::SmallVector<int64_t> unpackShape(
      {grids[0] * innerBlocks[0], grids[1] * innerBlocks[1]});

  auto unpackTy = mlir::VectorType::get(unpackShape, srcTy.getElementType());
  return rewriter.create<xetile::TileUnpackOp>(
      src.getLoc(), unpackTy, src,
      mlir::DenseI64ArrayAttr::get(src.getContext(), innerBlocks));
}

static mlir::Value addPackOp(mlir::Value src,
                             llvm::ArrayRef<int64_t> targetBlkSizes,
                             mlir::ConversionPatternRewriter &rewriter) {
  auto srcTy = mlir::dyn_cast<mlir::VectorType>(src.getType());
  assert(srcTy && targetBlkSizes.size() == 2);
  auto shape = srcTy.getShape();
  llvm::SmallVector<int64_t> packShape({shape[0] / targetBlkSizes[0],
                                        shape[1] / targetBlkSizes[1],
                                        targetBlkSizes[0], targetBlkSizes[1]});

  auto packTy = mlir::VectorType::get(packShape, srcTy.getElementType());
  auto packOp = rewriter.create<xetile::TilePackOp>(
      src.getLoc(), packTy, src,
      mlir::DenseI64ArrayAttr::get(src.getContext(), targetBlkSizes));
  return packOp;
}

/// OpConversionPatternWithAnalysis is a wrapper around OpConversionPattern
/// but takes an extra AnalysisT object as an argument, such that patterns
/// can leverage the analysis results.
template <typename SourceOp, typename AnalysisT>
class OpConversionPatternWithAnalysis
    : public mlir::OpConversionPattern<SourceOp> {
public:
  using OpPatternRewriter = typename mlir::ConversionPatternRewriter;

  OpConversionPatternWithAnalysis(mlir::MLIRContext *context,
                                  AnalysisT &analysis)
      : mlir::OpConversionPattern<SourceOp>(context), analysis(analysis) {}

protected:
  AnalysisT &analysis;
};

/// OpTraitConversionPatternWithAnalysis is a wrapper around
/// OpTraitConversionPattern but takes an extra AnalysisT object as an argument,
/// such that patterns can leverage the analysis results.
template <template <typename> class TraitType, typename AnalysisT>
class OpTraitConversionPatternWithAnalysis
    : public mlir::OpTraitConversionPattern<TraitType> {
public:
  OpTraitConversionPatternWithAnalysis(mlir::MLIRContext *context,
                                       AnalysisT &analysis,
                                       PatternBenefit benefit = 1)
      : mlir::OpTraitConversionPattern<TraitType>(context, benefit),
        analysis(analysis) {}

protected:
  AnalysisT &analysis;
};

// It blocks/extends a 2D constant dense vector into a
// 4D vector with the last 2 dim corresponding to block size.
// which is chosed based on requests from its users.
// example: arith.constant dense<0.0>: vector<32x32xf16>
//      --> arith.constant dense<0.0>: vector<4x2x8x16xf16>
// assuming [8, 16] is the block size.
struct ArithConstantOpPattern
    : public OpConversionPatternWithAnalysis<mlir::arith::ConstantOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      mlir::arith::ConstantOp,
      BlockingAnalysis>::OpConversionPatternWithAnalysis;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());

    // TODO: it maybe unstable to determine whether doing blocking or not
    //  for a constant op simply based on its 2D shape.
    if (!value || value.getType().getRank() != 2)
      return mlir::failure();

    auto blockSize = analysis.getDefBlockSize(op.getResult());
    if (!blockSize)
      return rewriter.notifyMatchFailure(op, "Invalid block size.");

    auto shape = value.getType().getShape();
    auto newTy =
        mlir::VectorType::get({shape[0] / blockSize[0], shape[1] / blockSize[1],
                               blockSize[0], blockSize[1]},
                              value.getElementType());

    // TODO: it is logically incorrect to reshape a dense value.
    // it doesn't show the impact of pack effect. It works on some
    // cases in which all elements has the same value, but not general.
    value = value.reshape(newTy);
    auto loc = op.getLoc();
    auto newOp = rewriter.create<mlir::arith::ConstantOp>(loc, value);
    auto unpack = addUnpackOp(newOp, rewriter);

    rewriter.replaceOp(op, unpack);
    return mlir::success();
  }
};

// It updates init_tile by attaching innerBlock attribute to the result
// tile. The block size is choosed based on requests from its users.
struct InitTileOpPattern
    : public OpConversionPatternWithAnalysis<xetile::InitTileOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::InitTileOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = op.getType();
    auto shape = tileTy.getShape();
    if (tileTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "Skipped InitTileOp because the result tile is not rank 2.\n");

    auto innerBlockAttr = tileTy.getInnerBlocks();

    // skip it if innerBlocks has been set by user or compiler.
    if (innerBlockAttr)
      return mlir::failure();

    auto blockSize = analysis.getDefBlockSize(op.getTile());
    if (!blockSize)
      return rewriter.notifyMatchFailure(op, "Invalid block size.");

    innerBlockAttr =
        mlir::DenseI64ArrayAttr::get(getContext(), blockSize.asArrayRef());

    if (innerBlockAttr.empty())
      return rewriter.notifyMatchFailure(op, "Invalid inner block sizes ");

    auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), tileTy.getSgMap(), tileTy.getWgMap(),
        tileTy.getOrder(), innerBlockAttr, tileTy.getMemorySpace());

    auto elemTy = tileTy.getElementType();
    auto newTileTy = imex::xetile::TileType::get(shape, elemTy, attr);

    auto newOp = rewriter.create<xetile::InitTileOp>(
        op.getLoc(), mlir::TypeRange({newTileTy}), op->getOperands(),
        op->getAttrs());

    rewriter.replaceOp(op, newOp);

    return mlir::success();
  }
};

// It updates tile operand of prefetch_tile.
struct PrefetchTileOpPattern
    : public OpConversionPatternWithAnalysis<xetile::PrefetchTileOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::PrefetchTileOp,
      BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tile = adaptor.getTile();
    auto tileTy = mlir::cast<xetile::TileType>(tile.getType());
    auto blockSize = tileTy.getInnerBlocks();
    // define op is not updated yet.
    if (!blockSize)
      return failure();

    rewriter.startOpModification(op);
    op->setOperand(0, tile);
    rewriter.finalizeOpModification(op);

    return mlir::success();
  }
};

// It updates load_tile to reveal effects of innerblock attribute by
// representing value as 4D vector. An unpack op is added at the end
// to make this change to be transparent to its users.
struct LoadTileOpPattern
    : public OpConversionPatternWithAnalysis<xetile::LoadTileOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::LoadTileOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto source = adaptor.getSource();
    auto tileTy = mlir::cast<xetile::TileType>(source.getType());
    auto blockSize = tileTy.getInnerBlocks();
    auto rank = op.getValue().getType().getRank();

    if (!blockSize || rank == 4)
      return rewriter.notifyMatchFailure(
          op, "Input is not updated or the op has been updated.\n");

    auto shape = tileTy.getShape();
    auto vecTy = ::mlir::VectorType::get({shape[0] / blockSize[0],
                                          shape[1] / blockSize[1], blockSize[0],
                                          blockSize[1]},
                                         tileTy.getElementType());
    mlir::Value newOp = rewriter.create<xetile::LoadTileOp>(
        op.getLoc(), vecTy, adaptor.getSource(),
        op.getPadding().value_or(mlir::Attribute()));
    newOp = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// It updates store_tile to reveal effects of innerblock attribute.
// It uses pack op to align the shape of its vector value to the tile shape.
struct StoreTileOpPattern
    : public OpConversionPatternWithAnalysis<xetile::StoreTileOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::StoreTileOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto value = adaptor.getValue();
    auto valTy = mlir::dyn_cast<mlir::VectorType>(value.getType());
    auto tile = adaptor.getTile();
    auto tileTy = mlir::cast<xetile::TileType>(tile.getType());
    auto blockSize = tileTy.getInnerBlocks();

    // its inputs has not been updated yet.
    if (blockSize && valTy.getRank() == 2) {
      value = addPackOp(value, blockSize.asArrayRef(), rewriter);
      rewriter.replaceOpWithNewOp<xetile::StoreTileOp>(op, value, tile);
      return mlir::success();
    }
    return mlir::failure();
  }
};

// It updates update_tile_offset to reveal effects of innerblock attribute
// by updating the type of it result.
struct UpdateTileOffsetOpPattern
    : public OpConversionPatternWithAnalysis<xetile::UpdateTileOffsetOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::UpdateTileOffsetOp,
      BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tile = adaptor.getTile();
    auto tileTy = mlir::cast<xetile::TileType>(tile.getType());
    auto blockSize = tileTy.getInnerBlocks();
    // define op is not updated yet.
    if (!blockSize)
      return failure();

    rewriter.replaceOpWithNewOp<xetile::UpdateTileOffsetOp>(
        op, tileTy, tile, adaptor.getOffsetX(), adaptor.getOffsetY());
    return mlir::success();
  }
};

// It updates tile_mma to reveal effects of innerblock attribute.
// Values will be reprented as 4D vectors. An unpack op is applied
// to its result to make the change transparent to its users.
struct TileMMAOpPattern
    : public OpConversionPatternWithAnalysis<xetile::TileMMAOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::TileMMAOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto resultTy = op.getResult().getType();
    if (resultTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "The result of tile_mma must be 2D vector.\n");

    auto a = adaptor.getA();
    auto b = adaptor.getB();
    auto c = adaptor.getC();

    assert(a && b && "a operand or b operand is (are) missing.\n");

    auto getBlockingSize = [&](mlir::Value val,
                               int pos) -> mlir::FailureOr<Block> {
      auto blk = analysis.getUseBlockSize(val, UsePoint(op, pos));
      if (!blk)
        return rewriter.notifyMatchFailure(op, "Invalid block size.");
      return blk;
    };

    auto aBlockSize = getBlockingSize(op.getA(), 0);
    auto bBlockSize = getBlockingSize(op.getB(), 1);
    if (mlir::failed(aBlockSize) || mlir::failed(bBlockSize))
      return mlir::failure();
    if (c) {
      auto cBlockSize = getBlockingSize(op.getC(), 2);
      if (mlir::failed(cBlockSize))
        return mlir::failure();
      c = addPackOp(c, cBlockSize->asArrayRef(), rewriter);
    }

    a = addPackOp(a, aBlockSize->asArrayRef(), rewriter);
    b = addPackOp(b, bBlockSize->asArrayRef(), rewriter);

    assert(
        mlir::dyn_cast<mlir::VectorType>(a.getType()).getRank() == 4 &&
        mlir::dyn_cast<mlir::VectorType>(b.getType()).getRank() == 4 &&
        (!c || mlir::dyn_cast<mlir::VectorType>(c.getType()).getRank() == 4) &&
        "a, b and c (if has) should be transformed into 4D vectors.\n");

    Block dBlockSize(aBlockSize->asArrayRef()[0], bBlockSize->asArrayRef()[1]);
    auto shape = resultTy.getShape();
    auto vecTy = ::mlir::VectorType::get({shape[0] / dBlockSize[0],
                                          shape[1] / dBlockSize[1],
                                          dBlockSize[0], dBlockSize[1]},
                                         resultTy.getElementType());

    mlir::Value newOp = rewriter.create<imex::xetile::TileMMAOp>(
        op.getLoc(), vecTy, a, b, c, nullptr, nullptr, nullptr);
    newOp = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct TileReductionOpPattern
    : public OpConversionPatternWithAnalysis<xetile::ReductionOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::ReductionOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  mlir::LogicalResult
  matchAndRewrite(xetile::ReductionOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto source = op.getSource();
    auto srcTy = source.getType();
    auto reductionDims = op.getReductionDims();

    if (srcTy.getRank() != 2 || reductionDims.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "source type is not 2D vector or reduction dims are not 1");

    auto blkSize = analysis.getUseBlockSize(source, UsePoint(op, 0));
    if (!blkSize)
      return rewriter.notifyMatchFailure(op, "Invalid block size.");

    // reduction on one dim becomes reduction on two dims after blocking.
    // For example:
    // reduce<add>, %e [1]: vector<16x32xf16> to vector<16x1xf16>
    // will be transformed to
    // reduce<add>, %e [1, 3]: vector<16x2x1x16xf16> to
    // vector<16x1x1x1xf16>
    auto dim = reductionDims[0];

    auto ctx = op.getContext();
    auto shape = srcTy.getShape();
    auto newReductionDims = mlir::DenseI64ArrayAttr::get(ctx, {dim, dim + 2});
    llvm::SmallVector<int64_t> resultShape(
        {shape[0] / blkSize[0], shape[1] / blkSize[1], blkSize[0], blkSize[1]});
    for (auto dim : newReductionDims.asArrayRef())
      resultShape[dim] = 1;

    auto elemTy = srcTy.getElementType();
    auto resultType = mlir::VectorType::get(resultShape, elemTy);

    auto newSource = addPackOp(source, {blkSize[0], blkSize[1]}, rewriter);
    mlir::Value newOp = rewriter.create<xetile::ReductionOp>(
        op.getLoc(), resultType, op.getKind(), newSource, newReductionDims);
    newOp = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct TileBroadcastOpPattern
    : public OpConversionPatternWithAnalysis<xetile::BroadcastOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      xetile::BroadcastOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  mlir::LogicalResult
  matchAndRewrite(xetile::BroadcastOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSource();
    auto srcTy = src.getType();
    auto elemTy = srcTy.getElementType();
    auto broadcastDims = op.getBroadcastDim();

    if (srcTy.getRank() != 2 || broadcastDims.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "source type is not 2D vector or rank of broadcastDims is not 1");

    auto srcBlkSize = analysis.getUseBlockSize(src, UsePoint(op, 0));
    auto resBlkSize = analysis.getDefBlockSize(op.getResult());
    if (!srcBlkSize || !resBlkSize)
      return rewriter.notifyMatchFailure(op, "Invalid block size.");

    auto outShape = op.getResult().getType().getShape();

    // TODO: move this into analysis
    llvm::SmallVector<int64_t> resultShape(
        {outShape[0], outShape[1] / resBlkSize[1], 1, resBlkSize[1]});

    auto newSource = addPackOp(adaptor.getSource(),
                               {srcBlkSize[0], srcBlkSize[1]}, rewriter);

    auto resultType = mlir::VectorType::get(resultShape, elemTy);

    // broadcast on one dim becomes broadcast on two dims after blocking.
    // For example:
    // broadcast %a [0]: vector<1x32xf16> to vector<16x32xf16>
    // will be transformed to
    // broadcast %a [0, 2]: vector<1x2x1x16xf16> to vector<16x2x1x16xf16>
    auto dim = broadcastDims[0];
    auto newBroadcastDims =
        mlir::DenseI64ArrayAttr::get(op.getContext(), {dim, dim + 2});
    mlir::Value newOp = rewriter.create<xetile::BroadcastOp>(
        loc, resultType, newSource, newBroadcastDims);
    newOp = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct TileTransposeOpPattern
    : public OpConversionPatternWithAnalysis<xetile::TransposeOp,
                                             BlockingAnalysis> {
  using OpConversionPatternWithAnalysis<
      xetile::TransposeOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  mlir::LogicalResult
  matchAndRewrite(xetile::TransposeOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto input = op.getVector();
    auto inputTy = input.getType();
    auto result = op.getResult();
    auto resultTy = result.getType();
    if (resultTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "type is not 2D vector");

    auto permutation = op.getPermutation();
    if (permutation != mlir::ArrayRef<int64_t>({1, 0}))
      return rewriter.notifyMatchFailure(op, "Unsupported permutation");

    UsePoint p(op, 0);
    auto inBlockSize = analysis.getUseBlockSize(input, p);
    auto outBlockSize = analysis.getDefBlockSize(result);
    if (!inBlockSize || !outBlockSize)
      return rewriter.notifyMatchFailure(op, "Invalid block size.");

    auto srcShape = inputTy.getShape();
    auto newSrcTy = mlir::VectorType::get({srcShape[0] / inBlockSize[0],
                                           srcShape[1] / inBlockSize[1],
                                           inBlockSize[0], inBlockSize[1]},
                                          inputTy.getElementType());
    auto resShape = resultTy.getShape();
    auto newDstTy = mlir::VectorType::get({resShape[0] / outBlockSize[0],
                                           resShape[1] / outBlockSize[1],
                                           outBlockSize[0], outBlockSize[1]},
                                          resultTy.getElementType());

    mlir::Value src = adaptor.getVector();

    auto ctxt = op.getContext();
    auto blockAttr =
        mlir::DenseI64ArrayAttr::get(ctxt, inBlockSize.asArrayRef());
    Location loc = op->getLoc();
    mlir::Value pack =
        rewriter.create<xetile::TilePackOp>(loc, newSrcTy, src, blockAttr);

    int64_t newPermutation[4] = {1, 0, 3, 2};
    mlir::Value transpose = rewriter.create<xetile::TransposeOp>(
        loc, newDstTy, pack, newPermutation);

    blockAttr = mlir::DenseI64ArrayAttr::get(ctxt, outBlockSize.asArrayRef());
    mlir::Value unpack = rewriter.create<xetile::TileUnpackOp>(
        loc, resultTy, transpose, blockAttr);

    rewriter.replaceOp(op, unpack);

    return mlir::success();
  }
};

struct VectorizableOpPattern
    : public OpTraitConversionPatternWithAnalysis<mlir::OpTrait::Vectorizable,
                                                  BlockingAnalysis> {

  using OpTraitConversionPatternWithAnalysis::
      OpTraitConversionPatternWithAnalysis;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "op must have 1 result");

    auto res = op->getResult(0);
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    if (!resType || resType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "type is not 2D vector");

    auto blockSize =
        analysis.getUseBlockSize(op->getOperand(0), UsePoint(op, 0));
    if (!blockSize)
      return rewriter.notifyMatchFailure(op, "Invalid block size.");

    auto shape = resType.getShape();
    auto elemTy = resType.getElementType();
    auto blockSizeAttr =
        mlir::DenseI64ArrayAttr::get(getContext(), blockSize.asArrayRef());
    int64_t packShape[] = {shape[0] / blockSize[0], shape[1] / blockSize[1],
                           blockSize[0], blockSize[1]};

    auto newTy = mlir::VectorType::get(packShape, elemTy);

    Location loc = op->getLoc();
    mlir::OpBuilder::InsertionGuard g(rewriter);
    llvm::SmallVector<mlir::Value> newOperands;
    for (auto &&[i, arg] : llvm::enumerate(operands)) {
      auto argTy = mlir::dyn_cast<mlir::VectorType>(arg.getType());
      if (!argTy || argTy.getRank() != 2) {
        newOperands.push_back(arg);
        continue;
      }
      mlir::Value packOp = addPackOp(arg, blockSize.asArrayRef(), rewriter);
      newOperands.push_back(packOp);
    }

    mlir::OperationState opState(loc, op->getName(), newOperands,
                                 mlir::TypeRange(newTy), op->getAttrs(),
                                 op->getSuccessors());

    auto newOp = rewriter.create(opState);
    auto unpack = rewriter.create<xetile::TileUnpackOp>(
        loc, resType, newOp->getResult(0), blockSizeAttr);
    rewriter.replaceOp(op, unpack);
    return mlir::success();
  }
};

// It rewrites the SCF forOp, it mainly updates the arguments of its
// region block. unpack ops are added for VectorType operands if needed.
struct SCFForOpPattern
    : public OpConversionPatternWithAnalysis<mlir::scf::ForOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      mlir::scf::ForOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    // we don't need to update the forOp if it has no region
    // iter args, or the region iter args type are not changed.
    bool changed = false;
    for (auto [arg1, arg2] :
         llvm::zip_equal(op.getInitArgs(), adaptor.getInitArgs())) {
      changed |= (arg1 != arg2);
    }
    if (!changed)
      return mlir::failure();

    // add packOp for vector type operands if needed.
    llvm::SmallVector<mlir::Value> newInitArgs;
    for (auto [arg1, arg2] :
         llvm::zip_equal(op.getRegionIterArgs(), adaptor.getInitArgs())) {
      if (mlir::isa<mlir::VectorType>(arg1.getType())) {
        auto block = analysis.getDefBlockSize(arg1);
        if (!block) { // block could be null if the arg is not used in the loop
          auto argNo = arg1.getArgNumber() - 1;
          auto yieldOp = op.getBody()->getTerminator();
          auto opr = yieldOp->getOperand(argNo);
          block = analysis.getDefBlockSize(opr);
        }
        auto pack = addPackOp(arg2, block.asArrayRef(), rewriter);
        newInitArgs.push_back(pack);
      } else {
        newInitArgs.push_back(arg2);
      }
    }

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), newInitArgs);

    mlir::Block *newBlock = newOp.getBody();
    // remove the terminator of the new block
    if (newBlock->mightHaveTerminator())
      rewriter.eraseOp(newBlock->getTerminator());

    auto savedIP = rewriter.saveInsertionPoint();
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(newBlock);

    mlir::Block *block = op.getBody();

    // contruct the inputs for the new scf::for block.
    // An unpackOp is inserted for the corresponding init arg
    // of the new block if its init value is updated with a pack op.
    llvm::SmallVector<mlir::Value> newArguments;
    for (auto [arg1, arg2] :
         llvm::zip_equal(block->getArguments(), newBlock->getArguments())) {
      auto block = analysis.getDefBlockSize(arg1);
      if (mlir::isa<mlir::VectorType>(arg1.getType()) && block) {
        auto unpack = addUnpackOp(arg2, rewriter);
        newArguments.push_back(unpack);
      } else {
        newArguments.push_back(arg2);
      }
    }

    rewriter.restoreInsertionPoint(savedIP);
    rewriter.mergeBlocks(block, newBlock, newArguments);

    llvm::SmallVector<mlir::Value> newValues;
    for (auto [i, result] : llvm::enumerate(newOp->getResults())) {
      // if corresponding init arg is updated with a pack op
      // an unpack op is needed for the result to make it
      // transparent to its users.
      if (newInitArgs[i].getDefiningOp<xetile::TilePackOp>()) {
        auto unpack = addUnpackOp(result, rewriter);
        newValues.push_back(unpack);
      } else {
        newValues.push_back(result);
      }
    }
    rewriter.replaceOp(op, newValues);
    return mlir::success();
  }
};

// It serves to insert pack ops for approriate vales if needed.
// for example, tile_mma result is vector<32x32xf16> (after unpack),
// but its corresponding argument in forOp is with type vector<1x2x32x16xf16>
// This op pattern will insert a pack op to make it consistent with the
// corresponding argument type.
struct SCFYieldOpPattern
    : public OpConversionPatternWithAnalysis<mlir::scf::YieldOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      mlir::scf::YieldOp, BlockingAnalysis>::OpConversionPatternWithAnalysis;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {

    llvm::SmallVector<mlir::Value> newResults;
    for (auto [arg1, arg2] :
         llvm::zip_equal(op.getResults(), adaptor.getResults())) {
      auto block = analysis.getDefBlockSize(arg1);
      if (mlir::isa<mlir::VectorType>(arg1.getType()) && block) {
        auto pack = addPackOp(arg2, block.asArrayRef(), rewriter);
        newResults.push_back(pack);
      } else {
        newResults.push_back(arg2);
      }
    }

    auto newOp = rewriter.create<mlir::scf::YieldOp>(op.getLoc(), newResults);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// Blocks a vector.create_mask op such that, ideally, it matches its consuming
// select op (which is an elementwise op). In this way, during the lowering to
// XeGPU, there will be a one-to-one correspondence and an
// unrealized_conversion_cast will not be needed.
struct VectorCreateMaskOpPattern
    : public OpConversionPatternWithAnalysis<mlir::vector::CreateMaskOp,
                                             BlockingAnalysis> {

  using OpConversionPatternWithAnalysis<
      mlir::vector::CreateMaskOp,
      BlockingAnalysis>::OpConversionPatternWithAnalysis;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::CreateMaskOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto res = op.getResult();
    auto resType = res.getType();
    if (resType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "type is not 2D vector");

    // Only two cases are supported for now:
    // 1.The first operand is a constant equal to the first dimension of the
    // output shape (i.e., all rows are enabled). In other words, masking
    // columns within a row is supported.
    // 2.The second operand is a constant equal to the second dimension of the
    // output shape (i.e., all columns are enabled).
    auto shape = resType.getShape();
    APInt cstOp0, cstOp1;
    if (!(matchPattern(op->getOperand(0), m_ConstantInt(&cstOp0)) &&
          cstOp0.getSExtValue() == shape[0]) &&
        !(matchPattern(op->getOperand(1), m_ConstantInt(&cstOp1)) &&
          cstOp1.getSExtValue() == shape[1])) {
      op->emitOpError() << "Unsupported operands";
      return mlir::failure();
    }

    auto block = analysis.getDefBlockSize(res);
    if (!block)
      return rewriter.notifyMatchFailure(op, "Invalid block size.");

    // TODO: support blocking the outer dimension.
    if (block[0] != 1) {
      op->emitOpError() << "Unsupported inner block sizes";
      return mlir::failure();
    }

    auto newTy = mlir::VectorType::get(
        {shape[0] / block[0], shape[1] / block[1], block[0], block[1]},
        resType.getElementType());

    // Due to the simplifications mentioned above, for now, the index operands
    // are not adjusted. In fact, only the first index operand (masked rows) or
    // the second index operand (masked columns) will be used during the
    // lowering to XeGPU.
    auto operands = op.getOperands();
    auto newOp = rewriter.create<mlir::vector::CreateMaskOp>(
        op.getLoc(), newTy,
        ValueRange({operands[0], operands[1], operands[0], operands[1]}),
        op->getAttrs());
    auto unpack = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, unpack);
    return mlir::success();
  }
};

} // namespace Blocking

void populateXeTileBlockingPatterns(mlir::RewritePatternSet &patterns,
                                    BlockingAnalysis &analysis) {
  patterns.insert<
      Blocking::ArithConstantOpPattern, Blocking::InitTileOpPattern,
      Blocking::PrefetchTileOpPattern, Blocking::LoadTileOpPattern,
      Blocking::StoreTileOpPattern, Blocking::UpdateTileOffsetOpPattern,
      Blocking::TileMMAOpPattern, Blocking::TileReductionOpPattern,
      Blocking::TileBroadcastOpPattern, Blocking::TileTransposeOpPattern,
      Blocking::VectorizableOpPattern, Blocking::SCFForOpPattern,
      Blocking::SCFYieldOpPattern, Blocking::VectorCreateMaskOpPattern>(
      patterns.getContext(), analysis);
}

// Lowers XeTile to blocked layout with high-dim vector
class XeTileBlockingPass : public impl::XeTileBlockingBase<XeTileBlockingPass> {

public:
  XeTileBlockingPass() {
    uArchInterface = std::make_shared<imex::XePVCuArch>();
  }

  XeTileBlockingPass(const std::string &deviceName) {
    if (deviceName == "pvc") {
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    }
  }

  mlir::LogicalResult
  initializeOptions(mlir::StringRef options,
                    mlir::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler)))
      return mlir::failure();
    if (device == "pvc")
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    else
      return errorHandler(llvm::Twine("Invalid device: ") + device);
    return mlir::success();
  }

  void runOnOperation() override {
    auto mod = this->getOperation();
    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    if (!uArchInterface) {
      mod.emitOpError("Can not get GPU Arch Definition for given Arch param");
      return signalPassFailure();
    }

    BlockingAnalysis analysis(uArchInterface);
    if (mlir::failed(analysis.run(mod)))
      return signalPassFailure();

    // analysis.printAnalysisResult();

    mlir::MLIRContext &context = getContext();

    mlir::RewritePatternSet patterns(&context);
    populateXeTileBlockingPatterns(patterns, analysis);

    mlir::ConversionTarget target(context);
    target.addLegalOp<xetile::TilePackOp>();
    target.addLegalOp<xetile::TileUnpackOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<mlir::vector::ShapeCastOp>();
    target.addLegalOp<mlir::vector::StoreOp>();
    target.addLegalOp<mlir::vector::LoadOp>();

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp>(
        [&](mlir::arith::ConstantOp op) -> bool {
          auto vecTy = mlir::dyn_cast<mlir::VectorType>(op.getType());
          return (!vecTy || vecTy.getRank() != 2);
        });

    target.addDynamicallyLegalOp<xetile::InitTileOp>(
        [&](xetile::InitTileOp op) -> bool {
          return (op && op.getTile().getType().getInnerBlocks());
        });

    target.addDynamicallyLegalOp<xetile::PrefetchTileOp>(
        [&](xetile::PrefetchTileOp op) -> bool {
          return (op && op.getTile().getType().getInnerBlocks());
        });

    target.addDynamicallyLegalOp<xetile::UpdateTileOffsetOp>(
        [&](xetile::UpdateTileOffsetOp op) -> bool {
          return (op && op.getTile().getType().getInnerBlocks());
        });

    target.addDynamicallyLegalOp<xetile::LoadTileOp>(
        [&](xetile::LoadTileOp op) -> bool {
          return (op && op.getValue().getType().getRank() == 4);
        });

    target.addDynamicallyLegalOp<xetile::StoreTileOp>(
        [&](xetile::StoreTileOp op) -> bool {
          return (op && op.getValue().getType().getRank() == 4);
        });

    target.addDynamicallyLegalOp<xetile::TileMMAOp>(
        [&](xetile::TileMMAOp op) -> bool {
          return (op && op.getOutput().getType().getRank() == 4);
        });

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation *op) {
      bool result = true;
      for (auto ty : op->getOperandTypes()) {
        if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(ty))
          result &= (vecTy.getRank() != 2);
        if (auto tileTy = mlir::dyn_cast<xetile::TileType>(ty))
          result &= bool(tileTy.getInnerBlocks());
      }
      for (auto ty : op->getResultTypes()) {
        if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(ty))
          result &= (vecTy.getRank() != 2);
        if (auto tileTy = mlir::dyn_cast<xetile::TileType>(ty))
          result &= bool(tileTy.getInnerBlocks());
      }
      return result;
    });

    auto status = applyPartialConversion(mod, target, std::move(patterns));
    if (failed(status)) {
      return signalPassFailure();
    }
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

/// Create a pass
std::unique_ptr<::mlir::Pass>
createXeTileBlockingPass(const std::string &deviceName) {
  return std::make_unique<XeTileBlockingPass>(deviceName);
}
} // namespace imex
