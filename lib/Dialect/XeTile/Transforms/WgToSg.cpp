//===-------------- WgToSg.cpp --------- XeTileWgToSgPass Pass  -------*- C++
//-*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains pass for trasforming WG level XeTile to SG level XeTile
/// based on decomposition attributes attached to the IR.
///
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/OneToNTypeConversion.h"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
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

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <cassert>

#include <imex/Dialect/XeTile/Transforms/Passes.h>

using namespace mlir;
using namespace imex;
namespace imex {
#define GEN_PASS_DECL_XETILEWGTOSG
#define GEN_PASS_DEF_XETILEWGTOSG
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace {
// Create a Map to store SG layout_order if we have a load
// which is transposed before being passed to MMA.
// Sg layout_order [0, 1] means the subgroup ids are arranged
// in column major. Default is row-major [1, 0].
// For example:
// If we have a sgLayout [4, 8] with layout_order [0, 1]
// the sg id's will be arranged in the following manner
// | 0  | 4 | 8  | 12 | 16 | 20 | 24 | 28 |
// | 1  | 5 | 9  | 13 | 17 | 21 | 25 | 29 |
// | 2  | 6 | 10 | 14 | 18 | 22 | 26 | 30 |
// | 3  | 7 | 11 | 15 | 19 | 23 | 27 | 31 |

// Internally we use this layout_order information to calculate the
// offset for init and load tile

llvm::DenseMap<mlir::Value, std::array<int, 2>> opSgLayoutMap;
} // namespace

namespace imex {

// This pass transform the Ops at WG level to SG level using the
// decomposition attributes provided by wg_map.
// clang-format off
// Example (using init_tile):
// #wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [64, 64]>
// #tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c>
//  %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32>
//  -> !xetile.tile<256x256xf32, #tile_attr_c>
//  becomes
// %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32>
//    -> !xetile.tile<64x64xf32>


class WGToSGInitTileOpPattern : public OpConversionPattern<xetile::InitTileOp> {
  using OpConversionPattern<xetile::InitTileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tileTy = op.getType();

    auto order = tileTy.getOrder();

    // TODO: Add support for order
    if (order[0] == 0 && order[1] == 1)
      return mlir::failure();

    auto elemTy = tileTy.getElementType();

    auto wgTileShape = tileTy.getShape();
    auto sgTileShape = tileTy.getWgMap().getSgData();
    auto sgLayout = tileTy.getWgMap().getSgLayout();

    auto newTileTy =
        imex::xetile::TileType::get({sgTileShape[0], sgTileShape[1]}, elemTy);

    auto createIndexConstant = [&](mlir::Type type, int64_t value) {
      auto attr = rewriter.getIndexAttr(value);
      return rewriter.create<mlir::arith::ConstantOp>(loc, type, attr);
    };

    rewriter.setInsertionPoint(op);
    // get the subgroup Id
    auto sgID = rewriter.create<mlir::gpu::SubgroupIdOp>(loc, nullptr);
    auto indexType = rewriter.getIndexType();
    auto sgLayoutDimYConst = createIndexConstant(indexType, sgLayout[1]);
    auto sgDataDimYConst = createIndexConstant(indexType, sgTileShape[0]);
    auto sgDataDimXConst = createIndexConstant(indexType, sgTileShape[1]);

    // The sgID is a linear (1D) id. Convert it to 2D to get the x and y
    // coordinates of sg
    // row = i / cols
    // col =  i % cols
    auto sgIdY =
        rewriter.create<mlir::index::DivUOp>(loc, sgID, sgLayoutDimYConst);
    auto sgIdX =
        rewriter.create<mlir::index::RemUOp>(loc, sgID, sgLayoutDimYConst);

    llvm::SmallVector<mlir::Value> offsets;
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = op.getOffsets();
    for (size_t i = 0, j = 0; i != staticOffsets.size(); i++) {
      if (mlir::ShapedType::isDynamic(staticOffsets[i])) {
        offsets.push_back(dynamicOffsets[j++]);
      } else {
        offsets.push_back(rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(staticOffsets[i])));
      }
    }

    mlir::Value source = op.getSource();
    mlir::SmallVector<mlir::OpFoldResult> globalOffsetsX; // cols
    mlir::SmallVector<mlir::OpFoldResult> globalOffsetsY; // rows
    mlir::SmallVector<mlir::SmallVector<mlir::OpFoldResult>> offsetPermutations;

    // Calculate the global offsets for tiles using the sgData and sgLayout
    // configuration given in wg_map. If each SG works on one output tile, each
    // WG level op will be transformed to same op with SG shapes/sizes. If each
    // SG needs to process more than one output tile, the WG level op will be
    // decomposed to multiple ops with SG level shapes/sizes
    auto calculateGlobalOffsets =
        [&](mlir::SmallVector<mlir::OpFoldResult> &globalOffsets,
            int wgTileShape, int sgTileShape, int sgLayout,
            mlir::Value sgDataDimConst, mlir::Value sgId, mlir::Value offset) {
          for (int i = 0; i < wgTileShape / sgTileShape; i += sgLayout) {
            auto constI = createIndexConstant(indexType, i);
            auto off =
                rewriter.createOrFold<mlir::index::AddOp>(loc, constI, sgId);
            auto mod = rewriter.createOrFold<mlir::index::RemUOp>(
                loc, off,
                createIndexConstant(indexType, wgTileShape / sgTileShape));
            auto localOffset = rewriter.createOrFold<mlir::index::MulOp>(
                loc, mod, sgDataDimConst);
            auto globalOffset = rewriter.createOrFold<mlir::index::AddOp>(
                loc, offset, localOffset);
            globalOffsets.push_back(globalOffset);
          }
        };

    // Look up the map if the init_tile has a layout_order [0, 1]
    // If it does, tranpose the sg ids to get the correct tile.
    auto it = opSgLayoutMap.find(op.getResult());
    if (it != opSgLayoutMap.end()){
     assert((opSgLayoutMap[op->getResult(0)] == std::array<int, 2>{0, 1}));
     calculateGlobalOffsets(globalOffsetsY, wgTileShape[0], sgTileShape[0],
                           sgLayout[0], sgDataDimYConst, sgIdX, offsets[offsets.size() - 2]);
     calculateGlobalOffsets(globalOffsetsX, wgTileShape[1], sgTileShape[1],
                           sgLayout[1], sgDataDimXConst, sgIdY, offsets[offsets.size() - 1]);
    }
    else {
    calculateGlobalOffsets(globalOffsetsY, wgTileShape[0], sgTileShape[0],
                           sgLayout[0], sgDataDimYConst, sgIdY, offsets[offsets.size() - 2]);
    calculateGlobalOffsets(globalOffsetsX, wgTileShape[1], sgTileShape[1],
                           sgLayout[1], sgDataDimXConst, sgIdX, offsets[offsets.size() - 1]);
    }
    // TODO: check for how to broadcast
    for (auto y : globalOffsetsY) {
      for (auto x : globalOffsetsX) {
        offsetPermutations.push_back({y, x});
      }
    }

    mlir::SmallVector<mlir::Value> newInitTileOps;
    llvm::SmallVector<mlir::OpFoldResult> newOffsets;
    for (size_t j = 0; j < offsets.size() - 2; ++j) {
        newOffsets.push_back(offsets[j]);
    }
    for (size_t i = 0; i < offsetPermutations.size(); i++) {
      newOffsets.push_back(offsetPermutations[i][0]);
      newOffsets.push_back(offsetPermutations[i][1]);
      auto newInitTileOp = rewriter.create<xetile::InitTileOp>(
          loc, newTileTy, source, newOffsets);
      newOffsets.clear();
      newInitTileOps.push_back(newInitTileOp);
    }

    rewriter.replaceOpWithMultiple(op, {newInitTileOps});
    return mlir::success();
  }
};

class WGToSGLoadTileOpPattern : public OpConversionPattern<xetile::LoadTileOp> {
  using OpConversionPattern<xetile::LoadTileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto sources = adaptor.getSource();
    auto res = op.getValue();

    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());

    if (!resType || resType.getRank() != 2)
      return mlir::failure();

    llvm::SmallVector<::mlir::Value> newLoadOps;
    llvm::SmallVector<mlir::Type> newResultTypes;
    for (auto src : sources) {
      auto tileTy = llvm::dyn_cast<xetile::TileType>(src.getType());
      auto newResTy =
          mlir::VectorType::get({tileTy.getShape()[0], tileTy.getShape()[1]},
                                tileTy.getElementType());
      auto newLoadOp = rewriter.create<xetile::LoadTileOp>(
          op.getLoc(), newResTy, src, op.getPaddingAttr(), op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr());
      newLoadOps.push_back(newLoadOp);
      newResultTypes.push_back(newLoadOp.getResult().getType());
    }
    rewriter.replaceOpWithMultiple(op, {newLoadOps});
    return mlir::success();
  }
};

class WGToSGTileMMAOpPattern : public OpConversionPattern<xetile::TileMMAOp> {
  using OpConversionPattern<xetile::TileMMAOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = op.getResult().getType();

    if (resultTy.getRank() != 2)
      return mlir::failure();

    llvm::SmallVector<::mlir::Value> newTileMMAOps;
    llvm::SmallVector<mlir::Type> newResultTypes;
    size_t i = 0;
    for (auto a : adaptor.getA()) {
      for (auto b : adaptor.getB()) {

        mlir::Value tmpC;
        if (op.getC())
          tmpC = adaptor.getC()[i++];

        auto aShape = llvm::cast<mlir::VectorType>(a.getType()).getShape();
        auto bShape = llvm::cast<mlir::VectorType>(b.getType()).getShape();
        auto resTy = mlir::VectorType::get({aShape[0], bShape[1]},
                                           resultTy.getElementType());
        tmpC = rewriter.create<xetile::TileMMAOp>(
            op.getLoc(), resTy, a, b, tmpC, nullptr, nullptr, nullptr);
        newTileMMAOps.push_back(tmpC);
        newResultTypes.push_back(resTy);
      }
    }

    rewriter.replaceOpWithMultiple(op, {newTileMMAOps});
    return mlir::success();
  }
};

class WGToSGStoreTileOpPattern : public OpConversionPattern<xetile::StoreTileOp> {
  using OpConversionPattern<xetile::StoreTileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newValues = adaptor.getValue();
    auto newDstTiles = adaptor.getTile();

    for (size_t i = 0; i < newValues.size(); i++) {
      rewriter.create<xetile::StoreTileOp>(op.getLoc(), newValues[i],
                                           newDstTiles[i], op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr());
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class WGToSGSCFForOpPattern : public OpConversionPattern<mlir::scf::ForOp> {
  using OpConversionPattern<mlir::scf::ForOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Collect the sizes of the new argument mapping. This is needed for mapping
    // ForOp results.
    SmallVector<size_t> remappedArgSizes;
    llvm::ArrayRef<ValueRange> remappedInitArgs = adaptor.getInitArgs();
    SmallVector<Value> flattenedRemappedInitArgs;
    for (auto initArg : remappedInitArgs) {
      remappedArgSizes.push_back(initArg.size());
      flattenedRemappedInitArgs.append(initArg.begin(), initArg.end());
    }

    // Do a signature conversion for the old for body.
    auto oldBody = op.getBody();
    auto oldBodyArgTypes = oldBody->getArgumentTypes();
    TypeConverter::SignatureConversion signatureConversion(
        oldBodyArgTypes.size());
    signatureConversion.addInputs(0, oldBodyArgTypes[0]);
    for (unsigned i = 1; i < oldBodyArgTypes.size(); i++) {
      auto remappedTypes = llvm::to_vector(remappedInitArgs[i - 1].getTypes());
      signatureConversion.addInputs(i, remappedTypes);
    }
    rewriter.applySignatureConversion(oldBody, signatureConversion);
    // Create a new ForOp.
    auto newForOp = rewriter.create<scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        flattenedRemappedInitArgs);
    rewriter.eraseBlock(newForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().begin());

    // Compute the remapped results.
    SmallVector<ValueRange> remappedResults;
    unsigned newResultOffset = 0;
    for (unsigned i = 0; i < remappedArgSizes.size(); i++) {
      unsigned remappedResultSize = remappedArgSizes[i];
      ValueRange remappedResultValues =
          newForOp.getResults().slice(newResultOffset, remappedResultSize);
      remappedResults.push_back(remappedResultValues);
      newResultOffset += remappedResultSize;
    }

    rewriter.replaceOpWithMultiple(op, remappedResults);
    return success();
  }
};

struct WGToSGSCFYieldOpPattern : public OpConversionPattern<mlir::scf::YieldOp> {
  using OpConversionPattern<mlir::scf::YieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OneToNOpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedYields = adaptor.getOperands();
    SmallVector<Value> newYieldedValues;
    for (auto yield : remappedYields)
      newYieldedValues.append(yield.begin(), yield.end());

    rewriter.modifyOpInPlace(op, [&]() {
      op.getResultsMutable().clear();
      op.getResultsMutable().append(newYieldedValues);
    });
    return success();
  }
};

class WGToSGUpdateTileOffsetOpPattern
    : public OpConversionPattern<xetile::UpdateTileOffsetOp> {
  using OpConversionPattern<xetile::UpdateTileOffsetOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<::mlir::Value> newUpdateTileOffsetOps;
    llvm::SmallVector<mlir::Type> newResultTypes;
    for (auto tile : adaptor.getTile()) {

      auto newUpdateTileOffsetOp = rewriter.create<xetile::UpdateTileOffsetOp>(
          op.getLoc(), tile.getType(), tile, op.getOffsetX(), op.getOffsetY(), op.getIndices());
      newUpdateTileOffsetOps.push_back(newUpdateTileOffsetOp);
      newResultTypes.push_back(tile.getType());
    }

    rewriter.replaceOpWithMultiple(op, {newUpdateTileOffsetOps});
    return mlir::success();
  }
};

class WGToSGArithConstantOpPattern
    : public OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    auto valueType = mlir::dyn_cast<mlir::VectorType>(value.getType());
    auto wgTileShape = valueType.getShape();

    if (!value)
      return mlir::failure();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();
    auto sgLayout = mapAttr.getSgLayout();
    mlir::SmallVector<int64_t> outputShape;
    // If WG tile rank is 1, set the output shape as the
    // non-unit dim of sgData
    if(wgTileShape.size() == 1) {
      if(sgData[0] == 1)
        outputShape.push_back(sgData[1]);
      else
        outputShape.push_back(sgData[0]);
    } else {
      outputShape.push_back(sgData[0]);
      outputShape.push_back(sgData[1]);
    }

    auto newTy =
        mlir::VectorType::get(outputShape, value.getElementType());

    llvm::SmallVector<mlir::Attribute> elems(
        value.value_begin<mlir::Attribute>(),
        value.value_end<mlir::Attribute>());

    llvm::SmallVector<mlir::Attribute> newValues;
    for (int64_t i = 0; i < static_cast<int64_t>(sgData[0]) * sgData[1]; i++) {
      newValues.push_back(elems[i]);
    }

    auto attr = mlir::DenseElementsAttr::get(newTy, newValues);

    size_t numOps;
    // If WG tile is 1D vector just support 1:1 mapping.
    // TODO: Support round robin for 1D
    if(wgTileShape.size() == 1) {
      if (sgLayout[0] * sgData[0] == wgTileShape[0] ||
          sgLayout[1] * sgData[1] == wgTileShape[0])
            numOps = 1;
      else
        return mlir::failure();
    } else if(sgLayout[0] * sgData[0] == wgTileShape[0] &&
              sgLayout[1] * sgData[1] == wgTileShape[1]) {
                 numOps = 1;
    } else
        numOps = (wgTileShape[0] / (sgLayout[0] * sgData[0])) +
                 (wgTileShape[1] / (sgLayout[1] * sgData[1]));

    llvm::SmallVector<::mlir::Value> newOps;
    for (size_t i = 0; i < numOps; i++) {
      auto newOp = rewriter.create<arith::ConstantOp>(op.getLoc(), newTy, attr);
      newOps.push_back(newOp);
    }

    rewriter.replaceOpWithMultiple(op, {newOps});
    return mlir::success();
  }
};

class WGToSGVectorTranspose
    :public OpConversionPattern<mlir::vector::TransposeOp> {
  using OpConversionPattern<mlir::vector::TransposeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVector().getType().getRank() != 2)
      return mlir::failure();

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return mlir::failure();
    }

    auto it = opSgLayoutMap.find(op.getResult());
    // Transpose within subgroup if the sg layout order is {0, 1}
    if (it != opSgLayoutMap.end()){
      assert((opSgLayoutMap[op->getResult(0)] == std::array<int, 2>{0, 1}));
      auto sgData = mapAttr.getSgData();
      auto newTy = mlir::VectorType::get({sgData[0], sgData[1]},
                                                   resType.getElementType());
      auto newOp = rewriter.create<mlir::vector::TransposeOp>(
              op.getLoc(), newTy, adaptor.getVector(), op.getPermutation());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    else
    {
      //TODO : Transpose using SLM
      return mlir::failure();
    }
  }
};

// This pattern transforms the convert layout op in the following manner:
// 1. Store the original vector to slm using input operand layout
// 2. Add barrier
// 3. Load the vector from slm using the result layout

// Example:
// WG IR
// #wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>
// #wg_map_a = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>
// %vector_a = xetile.tile_conv_layout %vector_b {wg_map_result = #wg_map_a, wg_map_source = #wg_map_b}: vector<256x256xfloat> into vector<256x256xfloat>

// SG IR
// %slm = memref.alloc() : memref<256x256xf32, 3>
// %tile = xetile.init_tile %slm[offset_x, offset_y] : memref<256x256xf32, 3> -> xetile.tile<32x64xf32>
// xetile.store_tile %vector_b, %tile :vector<32x64xf32>, !xetile.tile<32x64xf32>
// gpu.barrier
// %remapped_tile = xetile.init_tile %slm[offsetX, offsetY] : memref<256x256xf32, 3> -> xetile.tile<8x256xf32>
// %remapped_vector = xetile.load_tile %reshaped_tile : xetile.tile<8x256xf32> -> vector<8x256xf32>
class WGToSGXeTileConvertLayout
    :public OpConversionPattern<xetile::ConvertLayoutOp> {
  using OpConversionPattern<xetile::ConvertLayoutOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getSource().getType().getRank() != 2)
      return mlir::failure();

    auto loc = op.getLoc();
    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    auto elemTy = resType.getElementType();
    auto resShape = resType.getShape();

    auto dstMapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("wg_map_result"));

    xetile::WorkGroupMapAttr srcMapAttr;
    srcMapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("wg_map_source"));

    if (!dstMapAttr) {
      return mlir::failure();
    }

    if(!srcMapAttr) {
      // Get the map from operand
      auto operand = op.getSource().getDefiningOp();
      srcMapAttr =  llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(operand->getAttr("map"));
      if (!srcMapAttr) {
        return mlir::failure();
      }
    }

    auto srcMapSgData = srcMapAttr.getSgData();
    auto srcSgLayout = srcMapAttr.getSgLayout();
    auto dstMapSgData = dstMapAttr.getSgData();
    auto dstSgLayout = dstMapAttr.getSgLayout();

    auto createIndexConstant = [&](mlir::Type type, int64_t value) {
      auto attr = rewriter.getIndexAttr(value);
      return rewriter.create<mlir::arith::ConstantOp>(loc, type, attr);
    };

    rewriter.setInsertionPoint(op);
    // Allocate SLM
    auto bitWidth = elemTy.getIntOrFloatBitWidth();
    auto flattenFactor = bitWidth / 8;
    auto slmShape = resShape[0] * resShape[1] * flattenFactor;
    auto slmTy = mlir::MemRefType::get(slmShape, rewriter.getI8Type(), {}, 3);
    auto slm = rewriter.create<mlir::memref::AllocOp>(loc, slmTy);
    ValueRange sizes;
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto viewTy = mlir::MemRefType::get({resShape[0], resShape[1]}, elemTy, {}, 3);
    auto viewOp = rewriter.create<mlir::memref::ViewOp>(
                op.getLoc(), viewTy, slm, zero, sizes);

    // Get SG id
    auto sgId = rewriter.create<mlir::gpu::SubgroupIdOp>(
        loc, rewriter.getIndexType(), nullptr);

    auto indexType = rewriter.getIndexType();
    auto srcMapDimY = createIndexConstant(indexType, srcSgLayout[1]);

    // The sgID is a linear (1D) id. Convert it to 2D to get the x and y
    // coordinates of sg
    // row = i / cols
    // col =  i % cols
    // x is row, y is col
    // TODO: Floorsdiv and Remu are expensive. Find alterate.
    auto storeSgIdX =
        rewriter.create<mlir::index::DivUOp>(loc, sgId, srcMapDimY);
    auto storeSgIdY =
        rewriter.create<mlir::index::RemUOp>(loc, sgId, srcMapDimY);

    // Store to SLM using src map
    auto memoryScopeAttr = mlir::IntegerAttr::get(rewriter.getIntegerType(32), 3);
    auto order = mlir::DenseI32ArrayAttr::get(op.getContext(), {1, 0});
    auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), nullptr /*sgMap*/, nullptr /*wgMap*/,
        order /*order*/, memoryScopeAttr /*memoryscope*/, nullptr /*scatterAttr*/);
    xetile::TileType srcTileTy =
      imex::xetile::TileType::get({srcMapSgData[0], srcMapSgData[1]}, elemTy, attr);

    auto storeOffsetX = rewriter.createOrFold<mlir::index::MulOp>(
                loc, storeSgIdX, createIndexConstant(indexType, srcMapSgData[0]));
    auto storeOffsetY = rewriter.createOrFold<mlir::index::MulOp>(
                loc, storeSgIdY, createIndexConstant(indexType, srcMapSgData[1]));
    auto storeInitTileOp = rewriter.create<xetile::InitTileOp>(
          loc, srcTileTy, viewOp, llvm::ArrayRef<mlir::OpFoldResult>({storeOffsetX, storeOffsetY}));
    //TODO: Set up cache attributes
    rewriter.create<xetile::StoreTileOp>(loc, adaptor.getSource(),
                                         storeInitTileOp, nullptr, nullptr, nullptr);

    // Add barrier
    rewriter.create<mlir::gpu::BarrierOp>(loc);

    // Load from SLM with result map
    xetile::TileType dstTileTy =
      imex::xetile::TileType::get({dstMapSgData[0], dstMapSgData[1]}, elemTy, attr);
    auto newResTy =
          mlir::VectorType::get({dstMapSgData[0], dstMapSgData[1]}, elemTy);

    auto dstMapDimY = createIndexConstant(indexType, dstSgLayout[1]);
    auto loadSgIdX = rewriter.create<mlir::index::DivUOp>(loc, sgId, dstMapDimY);
    auto loadSgIdY =  rewriter.create<mlir::index::RemUOp>(loc, sgId, dstMapDimY);
    mlir::Value loadOffsetX = rewriter.createOrFold<mlir::index::MulOp>(
                loc, loadSgIdX, createIndexConstant(indexType, dstMapSgData[0]));
    mlir::Value loadOffsetY = rewriter.createOrFold<mlir::index::MulOp>(
                loc, loadSgIdY, createIndexConstant(indexType, dstMapSgData[1]));
    loadOffsetX = rewriter.createOrFold<mlir::index::RemUOp>(
                loc, loadOffsetX, createIndexConstant(indexType, resShape[0]));
    loadOffsetY = rewriter.createOrFold<mlir::index::RemUOp>(
                loc, loadOffsetY, createIndexConstant(indexType, resShape[1]));
    auto loadInitTileOp = rewriter.create<xetile::InitTileOp>(
          loc, dstTileTy, viewOp, llvm::ArrayRef<mlir::OpFoldResult>({loadOffsetX, loadOffsetY}));
    //TODO: Set up cache attributes
    auto loadTile = rewriter.create<xetile::LoadTileOp>(
          loc, newResTy, loadInitTileOp, mlir::Attribute(), nullptr, nullptr, nullptr);

    rewriter.replaceOp(op, loadTile);
    return mlir::success();
    }
  };

class WGToSGVectorBroadcast
    :public OpConversionPattern<mlir::vector::BroadcastOp> {
  using OpConversionPattern<mlir::vector::BroadcastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVector().getType().getRank() != 2)
      return mlir::failure();

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());

    auto srcTy =  mlir::dyn_cast<mlir::VectorType>((adaptor.getSource()).getType());
    auto srcShape = srcTy.getShape();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();
    auto newTy = mlir::VectorType::get({sgData[0], sgData[1]},
                                       resType.getElementType());
    auto dstShape = newTy.getShape();

    if (!(srcShape[0] == 1 && srcShape[1] == dstShape[1]) &&
        !(srcShape[1] == 1 && srcShape[0] == dstShape[0]))
      return mlir::failure();

    auto newOp = rewriter.create<mlir::vector::BroadcastOp>(
            op.getLoc(), newTy, adaptor.getSource());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

class WGToSGPrefetchOpPattern : public OpConversionPattern<xetile::PrefetchTileOp> {
  using OpConversionPattern<xetile::PrefetchTileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto L1 = op.getL1HintAttr();
    auto L2 = op.getL2HintAttr();
    auto L3 = op.getL3HintAttr();

    for(auto tile : adaptor.getTile()) {
      rewriter.create<xetile::PrefetchTileOp>(op.getLoc(),  tile, L1, L2, L3);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class WGToSGVectorMultiDimReductionOp
    : public OpConversionPattern<mlir::vector::MultiDimReductionOp> {
  using OpConversionPattern<mlir::vector::MultiDimReductionOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::MultiDimReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    auto resRank = resType.getShape().size();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();

    auto src = adaptor.getSource();
    auto srcType = mlir::dyn_cast<mlir::VectorType>(src.getType());

    if (resRank == 2) {
      bool newReduceDim = sgData[0] == 1 ? 0 : 1;
      mlir::SmallVector<int64_t> redDims{newReduceDim};
      auto outputShape =
          newReduceDim == 0 ? srcType.getDimSize(1) : srcType.getDimSize(0);
      auto newTy = mlir::VectorType::get(outputShape, srcType.getElementType());

      // ShapeCast acc to match reduction op shape.
      auto acc = rewriter.create<vector::ShapeCastOp>(op->getLoc(), newTy,
                                                      adaptor.getAcc());

      auto newOp = rewriter.create<mlir::vector::MultiDimReductionOp>(
          op.getLoc(), newTy, op.getKind(), src, acc, redDims);

      // Shape Cast the output of reduction back to 2D
      auto accumalator = adaptor.getAcc();
      auto accumalatorType =
          mlir::dyn_cast<mlir::VectorType>(accumalator.getType());
      auto outputVectorTy = mlir::VectorType::get(
          accumalatorType.getShape(), accumalatorType.getElementType());
      auto shapeCastOp = rewriter.create<vector::ShapeCastOp>(
          op.getLoc(), outputVectorTy, newOp);
      rewriter.replaceOp(op, shapeCastOp);
      return mlir::success();
    }
    // Regular 2D vector.multi_reduction
    else {
      auto reductionDims = op.getReductionDims();
      if (reductionDims.size() != 1)
        return mlir::failure();

      bool reduceDim = reductionDims[0];
      auto outputShape =
          reduceDim == 0 ? srcType.getDimSize(1) : srcType.getDimSize(0);

      mlir::SmallVector<int64_t> redDims{reduceDim};
      auto newTy = mlir::VectorType::get(outputShape, srcType.getElementType());
      auto newOp = rewriter.create<mlir::vector::MultiDimReductionOp>(
          op.getLoc(), newTy, op.getKind(), adaptor.getSource(),
          adaptor.getAcc(), redDims);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
  }
};

// Shape cast will support going from 1D to 2D since the vector.multi_reduction
// produces 1D

class WGToSGVectorShapeCast
    : public OpConversionPattern<mlir::vector::ShapeCastOp> {
  using OpConversionPattern<mlir::vector::ShapeCastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::ShapeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    auto resShape = resType.getShape();

    // Assumption is 3D shape cast is used for partial reduction.
    // So just replace it with the transformed source of shape_cast
    if (resShape.size() == 3) {
      for (mlir::Operation *userOp : op.getResult().getUsers()) {
        // Check if the user operation is not a vector.multi_reduction
        if (!isa<mlir::vector::MultiDimReductionOp>(userOp)) {
          return mlir::failure();
        }
      }
      rewriter.replaceOp(op, adaptor.getSource());
      return mlir::success();
    }

    // One of the dims have to be a unit dim
    if (resShape[0] != 1 && resShape[1] != 1)
      return mlir::failure();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();
    auto newTy =
        mlir::VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    auto newOp = rewriter.create<mlir::vector::ShapeCastOp>(
        op.getLoc(), newTy, adaptor.getSource());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

template <typename Op, int numOperands>
Op createOp(ConversionPatternRewriter &rewriter, mlir::Location loc,
            llvm::SmallVector<llvm::SmallVector<mlir::Value>> operands, int i) {
  static_assert(numOperands >= 1 && numOperands <= 3,
                "Unsupported number of operands");

  if constexpr (numOperands == 1) {
    return rewriter.create<Op>(loc, operands[0][i]);
  } else if constexpr (numOperands == 2) {
    return rewriter.create<Op>(loc, operands[0][i], operands[1][i]);
  } else if constexpr (numOperands == 3) {
    return rewriter.create<Op>(loc, operands[0][i], operands[1][i],
                               operands[2][i]);
  }
}

// This Pattern transforms arith/math ops where the ops have same arg and result type
// Example ops :
// math.exp {{%.*}} : vector<40x96xf32>
// arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
template <typename Op, int numOperands>
class WGToSGElementWiseOpSameArgAndResultTypePattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OneToNOpAdaptor =
      typename Op::template GenericAdaptor<ArrayRef<ValueRange>>;

  mlir::LogicalResult
  matchAndRewrite(Op op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return mlir::failure();
    }

    auto wgTileShape = resType.getShape();
    auto sgData = mapAttr.getSgData();
    auto sgLayout = mapAttr.getSgLayout();

    auto newTy =
        mlir::VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    // Get all the slices of Operands
    auto operands = adaptor.getOperands();

    llvm::SmallVector<llvm::SmallVector<mlir::Value>> operand;
    if (numOperands == 1)
      operand.push_back(operands[0]);
    else if (numOperands == 2) {
      operand.push_back(operands[0]);
      operand.push_back(operands[1]);
    } else {
      operand.push_back(operands[0]);
      operand.push_back(operands[1]);
      operand.push_back(operands[2]);
    }

    size_t numOps;
    if (sgLayout[0] * sgData[0] == wgTileShape[0] &&
        sgLayout[1] * sgData[1] == wgTileShape[1])
      numOps = 1; // 1:1 mapping
    else
      numOps = (wgTileShape[0] / (sgLayout[0] * sgData[0])) +
               (wgTileShape[1] / (sgLayout[1] * sgData[1]));

    llvm::SmallVector<::mlir::Value> newOps;
    for (size_t i = 0; i < numOps; i++) {
      auto newOp = createOp<Op, numOperands>(rewriter, op.getLoc(), operand, i);
      newOp->getResult(0).setType(newTy);
      newOps.push_back(newOp);
    }

    rewriter.replaceOpWithMultiple(op, {newOps});
    return mlir::success();
  }
};

// This Pattern trasforms arith ops where the ops have same shape as arg but
// different result type
// Example ops :
// arith.bitcast {{%.*}} : vector<32x32xf16> to vector<32x32xi16>
// arith.uitofp {{%.*}} : vector<32x32xi16> to vector<32x32xf16>
template <typename Op>
class WGToSGArithDifferentResultTypePattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OpAdaptor = typename Op::template GenericAdaptor<RangeT>;

  mlir::LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();

    auto newTy =
        mlir::VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    auto newOp = rewriter.create<Op>(op.getLoc(), newTy, adaptor.getOperands()[0]);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// arith::CmpIOp and arith::CmpFOp
template <typename Op>
class WGToSGElementWiseOpComparisonOpsPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OpAdaptor = typename Op::template GenericAdaptor<RangeT>;

  mlir::LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto arg = op.getLhs();
    auto argType = mlir::dyn_cast<mlir::VectorType>(arg.getType());
    auto result = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(result.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();

    auto newTy =
        mlir::VectorType::get({sgData[0], sgData[1]}, argType.getElementType());

    auto resTy =
        mlir::VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    auto newOp = rewriter.create<Op>(op.getLoc(), newTy, op.getPredicate(),
                                 adaptor.getLhs()[0], adaptor.getRhs()[0]);
    newOp->getResult(0).setType(resTy);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

static bool hasMap(mlir::Operation* op){
  if (llvm::isa<imex::xetile::LoadTileOp>(op)){
    auto tileTy =  mlir::dyn_cast<xetile::TileType>(op->getOperand(0).getType());
    if (tileTy.getWgMap())
      return true;
    else
      return false;
  }

  auto mapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
  auto wgMapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("wg_map_a"));
  if (!mapAttr && !wgMapAttr)
    return false;
  else
    return true;
}

// Helper function to analyze the def-use chain of initTileOps. Currently we
// pattern match the following def-use chain as a candidate for
// load + tranpose optimization.
void analyzeInitTileOps(mlir::Operation *op) {

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

    // First check for simple pattern of init -> load -> transpose
    mlir::Operation *loadUser = nullptr;
    auto initOpUser = *initOp->user_begin();
    if (llvm::isa<xetile::LoadTileOp>(initOpUser)){
      loadUser = initOpUser;
      ops.push_back(loadUser);
    }

    // InitTileOp must be consumed by a ForOp
    mlir::BlockArgument loopArg;
    if (auto scfFor = llvm::dyn_cast_if_present<mlir::scf::ForOp>(initOpUser)) {
      auto opArgs = imex::getArgsForOperand(scfFor, initOp.getResult());
      assert (opArgs.size() == 1 && "Duplicated tiles are not supported");
      auto argument = opArgs[0];
      for (auto user : argument.getUsers()) {
        if (llvm::isa<imex::xetile::LoadTileOp>(user)) {
          loadUser = user;
          ops.push_back(scfFor);
          ops.push_back(user);
        } else if (llvm::isa<xetile::UpdateTileOffsetOp>(user)) {
          ops.push_back(scfFor);
          ops.push_back(user);
        }
        // Nested scf.for's
        // init_tile -> scf.for -> update_tile_offset
        //                  |
        //               scf.for -> load_tile -> vector.transpose -> (pre-op) ->
        //               tile_mma
        else if (auto scfFor =
                     llvm::dyn_cast_if_present<mlir::scf::ForOp>(user)) {
          for (auto iterOperand : llvm::enumerate(scfFor.getInitArgs())) {
            if (iterOperand.value() == argument) {
              loopArg = scfFor.getRegionIterArgs()[iterOperand.index()];
              break;
            }
          }

          for (auto scfForUser : loopArg.getUsers()) {
            if (llvm::isa<xetile::LoadTileOp>(scfForUser)) {
              loadUser = scfForUser;
              ops.push_back(scfFor);
              ops.push_back(scfForUser);
            } else if (llvm::isa<xetile::UpdateTileOffsetOp>(
                           scfForUser)) {
              ops.push_back(scfFor);
              ops.push_back(scfForUser);
            }
          }
        }
      }
    }
    if (!loadUser)
      return mlir::WalkResult::skip();
    // LoadOp must be consumed by a transpose
    if (!(loadUser->hasOneUse() &&
          llvm::isa<mlir::vector::TransposeOp>(*loadUser->user_begin())))
      return mlir::WalkResult::skip();
    auto transposeOp =
        llvm::cast<mlir::vector::TransposeOp>(*loadUser->user_begin());
    ops.push_back(transposeOp);

    auto consumerOp = *transposeOp->user_begin();

    // Check if vector.transpose is consumed by TileMMA directly or
    // is consumed by some pre-op and then TileMMA.
    if (!llvm::isa<xetile::TileMMAOp>(consumerOp)) {
      if (!OpTrait::hasElementwiseMappableTraits(consumerOp) &&
          !(llvm::isa<mlir::vector::BroadcastOp>(consumerOp))) {
        return mlir::WalkResult::skip();
      } else {
        if (!(consumerOp->hasOneUse() &&
              llvm::isa<xetile::TileMMAOp>(*consumerOp->user_begin())))
          return mlir::WalkResult::skip();
      }
    }

    // At this point, we have a candidate def-use chain for optimization.
    for (auto op : ops) {
      if (op->getNumResults() > 0)
        opSgLayoutMap[op->getResult(0)] = {0, 1};
    }

    return mlir::WalkResult::advance();
  });
}

void populateXeTileWgToSgPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<WGToSGInitTileOpPattern, WGToSGLoadTileOpPattern,
                  WGToSGTileMMAOpPattern, WGToSGStoreTileOpPattern,
                  WGToSGSCFForOpPattern, WGToSGUpdateTileOffsetOpPattern,
                  WGToSGSCFYieldOpPattern, WGToSGVectorTranspose, WGToSGVectorBroadcast,
                  WGToSGXeTileConvertLayout, WGToSGPrefetchOpPattern,
                  WGToSGVectorShapeCast, WGToSGVectorMultiDimReductionOp
                  >(patterns.getContext());
  patterns.insert<WGToSGElementWiseOpSameArgAndResultTypePattern<mlir::math::ExpOp, 1>,
                  WGToSGElementWiseOpSameArgAndResultTypePattern<mlir::math::SqrtOp, 1>,
                  WGToSGElementWiseOpSameArgAndResultTypePattern<mlir::arith::AddFOp, 2>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::TruncFOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::TruncIOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::ExtFOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::ExtSIOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::ExtUIOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::SIToFPOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::UIToFPOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::FPToSIOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::FPToUIOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::IndexCastUIOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::IndexCastOp>,
                  WGToSGArithDifferentResultTypePattern<mlir::arith::BitcastOp>,
                  WGToSGElementWiseOpComparisonOpsPattern<mlir::arith::CmpIOp>,
                  WGToSGElementWiseOpComparisonOpsPattern<mlir::arith::CmpFOp>,
                  WGToSGArithConstantOpPattern>(patterns.getContext());
}

// Transforms WG XeTile IR to SG XeTile
class XeTileWgToSgPass
    : public impl::XeTileWgToSgBase<XeTileWgToSgPass> {

public:
  XeTileWgToSgPass() = default;

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    auto mod = this->getOperation();

    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    mlir::Operation *op = getOperation();
    // Run the analysis to find the candidates for the transformation
    analyzeInitTileOps(op);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);

    target.addDynamicallyLegalOp<xetile::InitTileOp>(
        [&](xetile::InitTileOp op) -> bool {
          if (!op.getType().getWgMap())
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<xetile::LoadTileOp>(
        [&](xetile::LoadTileOp op) -> bool {
          if (!op.getSource().getType().getWgMap())
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<xetile::TileMMAOp>(
        [&](xetile::TileMMAOp op) -> bool {
          auto mapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(
              op->getAttr("wg_map_a"));
          if (!mapAttr)
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<xetile::StoreTileOp>(
        [&](xetile::StoreTileOp op) -> bool {
          if (!op.getTile().getType().getWgMap())
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<xetile::UpdateTileOffsetOp>(
        [&](xetile::UpdateTileOffsetOp op) -> bool {
          if (!op.getType().getWgMap())
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<mlir::scf::ForOp>(
        [&](mlir::scf::ForOp op) -> bool {
          for (auto arg : op.getInitArgs()) {
            auto tileTy = mlir::dyn_cast<xetile::TileType>(arg.getType());
            auto vecTy =  mlir::dyn_cast<mlir::VectorType>(arg.getType());
            if (tileTy && tileTy.getWgMap())
              return false;
            if (vecTy && hasMap(arg.getDefiningOp()))
              return false;
          }
          return true;
        });

    target.addDynamicallyLegalOp<mlir::scf::YieldOp>(
        [&](mlir::scf::YieldOp op) -> bool {
          // For cases with scf.if having hidden yield
          for (auto result: op.getResults()) {
            auto tileTy = mlir::dyn_cast<xetile::TileType>(result.getType());
            auto vecTy =  mlir::dyn_cast<mlir::VectorType>(result.getType());
            if (tileTy && tileTy.getWgMap())
              return false;
            if (vecTy && hasMap(result.getDefiningOp()))
              return false;
          }
          return true;
        });

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp, mlir::arith::AddFOp,
                                 mlir::math::ExpOp, mlir::math::SqrtOp, mlir::arith::ExtFOp,
                                 mlir::arith::ExtSIOp, mlir::arith::ExtUIOp, mlir::arith::FPToSIOp,
                                 mlir::arith::FPToUIOp, mlir::arith::UIToFPOp, mlir::arith::SIToFPOp,
                                 mlir::arith::TruncFOp, mlir::arith::TruncIOp, mlir::arith::CmpIOp,
                                 mlir::arith::CmpFOp,  mlir::arith::IndexCastUIOp,
                                 mlir::arith::IndexCastOp, mlir::arith::BitcastOp, mlir::vector::TransposeOp,
                                 mlir::vector::BroadcastOp, mlir::vector::MultiDimReductionOp,
                                 mlir::vector::ShapeCastOp>(
        [&](mlir::Operation *op) -> bool {
          auto mapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(
              op->getAttr("map"));
          if (!mapAttr)
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<xetile::PrefetchTileOp>(
        [&](xetile::PrefetchTileOp op) -> bool {
          if (!op.getTile().getType().getWgMap())
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<mlir::scf::IfOp>(
    [&](mlir::scf::IfOp op) -> bool {
          return true;
    });

    target.addIllegalOp<xetile::ConvertLayoutOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    populateXeTileWgToSgPatterns(patterns);
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

/// Create a pass
std::unique_ptr<::mlir::Pass> createXeTileWgToSgPass() {
  return std::make_unique<XeTileWgToSgPass>();
}
} // namespace imex
