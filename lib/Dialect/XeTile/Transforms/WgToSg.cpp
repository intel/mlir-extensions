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
public:
  llvm::DenseMap<mlir::Value, std::array<int, 2>> &sgLayoutMap;
  WGToSGInitTileOpPattern(MLIRContext *context, llvm::DenseMap<mlir::Value, std::array<int, 2>> &map)
      : OpConversionPattern<xetile::InitTileOp>(context), sgLayoutMap(map) {}

  LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tileTy = op.getType();

    auto order = tileTy.getOrder();

    // TODO: Add support for order
    if (order[0] == 0 && order[1] == 1)
      return failure();

    auto elemTy = tileTy.getElementType();

    auto wgTileShape = tileTy.getShape();
    auto sgTileShape = tileTy.getWgMap().getSgData();
    auto sgLayout = tileTy.getWgMap().getSgLayout();

    auto newTileTy =
        imex::xetile::TileType::get({sgTileShape[0], sgTileShape[1]}, elemTy);

    rewriter.setInsertionPoint(op);
    // get the subgroup Id
    auto sgID = rewriter.create<mlir::gpu::SubgroupIdOp>(loc, nullptr);

    // Handle the init tile for scatter ops
    if (tileTy.getScatterAttr() == mlir::BoolAttr::get(op.getContext(), true)) {
      auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), nullptr /*sgMap*/, nullptr /*wgMap*/,
        mlir::DenseI32ArrayAttr::get(tileTy.getContext(), {1, 0}),
        mlir::IntegerAttr() /*array_length*/, tileTy.getMemorySpace(),
        tileTy.getScatterAttr() /*scatterAttr*/);
      auto newTileTy =
        imex::xetile::TileType::get({sgTileShape[0], sgTileShape[1]}, elemTy, attr);
      auto newInitTileOp = rewriter.create<xetile::InitTileOp>(
        loc, newTileTy, mlir::cast<mlir::TypedValue<mlir::MemRefType>>(op.getSource()),
        mlir::cast<mlir::TypedValue<mlir::VectorType>>(adaptor.getIndices()[0]));
      rewriter.replaceOp(op, newInitTileOp);
      return mlir::success();
    }

    auto createIndexConstant = [&](mlir::Type type, int64_t value) {
      auto attr = rewriter.getIndexAttr(value);
      return rewriter.create<arith::ConstantOp>(loc, type, attr);
    };

    auto indexType = rewriter.getIndexType();
    auto sgLayoutDimXConst = createIndexConstant(indexType, sgLayout[0]);
    auto sgLayoutDimYConst = createIndexConstant(indexType, sgLayout[1]);
    auto sgDataDimXConst = createIndexConstant(indexType, sgTileShape[0]);
    auto sgDataDimYConst = createIndexConstant(indexType, sgTileShape[1]);

    // The sgID is a linear (1D) id. Convert it to 2D to get the x and y
    // coordinates of sg
    // row = i / cols or i / rows if col_major
    // col =  i % cols or i % rows if col_major
    mlir::Value sgIdX;
    mlir::Value sgIdY;

    if (sgLayoutMap.find(op.getResult()) != sgLayoutMap.end()) {
      sgIdY =
          rewriter.create<mlir::index::DivUOp>(loc, sgID, sgLayoutDimXConst);
      sgIdX =
          rewriter.create<mlir::index::RemUOp>(loc, sgID, sgLayoutDimXConst);
    } else {
      sgIdY =
          rewriter.create<mlir::index::DivUOp>(loc, sgID, sgLayoutDimYConst);
      sgIdX =
          rewriter.create<mlir::index::RemUOp>(loc, sgID, sgLayoutDimYConst);
    }

    llvm::SmallVector<Value> offsets;
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = op.getOffsets();
    for (size_t i = 0, j = 0; i != staticOffsets.size(); i++) {
      if (ShapedType::isDynamic(staticOffsets[i])) {
        offsets.push_back(dynamicOffsets[j++]);
      } else {
        offsets.push_back(rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(staticOffsets[i])));
      }
    }
    mlir::Value source = op.getSource();
    mlir::SmallVector<mlir::OpFoldResult> globalOffsetsX; // rows
    mlir::SmallVector<mlir::OpFoldResult> globalOffsetsY; // cols
    mlir::SmallVector<mlir::SmallVector<mlir::OpFoldResult>> offsetPermutations;

    // Calculate the global offsets for tiles using the sgData and sgLayout
    // configuration given in wg_map. If each SG works on one output tile, each
    // WG level op will be transformed to same op with SG shapes/sizes. If each
    // SG needs to process more than one output tile, the WG level op will be
    // decomposed to multiple ops with SG level shapes/sizes
    auto calculateGlobalOffsets =
        [&](SmallVector<OpFoldResult> &globalOffsets,
            int wgTileShape, int sgTileShape, int sgLayout,
            Value sgDataDimConst, Value sgId, Value offset) {
          for (int i = 0; i < wgTileShape / sgTileShape; i += sgLayout) {
            auto constI = createIndexConstant(indexType, i);
            auto off =
                rewriter.createOrFold<index::AddOp>(loc, constI, sgId);
            auto mod = rewriter.createOrFold<index::RemUOp>(
                loc, off,
                createIndexConstant(indexType, wgTileShape / sgTileShape));
            auto localOffset = rewriter.createOrFold<index::MulOp>(
                loc, mod, sgDataDimConst);
            auto globalOffset = rewriter.createOrFold<index::AddOp>(
                loc, offset, localOffset);
            globalOffsets.push_back(globalOffset);
          }
        };
    // Look up the map if the init_tile has a layout_order [0, 1]
    // If it does, tranpose the sg ids to get the correct tile.
    auto it = sgLayoutMap.find(op.getResult());
    if (it != sgLayoutMap.end()){
     assert((sgLayoutMap[op->getResult(0)] == std::array<int, 2>{0, 1}));
     calculateGlobalOffsets(globalOffsetsX, wgTileShape[0], sgTileShape[0],
                           sgLayout[0], sgDataDimXConst, sgIdX, offsets[offsets.size() - 2]);
     calculateGlobalOffsets(globalOffsetsY, wgTileShape[1], sgTileShape[1],
                           sgLayout[1], sgDataDimYConst, sgIdY, offsets[offsets.size() - 1]);
    }
    else {
    calculateGlobalOffsets(globalOffsetsX, wgTileShape[0], sgTileShape[0],
                           sgLayout[0], sgDataDimXConst, sgIdY, offsets[offsets.size() - 2]);
    calculateGlobalOffsets(globalOffsetsY, wgTileShape[1], sgTileShape[1],
                           sgLayout[1], sgDataDimYConst, sgIdX, offsets[offsets.size() - 1]);
    }
    // TODO: check for how to broadcast
    for (auto y : globalOffsetsX) {
      for (auto x : globalOffsetsY) {
        offsetPermutations.push_back({y, x});
      }
    }

    SmallVector<Value> newInitTileOps;
    llvm::SmallVector<OpFoldResult> newOffsets;
    for (size_t j = 0; j < offsets.size() - 2; ++j) {
        newOffsets.push_back(offsets[j]);
    }
    for (size_t i = 0; i < offsetPermutations.size(); i++) {
      newOffsets.push_back(offsetPermutations[i][0]);
      newOffsets.push_back(offsetPermutations[i][1]);
      Value newInitTileOp = nullptr;
      auto sourceMemRefType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
      if (!sourceMemRefType) {
        return failure();
      }

      if (sourceMemRefType.hasStaticShape()) {
        newInitTileOp = rewriter.create<xetile::InitTileOp>(
            loc, newTileTy, source, newOffsets);
      }
      else {
        newInitTileOp = rewriter.create<xetile::InitTileOp>(
            loc, newTileTy, source, newOffsets, op.getMixedSizes(), op.getMixedStrides());
      }
      newOffsets.clear();
      newInitTileOps.push_back(newInitTileOp);
    }

    rewriter.replaceOpWithMultiple(op, {newInitTileOps});
    return success();
  }
};

class WGToSGLoadTileOpPattern : public OpConversionPattern<xetile::LoadTileOp> {
  using OpConversionPattern<xetile::LoadTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto sources = adaptor.getTile();
    auto res = op.getValues()[0];

    auto resType = dyn_cast<VectorType>(res.getType());

    if (!resType || resType.getRank() != 2)
      return failure();

    llvm::SmallVector<::mlir::Value> newLoadOps;
    for (auto src : sources) {
      auto tileTy = llvm::dyn_cast<xetile::TileType>(src.getType());
      auto newResTy =
          VectorType::get({tileTy.getShape()[0], tileTy.getShape()[1]},
                                tileTy.getElementType());
      auto newLoadOp = rewriter.create<xetile::LoadTileOp>(
          op.getLoc(), newResTy, src, op->getAttrs()).getResult(0);
      newLoadOps.push_back(newLoadOp);
    }
    rewriter.replaceOpWithMultiple(op, {newLoadOps});
    return mlir::success();
  }
};

class WGToSGLoadGatherOpPattern : public OpConversionPattern<xetile::LoadGatherOp> {
  using OpConversionPattern<xetile::LoadGatherOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::LoadGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto sources = adaptor.getTile();
    auto mask = adaptor.getMask();
    auto res = op.getValue();
    auto resType = res.getType();

    if (!resType || resType.getRank() != 2)
      return mlir::failure();

    llvm::SmallVector<::mlir::Value> newLoadOps;
    for (auto [src, mask] : llvm::zip(sources, mask)) {
      auto tileTy = llvm::dyn_cast<xetile::TileType>(src.getType());
      auto newResTy =
          mlir::VectorType::get({tileTy.getShape()[0], tileTy.getShape()[1]},
          tileTy.getElementType());
      auto newLoadOp = rewriter.create<xetile::LoadGatherOp>(
          op.getLoc(), newResTy, src, mask, op.getPaddingAttr(),
          op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr());
      newLoadOps.push_back(newLoadOp);
    }
    rewriter.replaceOpWithMultiple(op, {newLoadOps});
    return success();
  }
};

class WGToSGTileMMAOpPattern : public OpConversionPattern<xetile::TileMMAOp> {
  using OpConversionPattern<xetile::TileMMAOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = op.getResult().getType();

    if (resultTy.getRank() != 2)
      return failure();

    llvm::SmallVector<Value> newTileMMAOps;
    llvm::SmallVector<Type> newResultTypes;
    size_t i = 0;
    for (auto a : adaptor.getA()) {
      for (auto b : adaptor.getB()) {

        Value tmpC;
        if (op.getC())
          tmpC = adaptor.getC()[i++];

        auto aShape = llvm::cast<VectorType>(a.getType()).getShape();
        auto bShape = llvm::cast<VectorType>(b.getType()).getShape();
        auto resTy = VectorType::get({aShape[0], bShape[1]},
                                           resultTy.getElementType());
        tmpC = rewriter.create<xetile::TileMMAOp>(
            op.getLoc(), resTy, a, b, tmpC, nullptr, nullptr, nullptr);
        newTileMMAOps.push_back(tmpC);
        newResultTypes.push_back(resTy);
      }
    }

    rewriter.replaceOpWithMultiple(op, {newTileMMAOps});
    return success();
  }
};

class WGToSGStoreTileOpPattern : public OpConversionPattern<xetile::StoreTileOp> {
  using OpConversionPattern<xetile::StoreTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newValues = adaptor.getValue();
    auto newDstTiles = adaptor.getTile();

    for (size_t i = 0; i < newValues.size(); i++) {
      rewriter.create<xetile::StoreTileOp>(op.getLoc(), newValues[i],
                                           newDstTiles[i], op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class WGToSGStoreScatterOpPattern : public OpConversionPattern<xetile::StoreScatterOp> {
  using OpConversionPattern<xetile::StoreScatterOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(xetile::StoreScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newValues = adaptor.getValue();
    auto newDstTiles = adaptor.getTile();
    auto mask = adaptor.getMask();

    for (size_t i = 0; i < newValues.size(); i++) {
      rewriter.create<xetile::StoreScatterOp>(op.getLoc(), newValues[i],
                                           newDstTiles[i], mask[i], op.getL1HintAttr(),
                                           op.getL2HintAttr(), op.getL3HintAttr());
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class WGToSGSCFForOpPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OneToNOpAdaptor adaptor,
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

struct WGToSGSCFYieldOpPattern : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OneToNOpAdaptor adaptor,
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

struct WGToSGSCFIfOpPattern : public OpConversionPattern<mlir::scf::IfOp> {
  using OpConversionPattern<mlir::scf::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op, OneToNOpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const override {
    if (op.getThenRegion().empty()) {
      return mlir::failure();
    }

    // Get the yield ops from then region
    auto thenYield = cast<scf::YieldOp>(op.getThenRegion().front().getTerminator());
    if (thenYield.getNumOperands() != op.getNumResults()) {
      return mlir::failure();
    }

    SmallVector<mlir::Type> newResultTypes;
    bool needsChange = false;

    // Check if inside an scf.for
    if (auto parentFor = dyn_cast<scf::ForOp>(op->getParentOp())) {
      // Use parent for's iter_args types
      for (size_t i = 0; i < parentFor.getNumResults(); ++i) {
        mlir::Type iterType = parentFor.getRegionIterArgs()[i].getType();
        if (i < op.getNumResults() && iterType != op.getResultTypes()[i]) {
          newResultTypes.push_back(iterType);
          needsChange = true;
        } else {
          newResultTypes.push_back(op.getResultTypes()[i]);
        }
      }
    } else {
      for (size_t i = 0; i < op.getNumResults(); ++i) {
        mlir::Value thenVal = thenYield.getOperand(i);
        mlir::Type currentType = op.getResultTypes()[i];

        if (auto defOp = thenVal.getDefiningOp()) {
          if (auto mapAttr = defOp->getAttrOfType<xetile::WorkGroupMapAttr>("map")) {
            auto sgData = mapAttr.getSgData();
            auto expectedShape = llvm::SmallVector<int64_t>{sgData[0], sgData[1]};
            mlir::Type expectedType = mlir::VectorType::get(
                expectedShape, cast<mlir::VectorType>(thenVal.getType()).getElementType());
            if (currentType != expectedType) {
              newResultTypes.push_back(expectedType);
              needsChange = true;
            } else {
              newResultTypes.push_back(currentType);
            }
            continue;
          }
        }
        newResultTypes.push_back(currentType);
      }
    }

    if (!needsChange) {
      return mlir::failure();
    }

    auto newIfOp = rewriter.create<scf::IfOp>(
        op.getLoc(), newResultTypes, op.getCondition(), !op.getElseRegion().empty());

    // Move the then region to the new if op
    rewriter.eraseBlock(&newIfOp.getThenRegion().front());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().end());

    // Handle the else region if it exists
    if (!op.getElseRegion().empty()) {
      rewriter.eraseBlock(&newIfOp.getElseRegion().front());
      rewriter.inlineRegionBefore(op.getElseRegion(), newIfOp.getElseRegion(),
                                  newIfOp.getElseRegion().end());

      // Adjust else yield if inside scf.for
      if (auto parentFor = dyn_cast<scf::ForOp>(op->getParentOp())) {
        auto *elseBlock = &newIfOp.getElseRegion().front();
        auto elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());
        SmallVector<mlir::Value> newOperands;

        for (size_t i = 0; i < elseYield.getNumOperands(); ++i) {
          if (elseYield.getOperand(i).getType() != newResultTypes[i]) {
            newOperands.push_back(parentFor.getRegionIterArgs()[i]);
          } else {
            newOperands.push_back(elseYield.getOperand(i));
          }
        }

        rewriter.setInsertionPoint(elseYield);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(elseYield, newOperands);
      }
    }

    rewriter.replaceOp(op, newIfOp.getResults());
    return mlir::success();
  }
};

class WGToSGUpdateTileOffsetOpPattern
    : public OpConversionPattern<xetile::UpdateTileOffsetOp> {
  using OpConversionPattern<xetile::UpdateTileOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> newUpdateTileOffsetOps;
    llvm::SmallVector<Type> newResultTypes;
    for (auto tile : adaptor.getTile()) {

      auto newUpdateTileOffsetOp = rewriter.create<xetile::UpdateTileOffsetOp>(
          op.getLoc(), tile.getType(), tile, op.getOffsetX(), op.getOffsetY(), op.getIndices());
      newUpdateTileOffsetOps.push_back(newUpdateTileOffsetOp);
      newResultTypes.push_back(tile.getType());
    }

    rewriter.replaceOpWithMultiple(op, {newUpdateTileOffsetOps});
    return success();
  }
};

class WGToSGArithConstantOpPattern
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    auto valueType = mlir::dyn_cast<mlir::VectorType>(value.getType());
    auto wgTileShape = valueType.getShape();

    if (!value)
      return failure();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return failure();
    }

    auto sgData = mapAttr.getSgData();
    auto sgLayout = mapAttr.getSgLayout();
    SmallVector<int64_t> outputShape;
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
        VectorType::get(outputShape, value.getElementType());

    llvm::SmallVector<Attribute> elems(
        value.value_begin<Attribute>(),
        value.value_end<Attribute>());

    llvm::SmallVector<Attribute> newValues;
    for (int64_t i = 0; i < static_cast<int64_t>(sgData[0]) * sgData[1]; i++) {
      newValues.push_back(elems[i]);
    }

    auto attr = DenseElementsAttr::get(newTy, newValues);

    size_t numOps;
    // If WG tile is 1D vector just support 1:1 mapping.
    // TODO: Support round robin for 1D
    if(wgTileShape.size() == 1) {
      if (sgLayout[0] * sgData[0] == wgTileShape[0] ||
          sgLayout[1] * sgData[1] == wgTileShape[0])
            numOps = 1;
      else
        return failure();
    } else if(sgLayout[0] * sgData[0] == wgTileShape[0] &&
              sgLayout[1] * sgData[1] == wgTileShape[1]) {
                 numOps = 1;
    } else
        numOps = (wgTileShape[0] / (sgLayout[0] * sgData[0])) +
                 (wgTileShape[1] / (sgLayout[1] * sgData[1]));

    llvm::SmallVector<Value> newOps;
    for (size_t i = 0; i < numOps; i++) {
      auto newOp = rewriter.create<arith::ConstantOp>(op.getLoc(), newTy, attr);
      newOps.push_back(newOp);
    }
    rewriter.replaceOpWithMultiple(op, {newOps});
    return success();
  }
};

class WGToSGVectorTranspose
    :public OpConversionPattern<mlir::vector::TransposeOp> {
  using OpConversionPattern<mlir::vector::TransposeOp>::OpConversionPattern;
public:
  llvm::DenseMap<mlir::Value, std::array<int, 2>> &sgLayoutMap;
  WGToSGVectorTranspose(MLIRContext *context, llvm::DenseMap<mlir::Value, std::array<int, 2>> &map)
      : OpConversionPattern<mlir::vector::TransposeOp>(context), sgLayoutMap(map) {}

  LogicalResult
  matchAndRewrite(vector::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVector().getType().getRank() != 2)
      return failure();

    auto res = op.getResult();
    auto resType = dyn_cast<VectorType>(res.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return failure();
    }

    auto it = sgLayoutMap.find(op.getResult());
    // Transpose within subgroup if the sg layout order is {0, 1}
    if (it != sgLayoutMap.end()){
      assert((sgLayoutMap[op->getResult(0)] == std::array<int, 2>{0, 1}));
      auto sgData = mapAttr.getSgData();
      auto newTy = VectorType::get({sgData[0], sgData[1]},
                                                   resType.getElementType());
      auto newOp = rewriter.create<vector::TransposeOp>(
              op.getLoc(), newTy, adaptor.getVector(), op.getPermutation());
      rewriter.replaceOp(op, newOp);
      return success();
    }
    else
    {
      //TODO : Transpose using SLM
      return failure();
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

// If the input value is defined by a transpose op, it also try to fold the transpose effect
// into the store op to the slm using a transposed view.

// Example:
// WG IR
// #wg_map_c = #xetile.wg_map<sg_layout = [4, 8], sg_data = [64, 32]>
// #wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>
// #wg_map_a = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>
// %vector_b = xetile.transpose %c {#wg_map_c} : vector<256x256xfloat> -> vector<256x256xfloat>
// %vector_a = xetile.tile_conv_layout %vector_b {wg_map_result = #wg_map_a, wg_map_source = #wg_map_b}: vector<256x256xfloat> into vector<256x256xfloat>

// SG IR
// %slm = memref.alloc() : memref<256x256xf32, 3>
// %view = memref.transpose %slm : memref<256x256xf32, 3> to memref<256x256xf32, strided<[1, 256]>, 3>
// %tile = xetile.init_tile %view[offset_x, offset_y] : memref<256x256xf32, strided<[1, 256]>, 3> -> xetile.tile<64x32xf32, #xetile.tile_attr<order=[0, 1]>>
// xetile.store_tile %in, %tile :vector<64x32xf32>, !xetile.tile<64x32xf32, #xetile.tile_attr<order=[0, 1]>>
// gpu.barrier
// %remapped_tile = xetile.init_tile %slm[offsetX, offsetY] : memref<256x256xf32, 3> -> xetile.tile<8x256xf32>
// %remapped_vector = xetile.load_tile %reshaped_tile : xetile.tile<8x256xf32> -> vector<8x256xf32>

class WGToSGXeTileConvertLayout
    :public OpConversionPattern<xetile::ConvertLayoutOp> {
  using OpConversionPattern<xetile::ConvertLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xetile::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getSource().getType().getRank() != 2)
      return failure();

    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto res = op.getResult();
    auto defOp = op.getSource().getDefiningOp();
    auto resType = res.getType();
    auto elemTy = resType.getElementType();
    auto resShape = resType.getShape();
    auto slmScopeAttr = rewriter.getI32IntegerAttr(3);

    auto createIndexConstant = [&](int64_t value) {
      return rewriter.create<arith::ConstantIndexOp>(loc, value);
    };

    // get the workgroup map attribute for a value from its defining op.
    auto getWorkGroupMapAttr = [&](Value val) {
      auto defOp = val.getDefiningOp();
      if (auto ld = dyn_cast<xetile::LoadTileOp>(defOp)) {
        return ld.getTile().getType().getWgMap();
      }
      return defOp->getAttrOfType<xetile::WorkGroupMapAttr>("map");
    };

    auto isOneUseTranspose = [&](Operation *op) {
      return isa<xetile::TransposeOp, vector::TransposeOp>(op) && op->hasOneUse();
    };

    auto getOffsets = [&](Value sgId, DenseI32ArrayAttr sgLayout, DenseI32ArrayAttr sgData) {
      // The sgID is a linear (1D) id. Convert it to 2D to get the x and y
      // coordinates of sg
      // row = i / cols
      // col =  i % cols
      // x is row, y is col
      // TODO: Div and Rem are expensive. Find alterate.
      auto dimY = createIndexConstant(sgLayout[1]);
      auto sgIdX = rewriter.create<index::DivUOp>(loc, sgId, dimY);
      auto sgIdY = rewriter.create<index::RemUOp>(loc, sgId, dimY);

      auto offsetX = rewriter.createOrFold<index::MulOp>(loc, sgIdX, createIndexConstant(sgData[0]));
      auto offsetY = rewriter.createOrFold<index::MulOp>(loc, sgIdY, createIndexConstant(sgData[1]));
      return std::make_pair(offsetX, offsetY);
    };

    auto srcMapAttr = isOneUseTranspose(defOp) ? getWorkGroupMapAttr(defOp->getOperand(0))
                                               : op->hasAttr("wg_map_source") ? op->getAttrOfType<xetile::WorkGroupMapAttr>("wg_map_source")
                                               : getWorkGroupMapAttr(op.getSource());

    auto dstMapAttr = op->getAttrOfType<xetile::WorkGroupMapAttr>("wg_map_result");

    if (!srcMapAttr || !dstMapAttr)
      return failure();

    rewriter.setInsertionPoint(op);

    // Allocate SLM
    auto bitWidth = elemTy.getIntOrFloatBitWidth();
    auto flattenFactor = bitWidth / 8;
    auto slmSize = resType.getNumElements() * flattenFactor;
    auto slmTy = MemRefType::get(slmSize, rewriter.getI8Type(), {}, 3);
    auto slm = rewriter.create<memref::AllocOp>(loc, slmTy);
    auto viewTy = MemRefType::get(resShape, elemTy, {}, 3);
    auto view = rewriter.create<memref::ViewOp>(loc, viewTy, slm, createIndexConstant(0), ValueRange());

    // Get SG id
    auto sgId = rewriter.create<gpu::SubgroupIdOp>(loc, rewriter.getIndexType(), nullptr);

    { // store to slm
      auto sgData = srcMapAttr.getSgData();
      auto sgLayout = srcMapAttr.getSgLayout();

      auto [offsetX, offsetY] = getOffsets(sgId, sgLayout, sgData);

      Value stView = view;
      Value data = adaptor.getSource();
      DenseI32ArrayAttr order = rewriter.getDenseI32ArrayAttr({1, 0});
      if (isOneUseTranspose(defOp)) {
        data = rewriter.getRemappedValue(defOp->getOperand(0));
        order = rewriter.getDenseI32ArrayAttr({0, 1});

        auto permMap = AffineMap::getPermutationMap(llvm::ArrayRef<int64_t>({1, 0}), ctx);
        auto permAttr = AffineMapAttr::get(permMap);
        stView = rewriter.create<memref::TransposeOp>(loc, view, permAttr);
      }

      auto attr = imex::xetile::XeTileAttr::get(
                      ctx, nullptr /*sgMap*/, nullptr /*wgMap*/, order,
                      nullptr /*array_length*/, slmScopeAttr, nullptr /*scatterAttr*/);
      auto tileTy = imex::xetile::TileType::get({sgData[0], sgData[1]}, elemTy, attr);

      auto tile = rewriter.create<xetile::InitTileOp>(
                      loc, tileTy, stView, llvm::ArrayRef<OpFoldResult>({offsetX, offsetY}));
      rewriter.create<xetile::StoreTileOp>(loc, data, tile, nullptr, nullptr, nullptr);
    }

    // Add barrier to wait for all threads to finish writing to SLM
    rewriter.create<gpu::BarrierOp>(loc);

    { // load from slm
      auto sgData = dstMapAttr.getSgData();
      auto sgLayout = dstMapAttr.getSgLayout();

      auto [offsetX, offsetY] = getOffsets(sgId, sgLayout, sgData);
      offsetX = rewriter.createOrFold<index::RemUOp>(
                    loc, offsetX, createIndexConstant(resShape[0]));
      offsetY = rewriter.createOrFold<index::RemUOp>(
                    loc, offsetY, createIndexConstant(resShape[1]));

      auto order = rewriter.getDenseI32ArrayAttr({1, 0});
      auto attr = imex::xetile::XeTileAttr::get(
                      ctx, nullptr /*sgMap*/, nullptr /*wgMap*/, order,
                      nullptr /*array_length*/, slmScopeAttr, nullptr /*scatterAttr*/);
      auto tileTy = xetile::TileType::get({sgData[0], sgData[1]}, elemTy, attr);
      auto newResTy = VectorType::get({sgData[0], sgData[1]}, elemTy);

      auto tile = rewriter.create<xetile::InitTileOp>(
                    loc, tileTy, view, llvm::ArrayRef<OpFoldResult>({offsetX, offsetY}));
      //TODO: Set up cache attributes
      auto ld = rewriter.create<xetile::LoadTileOp>(
                    loc, newResTy, tile, Attribute(), nullptr, nullptr, nullptr);
      rewriter.replaceOp(op, ld);
    }

    if (isOneUseTranspose(defOp))
      rewriter.eraseOp(defOp);

    return success();
  }
};

class WGToSGVectorBroadcast
    :public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVector().getType().getRank() != 2)
      return failure();

    auto res = op.getResult();
    auto resType = dyn_cast<VectorType>(res.getType());

    auto srcTy =  dyn_cast<VectorType>((adaptor.getSource()).getType());
    auto srcShape = srcTy.getShape();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return failure();
    }

    auto sgData = mapAttr.getSgData();
    auto newTy = VectorType::get({sgData[0], sgData[1]},
                                       resType.getElementType());
    auto dstShape = newTy.getShape();

    if (!(srcShape[0] == 1 && srcShape[1] == dstShape[1]) &&
        !(srcShape[1] == 1 && srcShape[0] == dstShape[0]))
      return failure();

    auto newOp = rewriter.create<vector::BroadcastOp>(
            op.getLoc(), newTy, adaptor.getSource());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class WGToSGPrefetchOpPattern : public OpConversionPattern<xetile::PrefetchTileOp> {
  using OpConversionPattern<xetile::PrefetchTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto L1 = op.getL1HintAttr();
    auto L2 = op.getL2HintAttr();
    auto L3 = op.getL3HintAttr();

    for(auto tile : adaptor.getTile()) {
      rewriter.create<xetile::PrefetchTileOp>(op.getLoc(),  tile, L1, L2, L3);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class WGToSGVectorMultiDimReductionOp
    : public OpConversionPattern<vector::MultiDimReductionOp> {
  using OpConversionPattern<vector::MultiDimReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MultiDimReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = dyn_cast<VectorType>(res.getType());
    auto resRank = resType.getShape().size();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return failure();
    }

    auto sgData = mapAttr.getSgData();

    auto src = adaptor.getSource();
    auto srcType = dyn_cast<VectorType>(src.getType());

    if (resRank == 2) {
      bool newReduceDim = sgData[0] == 1 ? 0 : 1;
      SmallVector<int64_t> redDims{newReduceDim};
      auto outputShape =
          newReduceDim == 0 ? srcType.getDimSize(1) : srcType.getDimSize(0);
      auto newTy = VectorType::get(outputShape, srcType.getElementType());

      // ShapeCast acc to match reduction op shape.
      auto acc = rewriter.create<vector::ShapeCastOp>(op->getLoc(), newTy,
                                                      adaptor.getAcc());

      auto newOp = rewriter.create<vector::MultiDimReductionOp>(
          op.getLoc(), newTy, op.getKind(), src, acc, redDims);

      // Shape Cast the output of reduction back to 2D
      auto accumalator = adaptor.getAcc();
      auto accumalatorType =
          dyn_cast<VectorType>(accumalator.getType());
      auto outputVectorTy = VectorType::get(
          accumalatorType.getShape(), accumalatorType.getElementType());
      auto shapeCastOp = rewriter.create<vector::ShapeCastOp>(
          op.getLoc(), outputVectorTy, newOp);
      rewriter.replaceOp(op, shapeCastOp);
      return success();
    }
    // Regular 2D vector.multi_reduction
    else {
      auto reductionDims = op.getReductionDims();
      if (reductionDims.size() != 1)
        return failure();

      bool reduceDim = reductionDims[0];
      auto outputShape =
          reduceDim == 0 ? srcType.getDimSize(1) : srcType.getDimSize(0);

      SmallVector<int64_t> redDims{reduceDim};
      auto newTy = VectorType::get(outputShape, srcType.getElementType());
      auto newOp = rewriter.create<vector::MultiDimReductionOp>(
          op.getLoc(), newTy, op.getKind(), adaptor.getSource(),
          adaptor.getAcc(), redDims);
      rewriter.replaceOp(op, newOp);
      return success();
    }
  }
};

// Shape cast will support going from 1D to 2D since the vector.multi_reduction
// produces 1D

class WGToSGVectorShapeCast
    : public OpConversionPattern<vector::ShapeCastOp> {
  using OpConversionPattern<vector::ShapeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShapeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = dyn_cast<VectorType>(res.getType());
    auto resShape = resType.getShape();

    // Assumption is 3D shape cast is used for partial reduction.
    // So just replace it with the transformed source of shape_cast
    if (resShape.size() == 3) {
      for (Operation *userOp : op.getResult().getUsers()) {
        // Check if the user operation is not a vector.multi_reduction
        if (!isa<vector::MultiDimReductionOp>(userOp)) {
          return failure();
        }
      }
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }

    // One of the dims have to be a unit dim
    if (resShape[0] != 1 && resShape[1] != 1)
      return failure();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return failure();
    }

    auto sgData = mapAttr.getSgData();
    auto newTy =
        VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    auto newOp = rewriter.create<vector::ShapeCastOp>(
        op.getLoc(), newTy, adaptor.getSource());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class WGToSGVectorCreateMask
    :public OpConversionPattern<mlir::vector::CreateMaskOp> {
  using OpConversionPattern<mlir::vector::CreateMaskOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::CreateMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));

    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();
    auto newTy = mlir::VectorType::get({sgData[0], sgData[1]},
                                       resType.getElementType());

    auto newOp = rewriter.create<mlir::vector::CreateMaskOp>(
            op.getLoc(), newTy, adaptor.getOperands());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

template <typename Op, int numOperands>
Op createOp(ConversionPatternRewriter &rewriter, Location loc,
            llvm::SmallVector<llvm::SmallVector<Value>> operands, int i) {
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
  using RangeT = llvm::ArrayRef<ValueRange>;
  using OneToNOpAdaptor =
      typename Op::template GenericAdaptor<ArrayRef<ValueRange>>;

  LogicalResult
  matchAndRewrite(Op op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res = op.getResult();
    auto resType = dyn_cast<VectorType>(res.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return failure();
    }

    auto wgTileShape = resType.getShape();
    auto sgData = mapAttr.getSgData();
    auto sgLayout = mapAttr.getSgLayout();

    auto newTy =
        VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    // Get all the slices of Operands
    auto operands = adaptor.getOperands();

    llvm::SmallVector<llvm::SmallVector<Value>> operand;
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
   if (sgLayout[0] * sgData[0] == wgTileShape[0] ||
       sgLayout[1] * sgData[1] == wgTileShape[1] ||
       sgLayout[1] * sgData[0] == wgTileShape[0] ||  // For pre-op between load
       sgLayout[0] * sgData[1] == wgTileShape[1])    // & transpose
      numOps = 1; // 1:1 mapping
    else
      numOps = (wgTileShape[0] / (sgLayout[0] * sgData[0])) +
               (wgTileShape[1] / (sgLayout[1] * sgData[1]));

    llvm::SmallVector<Value> newOps;
    for (size_t i = 0; i < numOps; i++) {
      auto newOp = createOp<Op, numOperands>(rewriter, op.getLoc(), operand, i);
      newOp->getResult(0).setType(newTy);
      newOps.push_back(newOp);
    }

    rewriter.replaceOpWithMultiple(op, {newOps});
    return success();
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
  using RangeT = llvm::ArrayRef<ValueRange>;
  using OpAdaptor = typename Op::template GenericAdaptor<RangeT>;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = dyn_cast<VectorType>(res.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return failure();
    }

    auto sgData = mapAttr.getSgData();

    auto newTy =
        VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    auto newOp = rewriter.create<Op>(op.getLoc(), newTy, adaptor.getOperands()[0]);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// arith::CmpIOp and arith::CmpFOp
template <typename Op>
class WGToSGElementWiseOpComparisonOpsPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using RangeT = llvm::ArrayRef<ValueRange>;
  using OpAdaptor = typename Op::template GenericAdaptor<RangeT>;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto arg = op.getLhs();
    auto argType = dyn_cast<VectorType>(arg.getType());
    auto result = op.getResult();
    auto resType = dyn_cast<VectorType>(result.getType());

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return failure();
    }

    auto sgData = mapAttr.getSgData();

    auto newTy =
        VectorType::get({sgData[0], sgData[1]}, argType.getElementType());

    auto resTy =
        VectorType::get({sgData[0], sgData[1]}, resType.getElementType());

    auto newOp = rewriter.create<Op>(op.getLoc(), newTy, op.getPredicate(),
                                 adaptor.getLhs()[0], adaptor.getRhs()[0]);
    newOp->getResult(0).setType(resTy);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class WGToSGArithSelectOpPattern : public OpConversionPattern<mlir::arith::SelectOp> {
  using OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return mlir::failure();
    }

    auto newOp = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), adaptor.getCondition(),
                                 adaptor.getTrueValue(), adaptor.getFalseValue());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

class WGToSGMathFPowIOpPattern : public OpConversionPattern<mlir::math::FPowIOp> {
  using OpConversionPattern<mlir::math::FPowIOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::math::FPowIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return mlir::failure();
    }

    auto newOp = rewriter.create<mlir::math::FPowIOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};



static bool hasMap(mlir::Operation* op){
  if (llvm::isa<imex::xetile::LoadTileOp>(op)){
    auto tileTy =  dyn_cast<xetile::TileType>(op->getOperand(0).getType());
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


// This function traverses backwards through loop-carried dependencies in SCF
// `for` loops to find the original (pre-loop) value.
static Value getPreLoopValue(Value val) {
  while (auto blockArg = mlir::dyn_cast<BlockArgument>(val)) {
    if (auto forOp = mlir::dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      unsigned argIndex = blockArg.getArgNumber();
      unsigned numIterArgs = forOp.getInitArgs().size();
      unsigned firstIterArgIdx = forOp.getRegion().getArguments().size() - numIterArgs;

      if (argIndex >= firstIterArgIdx) {
        val = forOp.getInitArgs()[argIndex - firstIterArgIdx]; // Corrected index
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return val;
}

// Generic function to find all operations of type OpType contributing to a value
template <typename OpType>
SmallVector<Operation *> findOps(Value val) {
  SmallVector<Operation *> matchedOps;
  SmallVector<Value, 4> worklist{val};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue; // Avoid cycles

    // Handle scf.for iter_args
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      current = getPreLoopValue(current);
    }

    // Check if the defining operation is of the desired type
    if (Operation *defOp = current.getDefiningOp()) {
      if (llvm::isa<OpType>(defOp)) matchedOps.push_back(defOp);
      for (Value operand : defOp->getOperands()) {
        worklist.push_back(operand);
      }
    }
  }
  return matchedOps;
}

// Analyze transpose operations and track corresponding loads and initOps
static void analyzeTransposeOps(mlir::Operation *op,
                         llvm::DenseMap<mlir::Value, std::array<int, 2>> &sgLayoutMap) {

  op->walk([&](mlir::vector::TransposeOp transposeOp) -> mlir::WalkResult {
    Value transposeInput = transposeOp->getOperand(0);

    // Find all LoadTileOps leading to this transpose
    SmallVector<Operation *> loadOps = findOps<imex::xetile::LoadTileOp>(transposeInput);
    if (loadOps.empty())
      return mlir::WalkResult::skip();

    for (Operation *loadOp : loadOps) {
      Value loadSource = loadOp->getOperand(0);

      // Find corresponding InitOps
      SmallVector<Operation *> initOps = findOps<imex::xetile::InitTileOp>(loadSource);
      if (initOps.empty())
        continue;

       sgLayoutMap[transposeOp->getResult(0)] = {0, 1};
      // Update sgLayoutMap for all relevant initOps
      for (Operation *initOp : initOps) {
        // If the tranpose is already present. We need to mark it row major.
        if (sgLayoutMap.find(initOp->getResult(0)) != sgLayoutMap.end()) {
          sgLayoutMap.erase(initOp->getResult(0));
        }
        else {
          sgLayoutMap[initOp->getResult(0)] = {0, 1};
        }
      }
    }
    return mlir::WalkResult::advance();
  });
}

void populateXeTileWgToSgPatterns(mlir::RewritePatternSet &patterns,
                                  llvm::DenseMap<mlir::Value, std::array<int, 2>> &sgLayoutMap) {
  patterns.insert<WGToSGInitTileOpPattern, WGToSGVectorTranspose>(patterns.getContext(),
                  sgLayoutMap);
  patterns.insert<WGToSGLoadTileOpPattern, WGToSGTileMMAOpPattern, WGToSGStoreTileOpPattern,
                  WGToSGSCFForOpPattern, WGToSGUpdateTileOffsetOpPattern,
                  WGToSGSCFYieldOpPattern, WGToSGVectorBroadcast, WGToSGSCFIfOpPattern,
                  WGToSGXeTileConvertLayout, WGToSGPrefetchOpPattern,
                  WGToSGVectorShapeCast, WGToSGVectorMultiDimReductionOp,
                  WGToSGArithSelectOpPattern, WGToSGMathFPowIOpPattern,
                  WGToSGVectorShapeCast, WGToSGVectorMultiDimReductionOp,
                  WGToSGLoadGatherOpPattern, WGToSGStoreScatterOpPattern,
                  WGToSGVectorCreateMask>(patterns.getContext());
  patterns.insert<WGToSGElementWiseOpSameArgAndResultTypePattern<math::ExpOp, 1>,
                  WGToSGElementWiseOpSameArgAndResultTypePattern<math::SqrtOp, 1>,
                  WGToSGElementWiseOpSameArgAndResultTypePattern<arith::AddFOp, 2>,
                  WGToSGArithDifferentResultTypePattern<arith::TruncFOp>,
                  WGToSGArithDifferentResultTypePattern<arith::TruncIOp>,
                  WGToSGArithDifferentResultTypePattern<arith::ExtFOp>,
                  WGToSGArithDifferentResultTypePattern<arith::ExtSIOp>,
                  WGToSGArithDifferentResultTypePattern<arith::ExtUIOp>,
                  WGToSGArithDifferentResultTypePattern<arith::SIToFPOp>,
                  WGToSGArithDifferentResultTypePattern<arith::UIToFPOp>,
                  WGToSGArithDifferentResultTypePattern<arith::FPToSIOp>,
                  WGToSGArithDifferentResultTypePattern<arith::FPToUIOp>,
                  WGToSGArithDifferentResultTypePattern<arith::IndexCastUIOp>,
                  WGToSGArithDifferentResultTypePattern<arith::IndexCastOp>,
                  WGToSGArithDifferentResultTypePattern<arith::BitcastOp>,
                  WGToSGElementWiseOpComparisonOpsPattern<arith::CmpIOp>,
                  WGToSGElementWiseOpComparisonOpsPattern<arith::CmpFOp>,
                  WGToSGArithConstantOpPattern>(patterns.getContext());
}

// Transforms WG XeTile IR to SG XeTile
class XeTileWgToSgPass
    : public impl::XeTileWgToSgBase<XeTileWgToSgPass> {

public:
  XeTileWgToSgPass() = default;

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

  llvm::DenseMap<mlir::Value, std::array<int, 2>> sgLayoutMap;

  void runOnOperation() override {
    MLIRContext &context = getContext();
    auto mod = this->getOperation();

    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    Operation *op = getOperation();
    // Run the analysis to find the candidates for the transformation
    analyzeTransposeOps(op, sgLayoutMap);
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
          if (!op.getTileType().getWgMap())
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<xetile::LoadGatherOp>(
        [&](xetile::LoadGatherOp op) -> bool {
          if (!op.getTile().getType().getWgMap())
            return true;
          else
            return false;
        });

    target.addDynamicallyLegalOp<xetile::LoadGatherOp>(
        [&](xetile::LoadGatherOp op) -> bool {
          if (!op.getTile().getType().getWgMap())
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

    target.addDynamicallyLegalOp<xetile::StoreScatterOp>(
        [&](xetile::StoreScatterOp op) -> bool {
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

    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) -> bool {
          for (auto arg : op.getInitArgs()) {
            auto tileTy = dyn_cast<xetile::TileType>(arg.getType());
            auto vecTy =  dyn_cast<VectorType>(arg.getType());
            if (tileTy && tileTy.getWgMap())
              return false;
            if (vecTy && hasMap(arg.getDefiningOp()))
              return false;
          }
          return true;
        });

    target.addDynamicallyLegalOp<mlir::scf::YieldOp>(
      [&](mlir::scf::YieldOp op) -> bool {
        for (auto operand : op.getOperands()) {
            if (auto tileTy = mlir::dyn_cast<xetile::TileType>(operand.getType())) {
                if (tileTy.getWgMap()) {
                    return false;
                }
            }
            auto vecTy = mlir::dyn_cast<mlir::VectorType>(operand.getType());
            if (!vecTy) {
                continue;
            }
            if (auto defOp = operand.getDefiningOp()) {
                if (hasMap(defOp)) {
                    return false;
                }
            } else if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                mlir::Block *block = blockArg.getOwner();
                mlir::Operation *parentOp = block->getParentOp();
                if (!parentOp) {
                    continue;
                }
                if (auto parentFor = dyn_cast<mlir::scf::ForOp>(parentOp)) {
                    size_t argNum = blockArg.getArgNumber();
                    if (argNum < parentFor.getInitArgs().size()) {
                        mlir::Value initArg = parentFor.getInitArgs()[argNum];
                        if (auto initDefOp = initArg.getDefiningOp()) {
                            if (hasMap(initDefOp)) {
                                return false;
                            }
                        }
                    }
                } else if (auto parentIf = dyn_cast<mlir::scf::IfOp>(parentOp)) {
                    if (auto nestedParent = dyn_cast<mlir::scf::ForOp>(parentOp->getParentOp())) {
                        size_t argNum = blockArg.getArgNumber();
                        if (argNum < nestedParent.getNumRegionIterArgs()) {
                            mlir::Value initArg = nestedParent.getInitArgs()[argNum];
                            if (auto initDefOp = initArg.getDefiningOp()) {
                                if (hasMap(initDefOp)) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    });

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp, mlir::arith::AddFOp,
                                 mlir::math::ExpOp, mlir::math::SqrtOp, mlir::arith::ExtFOp,
                                 mlir::arith::ExtSIOp, mlir::arith::ExtUIOp, mlir::arith::FPToSIOp,
                                 mlir::arith::FPToUIOp, mlir::arith::UIToFPOp, mlir::arith::SIToFPOp,
                                 mlir::arith::TruncFOp, mlir::arith::TruncIOp, mlir::arith::CmpIOp,
                                 mlir::arith::CmpFOp,  mlir::arith::IndexCastUIOp, mlir::arith::SelectOp,
                                 mlir::math::FPowIOp,mlir::arith::IndexCastOp, mlir::arith::BitcastOp,
                                 mlir::vector::TransposeOp, mlir::vector::BroadcastOp,
                                 mlir::vector::MultiDimReductionOp,mlir::vector::ShapeCastOp,
                                 mlir::vector::CreateMaskOp>(
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
          if (op.getNumResults() == 0)
            return true;

          if (op.getThenRegion().empty() || op.getElseRegion().empty())
            return true;

          mlir::Block *thenBlock = &op.getThenRegion().front();
          mlir::Block *elseBlock = &op.getElseRegion().front();

          auto thenYield = llvm::dyn_cast<mlir::scf::YieldOp>(thenBlock->getTerminator());
          auto elseYield = llvm::dyn_cast<mlir::scf::YieldOp>(elseBlock->getTerminator());

          if (!thenYield || !elseYield)
            return true;

          if (thenYield.getNumOperands() != op.getNumResults() ||
              elseYield.getNumOperands() != op.getNumResults())
            return true;

          llvm::SmallVector<mlir::Value> thenVals(thenYield.getOperands());
          llvm::SmallVector<mlir::Value> elseVals(elseYield.getOperands());

          for (unsigned i = 0; i < op.getNumResults(); ++i) {
            mlir::Value thenVal = thenVals[i];
            mlir::Value elseVal = elseVals[i];

            // Check if either branch has a map attribute on its defining op
            xetile::WorkGroupMapAttr mapAttr = nullptr;
            if (auto defOp = thenVal.getDefiningOp()) {
              mapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(defOp->getAttr("map"));
            }
            if (!mapAttr) {
              if (auto defOp = elseVal.getDefiningOp()) {
                mapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(defOp->getAttr("map"));
              }
            }

            if (!mapAttr)
              return true;

            auto sgData = mapAttr.getSgData();
            llvm::SmallVector<int64_t> expectedShape = {sgData[0], sgData[1]};

            mlir::VectorType currentResultType = mlir::dyn_cast<mlir::VectorType>(op.getResult(i).getType());
            if (!currentResultType)
              return true;

            llvm::ArrayRef<int64_t> currentShape = currentResultType.getShape();
            if (currentShape.size() != expectedShape.size())
              return false;

            bool shapesMatch = true;
            for (size_t j = 0; j < currentShape.size(); ++j) {
              if (currentShape[j] != expectedShape[j]) {
                shapesMatch = false;
                break;
              }
            }
            if (!shapesMatch)
              return false;
          }

          return true;
        });

    target.addIllegalOp<xetile::ConvertLayoutOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    populateXeTileWgToSgPatterns(patterns, sgLayoutMap);
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

/// Create a pass
std::unique_ptr<Pass> createXeTileWgToSgPass() {
  return std::make_unique<XeTileWgToSgPass>();
}
} // namespace imex
