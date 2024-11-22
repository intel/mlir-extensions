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

#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h>
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h>

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


class WGToSGInitTileOpPattern : public XeOneToNConversion<xetile::InitTileOp> {
  using XeOneToNConversion<xetile::InitTileOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

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
        rewriter.create<mlir::index::FloorDivSOp>(loc, sgID, sgLayoutDimYConst);
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
    llvm::SmallVector<mlir::Type> newResultTypes;
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
      newResultTypes.push_back(newTileTy);
    }

    // Mapping for the result types.
    mlir::OneToNTypeMapping newMapping(op.getResult().getType());
    newMapping.addInputs(0, newResultTypes);
    rewriter.replaceOp(op, newInitTileOps, newMapping);
    return mlir::success();
  }
};

class WGToSGLoadTileOpPattern : public XeOneToNConversion<xetile::LoadTileOp> {
  using XeOneToNConversion<xetile::LoadTileOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

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

    mlir::OneToNTypeMapping newMapping(op.getResult().getType());
    newMapping.addInputs(0, newResultTypes);
    rewriter.replaceOp(op, newLoadOps, newMapping);
    return mlir::success();
  }
};

class WGToSGTileMMAOpPattern : public XeOneToNConversion<xetile::TileMMAOp> {
  using XeOneToNConversion<xetile::TileMMAOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

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

    mlir::OneToNTypeMapping newMapping(resultTy);
    newMapping.addInputs(0, newResultTypes);
    rewriter.replaceOp(op, newTileMMAOps, newMapping);
    return mlir::success();
  }
};

class WGToSGStoreTileOpPattern : public XeOneToNConversion<xetile::StoreTileOp> {
  using XeOneToNConversion<xetile::StoreTileOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

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

class WGToSGSCFForOpPattern : public XeOneToNConversion<mlir::scf::ForOp> {
  using XeOneToNConversion<mlir::scf::ForOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

    llvm::SmallVector<mlir::Value> convertedArgs;
    llvm::SmallVector<llvm::SmallVector<mlir::Type>> newResultTypes;
    mlir::OneToNTypeMapping newMapping(
        op.getResults().getTypes()); /// get the old types

    for (auto &&[i, values] : llvm::enumerate(adaptor.getInitArgs())) {
      llvm::SmallVector<mlir::Type> newTypes(values.getTypes().begin(),
                                             values.getTypes().end());
      convertedArgs.append(values.begin(), values.end());
      newMapping.addInputs(i, newTypes);
      newResultTypes.push_back(newTypes);
    }

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        convertedArgs);

    auto argTys = op.getRegion().getArgumentTypes();
    mlir::OneToNTypeMapping argumentMapping(argTys);
    for (auto [j, arg] : llvm::enumerate(op.getRegion().getArgumentTypes())) {
      if (j == 0)
        argumentMapping.addInputs(j, arg); // 0th is index (k)
      else
        argumentMapping.addInputs(
            j, newResultTypes[j - 1]); // get the new types from
                                       // adaptor.getInitArgs()
    }

    rewriter.applySignatureConversion(&op.getRegion().getBlocks().front(), argumentMapping);
    newOp.getBody()->erase();
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(op, newOp.getResults(), newMapping);
    return mlir::success();
  }
};

struct WGToSGSCFYieldOpPattern : public XeOneToNConversion<mlir::scf::YieldOp> {
  using XeOneToNConversion<mlir::scf::YieldOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  imex::XeOneToNPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> convertedResults;
    llvm::SmallVector<mlir::Type> newResultTypes;
    for (auto &values : adaptor.getResults())
      convertedResults.append(values.begin(), values.end());

    for (auto result : convertedResults) {
      newResultTypes.push_back(result.getType());
    }

    auto newOp =
        rewriter.create<mlir::scf::YieldOp>(op.getLoc(), convertedResults);

    rewriter.replaceOp(op, newOp.getResults());
    return mlir::success();
  }
};

class WGToSGUpdateTileOffsetOpPattern
    : public XeOneToNConversion<xetile::UpdateTileOffsetOp> {
  using XeOneToNConversion<xetile::UpdateTileOffsetOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    llvm::SmallVector<::mlir::Value> newUpdateTileOffsetOps;
    llvm::SmallVector<mlir::Type> newResultTypes;
    for (auto tile : adaptor.getTile()) {

      auto newUpdateTileOffsetOp = rewriter.create<xetile::UpdateTileOffsetOp>(
          op.getLoc(), tile.getType(), tile, op.getOffsetX(), op.getOffsetY(), op.getIndices());
      newUpdateTileOffsetOps.push_back(newUpdateTileOffsetOp);
      newResultTypes.push_back(tile.getType());
    }

    mlir::OneToNTypeMapping newMapping(op.getResult().getType());
    newMapping.addInputs(0, newResultTypes);
    rewriter.replaceOp(op, newUpdateTileOffsetOps, newMapping);
    return mlir::success();
  }
};

template <typename Op, int numOperands>
Op createOp(XeOneToNPatternRewriter &rewriter, mlir::Location loc,
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

template <typename Op, int numOperands>
class WGToSGElementWiseOpPattern : public XeOneToNConversion<Op> {
  using XeOneToNConversion<Op>::XeOneToNConversion;
  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OpAdaptor = typename Op::template GenericAdaptor<RangeT>;

  mlir::LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
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
    llvm::SmallVector<mlir::Type> newResultTypes;
    for (size_t i = 0; i < numOps; i++) {
      auto newOp = createOp<Op, numOperands>(rewriter, op.getLoc(), operand, i);
      newOp->getResult(0).setType(newTy);
      newOps.push_back(newOp);
      newResultTypes.push_back(newTy);
    }

    mlir::OneToNTypeMapping newMapping(op.getResult().getType());
    newMapping.addInputs(0, newResultTypes);
    rewriter.replaceOp(op, newOps, newMapping);
    return mlir::success();
  }
};

class WGToSGArithConstantOpPattern
    : public XeOneToNConversion<mlir::arith::ConstantOp> {
  using XeOneToNConversion<mlir::arith::ConstantOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    auto valueType = mlir::dyn_cast<mlir::VectorType>(value.getType());
    auto wgTileShape = valueType.getShape();

    if (!value || value.getType().getRank() != 2)
      return mlir::failure();

    auto mapAttr =
        llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(op->getAttr("map"));
    if (!mapAttr) {
      return mlir::failure();
    }

    auto sgData = mapAttr.getSgData();
    auto sgLayout = mapAttr.getSgLayout();
    auto newTy =
        mlir::VectorType::get({sgData[0], sgData[1]}, value.getElementType());

    llvm::SmallVector<mlir::Attribute> elems(
        value.value_begin<mlir::Attribute>(),
        value.value_end<mlir::Attribute>());

    llvm::SmallVector<mlir::Attribute> newValues;
    for (int64_t i = 0; i < static_cast<int64_t>(sgData[0]) * sgData[1]; i++) {
      newValues.push_back(elems[i]);
    }

    auto attr = mlir::DenseElementsAttr::get(newTy, newValues);

    size_t numOps;
    if (sgLayout[0] * sgData[0] == wgTileShape[0] &&
        sgLayout[1] * sgData[1] == wgTileShape[1])
      numOps = 1; // 1:1 mapping
    else
      numOps = (wgTileShape[0] / (sgLayout[0] * sgData[0])) +
               (wgTileShape[1] / (sgLayout[1] * sgData[1]));

    llvm::SmallVector<::mlir::Value> newOps;
    llvm::SmallVector<mlir::Type> newResultTypes;
    for (size_t i = 0; i < numOps; i++) {
      auto newOp = rewriter.create<arith::ConstantOp>(op.getLoc(), newTy, attr);
      newOps.push_back(newOp);
      newResultTypes.push_back(newTy);
    }

    mlir::OneToNTypeMapping newMapping(op.getResult().getType());
    newMapping.addInputs(0, newResultTypes);
    rewriter.replaceOp(op, newOps, newMapping);
    return mlir::success();
  }
};

// TODO: Templatize this pattern for similar elementwise ops
class WGToSGArithExtFOpPattern
    : public XeOneToNConversion<mlir::arith::ExtFOp> {
  using XeOneToNConversion<mlir::arith::ExtFOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

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

    auto newOp = rewriter.create<mlir::arith::ExtFOp>(op.getLoc(), newTy, adaptor.getOperands()[0]);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

class WGToSGArithTruncFOpPattern
    : public XeOneToNConversion<mlir::arith::TruncFOp> {
  using XeOneToNConversion<mlir::arith::TruncFOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

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

    auto newOp = rewriter.create<mlir::arith::TruncFOp>(op.getLoc(), newTy, adaptor.getOperands()[0]);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

class WGToSGVectorTranspose
    :public XeOneToNConversion<mlir::vector::TransposeOp> {
  using XeOneToNConversion<mlir::vector::TransposeOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::TransposeOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
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
              op.getLoc(), newTy, adaptor.getVector()[0], op.getPermutation());
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
    :public XeOneToNConversion<xetile::ConvertLayoutOp> {
  using XeOneToNConversion<xetile::ConvertLayoutOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::ConvertLayoutOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
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
    // TODO: Allocate slm as 1D array of i8, and then create the expected view on it.
    auto slmTy = mlir::MemRefType::get({resShape[0], resShape[1]}, elemTy, {}, 3);
    auto slm = rewriter.create<mlir::memref::AllocOp>(loc, slmTy);

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
        rewriter.create<mlir::index::FloorDivSOp>(loc, sgId, srcMapDimY);
    auto storeSgIdY =
        rewriter.create<mlir::index::RemUOp>(loc, sgId, srcMapDimY);

    // Store to SLM using src map
    auto memoryScopeAttr = mlir::IntegerAttr::get(rewriter.getIntegerType(32), 3);
    auto order = mlir::DenseI32ArrayAttr::get(op.getContext(), {1, 0});
    auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), nullptr /*sgMap*/, nullptr /*wgMap*/,
        order /*order*/, nullptr /*innerblocks*/, memoryScopeAttr /*memoryscope*/,
        nullptr /*scatterAttr*/);
    xetile::TileType srcTileTy =
      imex::xetile::TileType::get({srcMapSgData[0], srcMapSgData[1]}, elemTy, attr);

    auto storeOffsetX = rewriter.createOrFold<mlir::index::MulOp>(
                loc, storeSgIdX, createIndexConstant(indexType, srcMapSgData[0]));
    auto storeOffsetY = rewriter.createOrFold<mlir::index::MulOp>(
                loc, storeSgIdY, createIndexConstant(indexType, srcMapSgData[1]));
    auto storeInitTileOp = rewriter.create<xetile::InitTileOp>(
          loc, srcTileTy, slm, llvm::ArrayRef<mlir::OpFoldResult>({storeOffsetX, storeOffsetY}));
    //TODO: Set up cache attributes
    rewriter.create<xetile::StoreTileOp>(loc, adaptor.getSource()[0],
                                         storeInitTileOp, nullptr, nullptr, nullptr);

    // Add barrier
    rewriter.create<mlir::gpu::BarrierOp>(loc);

    // Load from SLM with result map
    xetile::TileType dstTileTy =
      imex::xetile::TileType::get({dstMapSgData[0], dstMapSgData[1]}, elemTy, attr);
    auto newResTy =
          mlir::VectorType::get({dstMapSgData[0], dstMapSgData[1]}, elemTy);

    auto dstMapDimY = createIndexConstant(indexType, dstSgLayout[1]);
    auto loadSgIdX = rewriter.create<mlir::index::FloorDivSOp>(loc, sgId, dstMapDimY);
    auto loadSgIdY =  rewriter.create<mlir::index::RemUOp>(loc, sgId, dstMapDimY);
    auto loadOffsetX = rewriter.createOrFold<mlir::index::MulOp>(
                loc, loadSgIdX, createIndexConstant(indexType, dstMapSgData[0]));
    auto loadOffsetY = rewriter.createOrFold<mlir::index::MulOp>(
                loc, loadSgIdY, createIndexConstant(indexType, dstMapSgData[1]));
    auto loadInitTileOp = rewriter.create<xetile::InitTileOp>(
          loc, dstTileTy, slm, llvm::ArrayRef<mlir::OpFoldResult>({loadOffsetX, loadOffsetY}));
    //TODO: Set up cache attributes
    auto loadTile = rewriter.create<xetile::LoadTileOp>(
          loc, newResTy, loadInitTileOp, mlir::Attribute(), nullptr, nullptr, nullptr);

    rewriter.replaceOp(op, loadTile);
    return mlir::success();
    }
  };

class WGToSGVectorBroadcast
    :public XeOneToNConversion<mlir::vector::BroadcastOp> {
  using XeOneToNConversion<mlir::vector::BroadcastOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::BroadcastOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    if (op.getVector().getType().getRank() != 2)
      return mlir::failure();

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());

    auto srcTy =  mlir::dyn_cast<mlir::VectorType>((adaptor.getSource()[0]).getType());
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
            op.getLoc(), newTy, adaptor.getSource()[0]);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};


class WGToSGPrefetchOpPattern : public XeOneToNConversion<xetile::PrefetchTileOp> {
  using XeOneToNConversion<xetile::PrefetchTileOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

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

// Helper function to analyze the def-use chain of initTileOps. Currently we
// pattern match the following def-use chain as a candidate for
// load + tranpose optimization.
// init_tile -> scf.for -> load_tile -> vector.transpose -> (pre-op) -> tile_mma
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

    auto consumerOp = *transposeOp->user_begin();

    // Check if vector.transpose is consumed by TileMMA directly or
    // is consumed by some pre-op and then TileMMA.
    if(!llvm::isa<imex::xetile::TileMMAOp>(consumerOp)){
      if(!OpTrait::hasElementwiseMappableTraits(consumerOp) &&
         !(llvm::isa<mlir::vector::BroadcastOp>(consumerOp))) {
        return mlir::WalkResult::skip();
      }
      else {
        if (!(consumerOp->hasOneUse() &&
              llvm::isa<imex::xetile::TileMMAOp>(*consumerOp->user_begin())))
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

void populateXeTileWgToSgPatterns(imex::XeOneToNTypeConverter &converter,
                                  mlir::RewritePatternSet &patterns,
                                  TileUsageAnalysis &analysis) {
  patterns.insert<WGToSGInitTileOpPattern, WGToSGLoadTileOpPattern,
                  WGToSGTileMMAOpPattern, WGToSGStoreTileOpPattern,
                  WGToSGSCFForOpPattern, WGToSGUpdateTileOffsetOpPattern,
                  WGToSGSCFYieldOpPattern, WGToSGVectorTranspose, WGToSGVectorBroadcast,
                  WGToSGXeTileConvertLayout, WGToSGPrefetchOpPattern, WGToSGArithExtFOpPattern,
                  WGToSGArithTruncFOpPattern>(patterns.getContext(), converter, analysis);
  patterns.insert<WGToSGElementWiseOpPattern<mlir::math::ExpOp, 1>,
                  WGToSGElementWiseOpPattern<mlir::arith::AddFOp, 2>,
                  WGToSGArithConstantOpPattern>(patterns.getContext(),
                                                converter, analysis);
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

    auto &analysis = getAnalysis<TileUsageAnalysis>();
    mlir::Operation *op = getOperation();
    // Run the analysis to find the candidates for the transformation
    analyzeInitTileOps(op);
    XeOneToNTypeConverter typeConverter(context);
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
          if(op.getInitArgs().empty())
            return true;
          for (auto arg : op.getInitArgs()) {
            auto tileTy = mlir::dyn_cast<xetile::TileType>(arg.getType());
            if (!tileTy)
              continue;
            else if (!tileTy.getWgMap())
              return true;
          }
          return false;
        });

    target.addDynamicallyLegalOp<mlir::scf::YieldOp>(
        [&](mlir::scf::YieldOp op) -> bool {
          // For cases with scf.if having hidden yield
          for (auto result: op.getResults()) {
            auto tileTy = mlir::dyn_cast<xetile::TileType>(result.getType());
            if (tileTy && tileTy.getWgMap())
              return false;
          }
          return true;
        });

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp, mlir::arith::AddFOp,
                                 mlir::math::ExpOp, mlir::arith::ExtFOp,
                                 mlir::arith::TruncFOp, mlir::vector::TransposeOp,
                                 mlir::vector::BroadcastOp>(
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

    populateXeTileWgToSgPatterns(typeConverter, patterns, analysis);
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
