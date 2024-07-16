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
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <optional>

#include "imex/Dialect/XeTile/Transforms/Blocking.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/XeArch.h"
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h>
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h>

#include "PassDetail.h"
#include <iostream>

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
    auto sgID = rewriter.create<mlir::gpu::SubgroupIdOp>(loc);
    auto indexType = rewriter.getIndexType();
    auto sgLayoutDimYConst = createIndexConstant(indexType, sgLayout[1]);
    auto sgDataDimXConst = createIndexConstant(indexType, sgTileShape[0]);
    auto sgDataDimYConst = createIndexConstant(indexType, sgTileShape[1]);

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

    calculateGlobalOffsets(globalOffsetsY, wgTileShape[0], sgTileShape[0],
                           sgLayout[0], sgDataDimXConst, sgIdY, offsets[0]);
    calculateGlobalOffsets(globalOffsetsX, wgTileShape[1], sgTileShape[1],
                           sgLayout[1], sgDataDimYConst, sgIdX, offsets[1]);

    // TODO: check for how to broadcast
    for (auto y : globalOffsetsY) {
      for (auto x : globalOffsetsX) {
        offsetPermutations.push_back({y, x});
      }
    }

    mlir::SmallVector<mlir::Value> newInitTileOps;
    llvm::SmallVector<mlir::Type> newResultTypes;
    for (size_t i = 0; i < offsetPermutations.size(); i++) {
      auto newInitTileOp = rewriter.create<xetile::InitTileOp>(
          loc, newTileTy, source, offsetPermutations[i]);
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
          op.getLoc(), newResTy, src, op.getPaddingAttr());
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
                                           newDstTiles[i]);
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

    rewriter.applySignatureConversion(&op.getRegion(), argumentMapping);
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
          op.getLoc(), tile.getType(), tile, op.getOffsetX(), op.getOffsetY());
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
    for (int64_t i = 0; i < sgData[0] * sgData[1]; i++) {
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

void populateXeTileWgToSgPatterns(imex::XeOneToNTypeConverter &converter,
                                  mlir::RewritePatternSet &patterns,
                                  TileUsageAnalysis &analysis) {
  patterns.insert<WGToSGInitTileOpPattern, WGToSGLoadTileOpPattern,
                  WGToSGTileMMAOpPattern, WGToSGStoreTileOpPattern,
                  WGToSGSCFForOpPattern, WGToSGUpdateTileOffsetOpPattern,
                  WGToSGSCFYieldOpPattern>(patterns.getContext(), converter,
                                           analysis);
  patterns.insert<WGToSGElementWiseOpPattern<mlir::math::ExpOp, 1>,
                  WGToSGElementWiseOpPattern<mlir::arith::AddFOp, 2>,
                  WGToSGArithConstantOpPattern>(patterns.getContext(),
                                                converter, analysis);
}

// Transforms WG XeTile IR to SG XeTile
class XeTileWgToSgPass
    : public imex::impl::XeTileWgToSgBase<imex::XeTileWgToSgPass> {

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
          for (auto result : op.getResults()) {
            auto tileTy = mlir::dyn_cast<xetile::TileType>(result.getType());
            if (!tileTy)
              continue;
            else if (!tileTy.getWgMap())
              return true;
          }
          return false;
        });

    target.addDynamicallyLegalOp<mlir::arith::ConstantOp, mlir::arith::AddFOp,
                                 mlir::math::ExpOp>(
        [&](mlir::Operation *op) -> bool {
          auto mapAttr = llvm::dyn_cast_or_null<xetile::WorkGroupMapAttr>(
              op->getAttr("map"));
          if (!mapAttr)
            return true;
          else
            return false;
        });

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
