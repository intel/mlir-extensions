//===- XeTileOps.cpp - XeTile dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeTile dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace imex {
namespace xetile {

template <typename AttrType>
static mlir::ParseResult parseAttributeHelper(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result,
                                              llvm::StringRef attrKeyword) {
  AttrType attr;
  mlir::Type ty;

  if (std::is_same<AttrType, mlir::Attribute>::value) {
    ty = mlir::Type{};
  } else if (std::is_same<AttrType, mlir::DenseI64ArrayAttr>::value) {
    ty = mlir::Type{};
  } else {
    assert(0 && "Unreachable.\n");
  }

  if (parser.parseCustomAttributeWithFallback(attr, ty))
    return mlir::failure();

  if (attr)
    result.addAttribute(attrKeyword, attr);
  return mlir::success();
}

static mlir::ParseResult
parseOptionalAttrDict(mlir::OpAsmParser &parser, mlir::OperationState &result,
                      llvm::ArrayRef<llvm::StringRef> allowedKeys) {

  // try to parse the left brace
  if (mlir::failed(parser.parseOptionalLBrace()))
    return mlir::success();

  auto parseElt = [&]() -> mlir::ParseResult {
    auto loc = parser.getCurrentLocation();
    llvm::StringRef nameId;
    if (parser.parseOptionalKeyword(&nameId, allowedKeys))
      return parser.emitError(loc, "invalid ")
             << "attribute keyword: " << nameId << ".\n";

    if (parser.parseEqual())
      return mlir::failure();

    if (nameId == "inner_blocks")
      return parseAttributeHelper<mlir::DenseI64ArrayAttr>(parser, result,
                                                           nameId);
    if (nameId == "padding") {
      return parseAttributeHelper<mlir::Attribute>(parser, result, nameId);
    }

    if (nameId == "wg_map_a") {
      return parseAttributeHelper<mlir::Attribute>(parser, result, nameId);
    }

    if (nameId == "wg_map_b") {
      return parseAttributeHelper<mlir::Attribute>(parser, result, nameId);
    }

    if (nameId == "wg_map_c") {
      return parseAttributeHelper<mlir::Attribute>(parser, result, nameId);
    }

    assert(0 && "Unreachable!");
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();

  if (parser.parseRBrace())
    return mlir::failure();

  return mlir::success();
}

static bool isColumnMajor(mlir::AffineMap layoutMap) {
  if (layoutMap.getNumDims() != 2 || layoutMap.getNumResults() != 2) {
    return false;
  }

  auto results = layoutMap.getResults();
  if (mlir::isa<mlir::AffineDimExpr>(results[0]) &&
      mlir::isa<mlir::AffineDimExpr>(results[1])) {
    auto dimExpr0 = mlir::cast<mlir::AffineDimExpr>(results[0]);
    auto dimExpr1 = mlir::cast<mlir::AffineDimExpr>(results[1]);
    return dimExpr0.getPosition() == 1 && dimExpr1.getPosition() == 0;
  }
  return false;
}

static bool isConstantIndex(mlir::Value value) {
  return value.getDefiningOp<mlir::arith::ConstantOp>() != nullptr;
}

static int64_t getConstantValue(mlir::Value value) {
  auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>();
  return mlir::cast<mlir::IntegerAttr>(constOp.getValue()).getInt();
}

mlir::LogicalResult InitTileOp::verify() {

  // number of offsets must be equal to the memref's rank.
  if (auto memRefType = mlir::dyn_cast<mlir::MemRefType>(getSourceType())) {
    if (!memRefType.hasRank())
      return emitOpError("unranked memref is not supported");
    if (static_cast<size_t>(memRefType.getRank()) != getStaticOffsets().size())
      return emitOpError("number of offsets must equal memref's rank");
  } else if (getStaticOffsets().size() != 2) {
    // number of offsets must be 2 because init_tile creates 2D tiles
    // dynamic_offsets is always a subset of offsets, so checking this is
    // sufficient
    return emitOpError("number of offsets must be 2");
  }

  // if the source is a memref and has static shape, then dynamic shape and
  // strides arguments must not be present
  if (isSourceMemRef() && sourceMemRefHasStaticShape() &&
      (hasDynamicStrides() || hasDynamicShape()))
    return emitOpError("dynamic shape or strides are not allowed with a static "
                       "shaped memref as source");

  // if the source is a memref with dynamic shape, then a 2D dynamic shape
  // argument must be present
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getDynamicShape().size() != 2)
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic shape argument missing or it is not 2D");

  // if the source is a memref with dynamic shape, then a 2D dynamic strides
  // argument must be present
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getDynamicStrides().size() != 2)
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic strides argument missing or it is not 2D");

  // if the source is an address, the dynamic shape must be 2D
  if (isSourceInteger() && getDynamicShape().size() != 2)
    return emitOpError("address is used as source but dynamic shape argument "
                       "is missing or it is not 2D");

  // if the source is an address, dynamic strides must be 2D
  if (isSourceInteger() && getDynamicStrides().size() != 2)
    return emitOpError("address is used as source but dynamic strides argument "
                       "is missing or it is not 2D");

  // Check for order attribute
  bool row_major = true;
  bool col_major = false;
  auto tileTy = getType();
  auto order = tileTy.getOrder();
  if (order[0] == 0 && order[1] == 1) {
    col_major = true;
    row_major = false;
  }

  if (isSourceMemRef() && sourceMemRefHasStaticShape()) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(getSourceType());

    // Checks for memrefs with format: memref<[shape], strided<[strides],
    // offsets:[offset]>>
    llvm::SmallVector<int64_t, 4> strides;
    auto shape = getSourceMemrefStaticShape();
    int64_t offset;
    if (mlir::succeeded(
            mlir::getStridesAndOffset(memrefType, strides, offset))) {
      int64_t rank = memrefType.getRank();
      if (row_major &&
          !((strides[rank - 2] == shape[rank - 1]) && (strides[rank - 1] == 1)))
        return emitOpError(
            "memref operand is expected to have a row-major layout");

      if (col_major &&
          !((strides[rank - 2] == 1) && (strides[rank - 1] == shape[rank - 2])))
        return emitOpError(
            "memref operand is expected to have a column-major layout");
      return mlir::success();
    }

    // Checks for memrefs with affine maps : memref<[shape], affine_map<(d0, d1)
    // -> (d1, d0)>>
    if (row_major && !(memrefType.getLayout().isIdentity())) {
      // No affine map means it's using the default row-major layout
      return emitOpError(
          "memref operand is expected to have a row-major layout");
    }

    if (col_major) {
      auto layoutAttr =
          mlir::dyn_cast<mlir::AffineMapAttr>(memrefType.getLayout());
      if (!layoutAttr) {
        return emitOpError("expected a valid affine map in the layout");
      }
      mlir::AffineMap layoutMap = layoutAttr.getValue();

      if (!isColumnMajor(layoutMap)) {
        return emitOpError(
            "memref operand is expected to have a column-major layout");
      }
    }
  } else if (isSourceInteger()) {
    auto dynamicShape = getDynamicShape();
    auto dynamicStrides = getDynamicStrides();

    if (dynamicShape.size() == 0 || dynamicStrides.size() == 0) {
      return emitOpError("dynamic shape and strides must not be empty");
    }

    // Check if all shape and stride values are constant.
    if (!llvm::all_of(dynamicShape, isConstantIndex) ||
        !llvm::all_of(dynamicStrides, isConstantIndex)) {
      llvm::dbgs() << "Assuming user has verified the layout\n";
      return mlir::success();
    }

    auto shapeDim1 = getConstantValue(dynamicShape[1]);
    auto strideDim0 = getConstantValue(dynamicStrides[0]);
    auto strideDim1 = getConstantValue(dynamicStrides[1]);

    // checks for layouts where source is not memref and just an address
    if (row_major && (strideDim0 == 1 && strideDim1 == shapeDim1)) {
      return emitOpError(
          "memref operand is expected to have a row-major layout");
    }

    if (col_major && !(strideDim0 == 1 && strideDim1 == shapeDim1)) {
      return emitOpError(
          "memref operand is expected to have a column-major layout");
    }
  } else if (isSourceMemRef() && !sourceMemRefHasStaticShape())
    llvm::dbgs() << "Assuming user has verified the layout\n";

  return mlir::success();
}

void InitTileOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       xetile::TileType resultType, mlir::Value source,
                       llvm::ArrayRef<mlir::OpFoldResult> offsets) {
  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, resultType, source, dynamicOffsets, staticOffsets,
        mlir::ValueRange({}), /* empty dynamic shape*/
        mlir::ValueRange({})  /* empty dynamic strides*/
  );
}

void InitTileOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       xetile::TileType resultType, mlir::Value source,
                       llvm::ArrayRef<mlir::OpFoldResult> offsets,
                       llvm::ArrayRef<mlir::Value> dynamic_shape,
                       llvm::ArrayRef<mlir::Value> dynamic_strides) {
  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, resultType, source, dynamicOffsets, staticOffsets,
        dynamic_shape, dynamic_strides);
}

mlir::ParseResult InitTileOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceRawOperands);
  llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  mlir::DenseI64ArrayAttr static_offsetsAttr;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4>
      dynamic_shapeOperands;
  llvm::SMLoc dynamic_shapeOperandsLoc;
  (void)dynamic_shapeOperandsLoc;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4>
      dynamic_stridesOperands;
  llvm::SMLoc dynamic_stridesOperandsLoc;
  (void)dynamic_stridesOperandsLoc;
  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceRawTypes);
  mlir::Type tileRawTypes[1];
  llvm::ArrayRef<mlir::Type> tileTypes(tileRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    auto odsResult =
        parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr);
    if (odsResult)
      return mlir::failure();
    result.getOrAddProperties<InitTileOp::Properties>().static_offsets =
        static_offsetsAttr;
  }
  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseLSquare())
      return mlir::failure();

    dynamic_shapeOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(dynamic_shapeOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return mlir::failure();
  }
  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseLSquare())
      return mlir::failure();

    dynamic_stridesOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(dynamic_stridesOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return mlir::failure();
  }

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(sourceRawTypes[0]))
    return mlir::failure();
  if (parser.parseArrow())
    return mlir::failure();

  if (parser.parseType(tileRawTypes[0]))
    return mlir::failure();
  llvm::copy(llvm::ArrayRef<int32_t>(
                 {1, static_cast<int32_t>(offsetsOperands.size()),
                  static_cast<int32_t>(dynamic_shapeOperands.size()),
                  static_cast<int32_t>(dynamic_stridesOperands.size())}),
             result.getOrAddProperties<InitTileOp::Properties>()
                 .operandSegmentSizes.begin());
  mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(tileTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0,
                             offsetsOperandsLoc, result.operands))
    return mlir::failure();
  if (parser.resolveOperands(dynamic_shapeOperands, odsBuildableType0,
                             dynamic_shapeOperandsLoc, result.operands))
    return mlir::failure();
  if (parser.resolveOperands(dynamic_stridesOperands, odsBuildableType0,
                             dynamic_stridesOperandsLoc, result.operands))
    return mlir::failure();
  return mlir::success();
}

void InitTileOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  printDynamicIndexList(printer, *this, getOffsets(), getStaticOffsetsAttr());
  if (!getDynamicShape().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getDynamicShape();
    printer << "]";
  }
  if (!getDynamicStrides().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getDynamicStrides();
    printer << "]";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getSource().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTile().getType();
}

mlir::ParseResult LoadTileOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {

  mlir::OpAsmParser::UnresolvedOperand sourceTile;
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceTile);
  llvm::SMLoc sourceTileOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceTile))
    return mlir::failure();

  // try to parse the optional dictionary attributes
  if (parseOptionalAttrDict(parser, result, {"padding"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  mlir::Type sourceType;
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceType);
  if (parser.parseType(sourceType))
    return mlir::failure();

  if (parser.parseArrow())
    return mlir::failure();

  mlir::Type valueType;
  llvm::ArrayRef<mlir::Type> outputValueTypes(valueType);
  if (parser.parseType(valueType))
    return mlir::failure();

  result.addTypes(outputValueTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceTileOperandLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

static void printPaddingValue(mlir::Attribute paddingValue,
                              mlir::OpAsmPrinter &printer) {
  if (auto floatVal = llvm::dyn_cast<mlir::FloatAttr>(paddingValue)) {
    printer << floatVal.getValue() << " : " << floatVal.getType();
  } else if (auto intVal = llvm::dyn_cast<mlir::IntegerAttr>(paddingValue)) {
    printer << intVal.getValue() << " : " << intVal.getType();
  }
}

void LoadTileOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  printer << " { ";
  printer << "padding = ";
  printPaddingValue(getPaddingValueOrDefault(), printer);
  printer << " } ";
  printer << " : ";
  printer << getSource().getType();
  printer << " -> ";
  printer << getValue().getType();
}

bool verifyInnerBlocksWithVecShape(mlir::DenseI64ArrayAttr &innerBlocks,
                                   llvm::ArrayRef<int64_t> &vecShape,
                                   llvm::ArrayRef<int64_t> &tileShape) {
  if (!(vecShape[2] == innerBlocks[0] && vecShape[3] == innerBlocks[1] &&
        ((tileShape[0] / innerBlocks[0]) == vecShape[0]) &&
        ((tileShape[1] / innerBlocks[1]) == vecShape[1])))
    return false;

  return true;
}

mlir::LogicalResult LoadTileOp::verify() {
  auto encoding = getSource().getType().getEncoding();
  auto tileShape = getSource().getType().getShape();
  if (!encoding)
    return mlir::success();

  auto tileAttr = mlir::dyn_cast<xetile::XeTileAttr>(encoding);
  auto innerBlocks = tileAttr.getInnerBlocks();

  // if inner_blocks is not present in the tile_attr, the output of the load
  // must be 2D
  if (innerBlocks == mlir::DenseI32ArrayAttr() &&
      getResult().getType().getShape().size() != 2)
    return emitOpError(
        "output must be a 2D vector if inner_blocks is not used in tile_attr.");

  if (innerBlocks != mlir::DenseI32ArrayAttr()) {
    auto vecShape = getResult().getType().getShape();
    // if inner_blocks is present in the tile_attr, the output of the load
    // must be 4D
    if (vecShape.size() != 4)
      return emitOpError(
          "output must be a 4D vector if inner_blocks is used in tile_attr.");
    // and, tile shape, output vector shape must be consistent with inner_blocks
    if (!verifyInnerBlocksWithVecShape(innerBlocks, vecShape, tileShape))
      return emitOpError(
          "shapes of the source tile, output value and inner_blocks must "
          "satisfy : "
          "valueShape[0] == tileShape[0]/innerBlocks[0] && valueShape[1] == "
          "tileShape[1]/innerBlocks[1] && "
          "valueShape[2] == innerBlocks[0] && valueShape[3] == "
          "innerBlocks[1].");
  }
  return mlir::success();
}

mlir::LogicalResult StoreTileOp::verify() {
  auto encoding = getTile().getType().getEncoding();
  if (!encoding)
    return mlir::success();

  auto tileAttr = mlir::dyn_cast<xetile::XeTileAttr>(encoding);
  auto innerBlocks = tileAttr.getInnerBlocks();
  auto tileShape = getTile().getType().getShape();

  // if inner_blocks is not present in the tile_attr, the stored value
  // must be 2D
  if (innerBlocks == mlir::DenseI32ArrayAttr() &&
      getValue().getType().getShape().size() != 2)
    return emitOpError(
        "value must be a 2D vector if inner_blocks is not used in tile_attr.");

  if (innerBlocks != mlir::DenseI32ArrayAttr()) {
    auto vecShape = getValue().getType().getShape();
    // if inner_blocks is present in the tile_attr, the stored value
    // must be 4D
    if (vecShape.size() != 4)
      return emitOpError(
          "value must be a 4D vector if inner_blocks is used in tile_attr.");
    // and, tile shape, input vector shape must be consistent with inner_blocks
    if (!verifyInnerBlocksWithVecShape(innerBlocks, vecShape, tileShape))
      return emitOpError(
          "shapes of the destination tile, value and inner_blocks must "
          "satisfy : "
          "valueShape[0] == tileShape[0]/innerBlocks[0] && valueShape[1] == "
          "tileShape[1]/innerBlocks[1] && "
          "valueShape[2] == innerBlocks[0] && valueShape[3] == "
          "innerBlocks[1].");
  }

  return mlir::success();
}

mlir::ParseResult TileMMAOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {

  mlir::OpAsmParser::UnresolvedOperand aRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> aOperands(aRawOperands);
  llvm::SMLoc aOperandsLoc;
  mlir::OpAsmParser::UnresolvedOperand bRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> bOperands(bRawOperands);
  llvm::SMLoc bOperandsLoc;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> cOperands;
  llvm::SMLoc cOperandsLoc;

  mlir::Type aRawTypes[1];
  llvm::ArrayRef<mlir::Type> aTypes(aRawTypes);
  mlir::Type bRawTypes[1];
  llvm::ArrayRef<mlir::Type> bTypes(bRawTypes);
  llvm::SmallVector<mlir::Type> cTypes;
  mlir::Type outputRawTypes[1];
  llvm::ArrayRef<mlir::Type> outputTypes(outputRawTypes);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperands[0]))
    return mlir::failure();

  // try to parse optional C vector
  if (mlir::succeeded(parser.parseOptionalComma())) {
    cOperandsLoc = parser.getCurrentLocation();
    mlir::OpAsmParser::UnresolvedOperand operand;
    mlir::OptionalParseResult parseResult =
        parser.parseOptionalOperand(operand);

    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return mlir::failure();
      cOperands.push_back(operand);
    }
  }

  // try to parse the optional dictionary attributes
  if (parseOptionalAttrDict(parser, result,
                            {"wg_map_a", "wg_map_b", "wg_map_c"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(aRawTypes[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  if (parser.parseType(bRawTypes[0]))
    return mlir::failure();

  if (mlir::succeeded(parser.parseOptionalComma())) {
    mlir::Type optionalType;
    mlir::OptionalParseResult parseResult =
        parser.parseOptionalType(optionalType);

    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return mlir::failure();
      cTypes.push_back(optionalType);
    }
  }

  if (parser.parseArrow())
    return mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return mlir::failure();

  result.addTypes(outputTypes);

  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return mlir::failure();

  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return mlir::failure();

  if (parser.resolveOperands(cOperands, cTypes, cOperandsLoc, result.operands))
    return mlir::failure();

  return mlir::success();
}

void TileMMAOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getA();
  printer << ", ";
  printer << getB();

  if (getC()) {
    printer << ", ";
    printer << getC();
  }

  if (getWgMapA()) {
    printer << " {wg_map_a =";
    printer << getWgMapA();
    printer << ", ";
    printer << "wg_map_b =";
    printer << getWgMapB();
  }

  if (getWgMapC()) {
    printer << ", ";
    printer << "wg_map_c =";
    printer << getWgMapC();
    printer << "}";
  } else if (getWgMapA()) {
    printer << "}";
  }

  printer << " : ";
  printer << getA().getType() << ", ";
  printer << getB().getType();
  if (getC()) {
    printer << ", ";
    printer << getC().getType();
  }
  printer << " -> ";
  printer << getOutput().getType();
}

mlir::LogicalResult TileMMAOp::verify() {
  int64_t aRank = getAType().getRank();
  int64_t bRank = getBType().getRank();

  mlir::Type aElemType = getAType().getElementType();
  mlir::Type bElemType = getBType().getElementType();
  mlir::Type outElemType = getOutput().getType().getElementType();

  auto aShape = getAType().getShape();
  auto bShape = getBType().getShape();
  auto outShape = getOutput().getType().getShape();

  // two vectors must have the same rank
  if (aRank != bRank)
    return emitOpError("A and B inputs must have the same rank.");

  // the two vector inputs must have the same element type
  if (aElemType != bElemType)
    return emitOpError("A and B inputs must have the same type.");

  if (getC() &&
      (llvm::cast<mlir::VectorType>(getC().getType()).getElementType() !=
       outElemType))
    return emitOpError("C and output vector must have the same type.");

  auto check4DMmaShapes = [](llvm::ArrayRef<int64_t> &A,
                             llvm::ArrayRef<int64_t> &B,
                             llvm::ArrayRef<int64_t> &Out) -> bool {
    return A[1] == B[0] && A[3] == B[2] && Out[0] == A[0] && Out[1] == B[1] &&
           Out[2] == A[2] && Out[3] == B[3];
  };

  auto check2DMmaShapes = [](llvm::ArrayRef<int64_t> &A,
                             llvm::ArrayRef<int64_t> &B,
                             llvm::ArrayRef<int64_t> &Out) -> bool {
    return A[1] == B[0] && Out[0] == A[0] && Out[1] == B[1];
  };

  // check mma shapes for 4D case
  if (aRank == 4 && !check4DMmaShapes(aShape, bShape, outShape))
    return emitOpError("incompatible A, B and output sizes for 4D tile mma op. "
                       "4D tile mma should have the shape (m x k x Bm x Bk) x "
                       "(k x n x Bk x Bn) = (m x n x Bm x Bn).");

  // check mma shape for 2D case
  if (aRank == 2 && !check2DMmaShapes(aShape, bShape, outShape))
    return emitOpError(
        "incompatible A, B and output sizes for 2D tile mma op. "
        "2D tile mma should have the shape (m x k) x (k x n) = (m x n).");

  // optional input C must have the same shape as output
  if (getC() &&
      llvm::cast<mlir::VectorType>(getC().getType()).getShape() != outShape)
    return emitOpError("input C must have the same shape as output.");

  return mlir::success();
}

mlir::ParseResult TilePackOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand in_vecRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> in_vecOperands(
      in_vecRawOperands);
  llvm::SMLoc in_vecOperandsLoc;
  (void)in_vecOperandsLoc;
  mlir::Type in_vecRawTypes[1];
  llvm::ArrayRef<mlir::Type> in_vecTypes(in_vecRawTypes);
  mlir::Type out_vecRawTypes[1];
  llvm::ArrayRef<mlir::Type> out_vecTypes(out_vecRawTypes);

  in_vecOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(in_vecRawOperands[0]))
    return mlir::failure();
  // try to parse the optional dictionary attributes
  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parseOptionalAttrDict(parser, result, {"inner_blocks"}))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(in_vecRawTypes[0]))
    return mlir::failure();
  if (parser.parseArrow())
    return mlir::failure();

  if (parser.parseType(out_vecRawTypes[0]))
    return mlir::failure();
  result.addTypes(out_vecTypes);
  if (parser.resolveOperands(in_vecOperands, in_vecTypes, in_vecOperandsLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void TilePackOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getInVec();
  printer << " { ";
  printer << "inner_blocks = ";
  getInnerBlocksAttr().print(printer);
  printer << " } ";
  printer << ' ' << ":";
  printer << ' ';
  printer << getInVec().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getOutVec().getType();
}

mlir::LogicalResult TilePackOp::verify() {
  auto inVecShape = getInVec().getType().getShape();
  auto outVecShape = getOutVec().getType().getShape();
  auto innerBlocks = getInnerBlocks();
  auto inElemTy = getInVec().getType().getElementType();
  auto outElemTy = getOutVec().getType().getElementType();

  // input and output vector element types must match
  if (inElemTy != outElemTy)
    return emitOpError("input and output vector element type mismatch.");

  // innermost 2 dimensions of the output vector must satisfy:
  //    outVecShape[2] == innerBlocks[0]
  //    outVecShape[3] == innerBlocks[1]
  if (!(outVecShape[2] == innerBlocks[0] && outVecShape[3] == innerBlocks[1]))
    return emitOpError(
        "innermost 2 dimensions of output vector must satisfy : "
        "outVecShape[2] == innerBlocks[0] && outVecShape[3] == innerBlocks[1]");

  // outermost 2 dimensions of the output vector must satisfy:
  //    outVecShape[0] == inVecShape[0]/innerBlocks[0]
  //    outVecShape[1] == inVecShape[1]/innerBlocks[1]
  if (!(outVecShape[0] == inVecShape[0] / innerBlocks[0] &&
        outVecShape[1] == inVecShape[1] / innerBlocks[1]))
    return emitOpError(
        "outermost 2 dimensions of the output vector must satisfy : "
        "outVecShape[0] == inVecShape[0]/innerBlocks[0] && "
        "outVecShape[1] == inVecShape[1]/innerBlocks[1]");

  return mlir::success();
}

mlir::OpFoldResult TilePackOp::fold(FoldAdaptor /*adaptor*/) {
  mlir::Value in = this->getInVec();
  if (auto unpack = in.getDefiningOp<TileUnpackOp>()) {
    mlir::Value src = unpack.getInVec();
    if (src.getType() != this->getType() ||
        unpack.getInnerBlocks() != this->getInnerBlocks())
      return nullptr;

    return src;
  }
  return nullptr;
}

mlir::ParseResult TileUnpackOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand in_vecRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> in_vecOperands(
      in_vecRawOperands);
  llvm::SMLoc in_vecOperandsLoc;
  (void)in_vecOperandsLoc;
  mlir::Type in_vecRawTypes[1];
  llvm::ArrayRef<mlir::Type> in_vecTypes(in_vecRawTypes);
  mlir::Type out_vecRawTypes[1];
  llvm::ArrayRef<mlir::Type> out_vecTypes(out_vecRawTypes);

  in_vecOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(in_vecRawOperands[0]))
    return mlir::failure();
  // try to parse the optional dictionary attributes
  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parseOptionalAttrDict(parser, result, {"inner_blocks"}))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(in_vecRawTypes[0]))
    return mlir::failure();
  if (parser.parseArrow())
    return mlir::failure();

  if (parser.parseType(out_vecRawTypes[0]))
    return mlir::failure();
  result.addTypes(out_vecTypes);
  if (parser.resolveOperands(in_vecOperands, in_vecTypes, in_vecOperandsLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void TileUnpackOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getInVec();
  printer << " { ";
  printer << "inner_blocks = ";
  getInnerBlocksAttr().print(printer);
  printer << " } ";
  printer << ' ' << ":";
  printer << ' ';
  printer << getInVec().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getOutVec().getType();
}

mlir::LogicalResult TileUnpackOp::verify() {
  auto inVecShape = getInVec().getType().getShape();
  auto outVecShape = getOutVec().getType().getShape();
  auto innerBlocks = getInnerBlocks();
  auto inElemTy = getInVec().getType().getElementType();
  auto outElemTy = getOutVec().getType().getElementType();

  // input and output vector element types must match
  if (inElemTy != outElemTy)
    return emitOpError("input and output vector element type mismatch.");

  // innermost 2 dimensions of the input vector must satisfy
  //    outVecShape[2] == innerBlocks[0]
  //    outVecShape[3] == innerBlocks[1]
  if (!(inVecShape[2] == innerBlocks[0] && inVecShape[3] == innerBlocks[1]))
    return emitOpError(
        "innermost 2 dimensions of the input vector must satisfy : "
        "inVecShape[2] == innerBlocks[0] && "
        "inVecShape[3] == innerBlocks[1]");

  // output vector must satisfy :
  //     outVecShape[0] == inVecShape[0] * innerBlocks[0]
  //     outVecShape[1] == inVecShape[1] * innerBlocks[1] &&
  if (!(outVecShape[0] == inVecShape[0] * innerBlocks[0] &&
        outVecShape[1] == inVecShape[1] * innerBlocks[1]))
    return emitOpError("output vector must satisfy : "
                       "outVecShape[0] == inVecShape[0] * innerBlocks[0] && "
                       "outVecShape[1] == inVecShape[1] * innerBlocks[1]");

  return mlir::success();
}

mlir::OpFoldResult TileUnpackOp::fold(FoldAdaptor /*adaptor*/) {
  mlir::Value in = this->getInVec();
  if (auto pack = in.getDefiningOp<TilePackOp>()) {
    mlir::Value src = pack.getInVec();
    if (src.getType() != this->getType() ||
        pack.getInnerBlocks() != this->getInnerBlocks())
      return nullptr;

    return src;
  }
  return nullptr;
}

mlir::LogicalResult TransposeOp::verify() {
  auto srcShape = getVector().getType().getShape();
  auto resShape = getResult().getType().getShape();
  auto permutation = getPermutation();

  auto transposeShape = srcShape.vec();
  for (auto [i, j] : llvm::enumerate(permutation)) {
    if (j >= (int64_t)srcShape.size())
      return emitOpError("permutation index out of bounds");
    transposeShape[i] = srcShape[j];
  }

  if (transposeShape != resShape.vec())
    return emitOpError("Incorrect transpose permutation");

  return mlir::success();
}

mlir::LogicalResult ReduceOp::verify() {
  auto dims = getReductionDim();
  auto resShape = getResult().getType().getShape();
  for (auto i : dims)
    if (resShape[i] != 1)
      return emitOpError("reduction dimension of result must have size 1");
  return mlir::success();
}

mlir::LogicalResult BroadcastOp::verify() {
  auto dims = getBroadcastDim();
  auto srcShape = getSource().getType().getShape();
  for (auto i : dims)
    if (srcShape[i] != 1)
      return emitOpError("broadcast dimension of source must have size 1");
  return mlir::success();
}

} // namespace xetile
} // namespace imex

#include <imex/Dialect/XeTile/IR/XeTileOpsEnums.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOps.cpp.inc>
