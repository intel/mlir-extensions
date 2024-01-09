//===- XeGPUOps.cpp - XeGPU dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeGPU dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/XeGPU/IR/XeGPU.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeUtilities.h>
#include <numeric>
#include <type_traits>

#include "imex/Utils/DebugUtils.h"

#define DEBUG_TYPE "xegpu"

namespace imex {
namespace xegpu {

const int MAX_2D_BLOCK_WIDTH_IN_ELEMENTS = 64;
const int MIN_2D_BLOCK_WIDTH_IN_ELEMENTS = 1;
const int MAX_2D_BLOCK_HEIGHT_IN_ELEMENTS = 32;
const int MIN_2D_BLOCK_HEIGHT_IN_ELEMENTS = 1;
// TODO: Generalize shapes for different architecture.
const int MAX_TM_SIZE = 8;
const int TN_SIZE = 16;
const int TK_SIZE_FOR_D16 = 16;
const int TK_SIZE_FOR_D8 = 32;

static size_t getRankOf(mlir::Value value) {
  if (value.getType().isIntOrIndexOrFloat())
    return 0;
  if (auto ty = llvm::dyn_cast_if_present<mlir::MemRefType>(value.getType()))
    return ty.getRank();
  if (auto ty = llvm::dyn_cast_if_present<mlir::VectorType>(value.getType()))
    return ty.getRank();
  llvm_unreachable("Unsupported value for getRankOf");
}

static void transpose(llvm::ArrayRef<int64_t> trans,
                      std::vector<int64_t> &shape) {
  std::vector<int64_t> old = shape;
  for (size_t i = 0; i < trans.size(); i++)
    shape[i] = old[trans[i]];
};

bool dpasSupportedTypes(mlir::Type type, bool isResult) {
  if (isResult) {
    if (type.isF32() || type.isInteger(32))
      return true;
    else
      return false;
  } else {
    if (type.isF16() || type.isBF16() || type.isInteger(16) ||
        type.isInteger(8))
      return true;
    else
      return false;
  }
}

extern bool printDefaultValues();

template <typename CustomEnum, typename CustomEnumAttr>
static mlir::ParseResult parseCustomEnumAttr(mlir::OpAsmParser &parser,
                                             mlir::OperationState &result,
                                             llvm::StringRef attrKeyword) {
  auto loc = parser.getCurrentLocation();
  auto attrOptional = mlir::FieldParser<CustomEnum, CustomEnum>::parse(parser);
  if (mlir::failed(attrOptional))
    return parser.emitError(loc, "invalid ") << "attribute specification";
  auto attr =
      CustomEnumAttr::get(parser.getBuilder().getContext(), *attrOptional);
  result.addAttribute(attrKeyword, attr);
  return mlir::success();
}

template <typename AttrType>
static mlir::ParseResult parseBoolAndIntegerAttr(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result,
                                                 llvm::StringRef attrKeyword) {
  AttrType attr;
  mlir::Type ty;

  if (std::is_same<AttrType, mlir::BoolAttr>::value) {
    ty = parser.getBuilder().getIntegerType(1);

  } else if (std::is_same<AttrType, mlir::IntegerAttr>::value) {
    ty = parser.getBuilder().getIntegerType(32);
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
};

/// @brief Parsing optional attribute list which are enclosed in braces "{}",
/// and seperated by comma
/// @param parser
/// @param result
/// @param allowedKeywords
/// @return
static mlir::ParseResult
parseOptionalAttrDict(mlir::OpAsmParser &parser, mlir::OperationState &result,
                      llvm::ArrayRef<llvm::StringRef> allowedKeywords,
                      bool isWrite = false) {
  // no optional attributes, return success
  if (mlir::failed(parser.parseOptionalLBrace()))
    return mlir::success();

  auto parseElt = [&]() -> mlir::ParseResult {
    auto loc = parser.getCurrentLocation();
    llvm::StringRef nameId;
    if (parser.parseOptionalKeyword(&nameId, allowedKeywords))
      return parser.emitError(loc, "invalid")
             << "attribute keyword: " << nameId << ".\n";

    if (parser.parseEqual())
      return mlir::failure();

    if (nameId == "l1_hint" || nameId == "l2_hint" || nameId == "l3_hint") {
      if (isWrite)
        return parseCustomEnumAttr<CacheWriteHint, CacheWriteHintAttr>(
            parser, result, nameId);
      else
        return parseCustomEnumAttr<CacheReadHint, CacheReadHintAttr>(
            parser, result, nameId);
    }

    if (nameId == "mode") {
      return parseCustomEnumAttr<Mode, ModeAttr>(parser, result, nameId);
    }

    if (nameId == "chunk_size_per_lane" || nameId == "vnni_axis")
      return parseBoolAndIntegerAttr<mlir::IntegerAttr>(parser, result, nameId);

    if (nameId == "boundary_check")
      return parseBoolAndIntegerAttr<mlir::BoolAttr>(parser, result, nameId);

    if (nameId == "transpose")
      return parseBoolAndIntegerAttr<mlir::DenseI64ArrayAttr>(parser, result,
                                                              nameId);

    assert(0 && "Unreachable!");
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();

  return parser.parseRBrace();
}

template <typename T>
static void printCacheHintAttrs(mlir::OpAsmPrinter &printer, T op,
                                bool printSep) {
  if (op.getL1HintAttr()) {
    if (printSep)
      printer << ", ";
    printer << "l1_hint = " << op.getL1Hint().value();
    printSep = true;
  }

  if (op.getL2HintAttr()) {
    if (printSep)
      printer << ", ";
    printer << "l2_hint = " << op.getL2Hint().value();
    printSep = true;
  }

  if (op.getL3HintAttr()) {
    if (printSep)
      printer << ", ";
    printer << "l3_hint = " << op.getL3Hint().value();
  }
}

static bool verifyAndInferShape(std::vector<int64_t> &shape,
                                imex::xegpu::SubGroupMapAttr sgMap) {
  if (sgMap) {
    auto wiLayout = sgMap.getWiLayout();
    auto wiData = sgMap.getWiData();

    if ((int64_t)shape.size() != wiData.size() ||
        (int64_t)shape.size() != wiLayout.size()) {
      return false;
    }

    for (size_t i = 0; i < shape.size(); i++) {

      if ((shape[i] % (wiLayout[i] * wiData[i]) != 0 &&
           (wiLayout[i] * wiData[i]) % shape[i] != 0) ||
          shape[i] % wiLayout[i] != 0 || shape[i] % wiData[i] != 0) {
        return false;
      }
      shape[i] /= wiLayout[i];
    }
  }

  return true;
}

/// @brief the base builder for CreateNdDescOp
/// @param builder, the mlir OpBuilder
/// @param state , the mlir OperationState
/// @param TensorDesc, the TensorDescType of the result
/// @param source, the base address of the data. It can be either 2D memref
/// object or simple integer value (pointer)
/// @param offsets, the dynamic offset given as mlir::Value
/// @param shape, the dynamic shape given as array of mlir::Values
/// @param strides, the dynamic shape given as array of mlir::Values
/// @param static_offsets, the static offset. If it is not used it should be
/// filled with mlir::ShapeType::kDynamic
/// @param mode, VC or SIMT
void CreateNdDescOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Type TensorDesc,
                           mlir::Value source, mlir::ValueRange offsets,
                           mlir::ValueRange shape, mlir::ValueRange strides,
                           llvm::ArrayRef<int64_t> static_offsets,
                           imex::xegpu::Mode mode) {
  auto offsetRank = static_offsets.size();
  auto shapeRank = shape.size() ? shape.size() : getRankOf(source);

  size_t dynOffsetRank =
      std::count_if(static_offsets.begin(), static_offsets.end(),
                    [](int64_t d) { return mlir::ShapedType::isDynamic(d); });

  // shape and strides should exists at the same time
  // and the final rank for shape and offset (dynamic + static)
  // should be the same
  assert(shape.size() == strides.size() && shapeRank == offsetRank &&
         offsets.size() == dynOffsetRank);

  state.addOperands(source);
  state.addOperands(offsets);
  state.addOperands(shape);
  state.addOperands(strides);
  state.addAttribute(
      getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()),
                                    static_cast<int32_t>(shape.size()),
                                    static_cast<int32_t>(strides.size())}));
  state.addAttribute(getStaticOffsetsAttrName(state.name),
                     builder.getDenseI64ArrayAttr(static_offsets));
  state.addAttribute(getModeAttrName(state.name),
                     xegpu::ModeAttr::get(builder.getContext(), mode));
  state.addTypes(TensorDesc);
}

void CreateNdDescOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Type tdesc,
                           mlir::Value source,
                           llvm::ArrayRef<mlir::OpFoldResult> offsets,
                           imex::xegpu::Mode mode) {
  auto ty = llvm::dyn_cast_if_present<mlir::MemRefType>(source.getType());
  assert(ty && ty.hasStaticShape() && offsets.size() == getRankOf(source));

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        mlir::ValueRange({}) /* empty dynamic shape */,
        mlir::ValueRange({}) /* empty dynamic strides */,
        staticOffsets /* static offsets */, mode);
}

void CreateNdDescOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Type tdesc,
                           mlir::Value source,
                           llvm::ArrayRef<mlir::OpFoldResult> offsets,
                           mlir::ValueRange shape, mlir::ValueRange stride,
                           xegpu::Mode mode) {
  assert(shape.size() && offsets.size() && stride.size() &&
         shape.size() == stride.size() && shape.size() == offsets.size());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<mlir::Value> dynamicOffsets;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        shape /* dynamic shape */, stride /* dynamic strides */,
        staticOffsets /* static offsets */, mode);
}

mlir::ParseResult CreateNdDescOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  // parse the source operand
  mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceRawOperands);
  llvm::SMLoc sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return mlir::failure();

  // parse the offset operand, in format of [x, y]
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  mlir::DenseI64ArrayAttr static_offsetsAttr;
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
    return mlir::failure();
  result.addAttribute("static_offsets", static_offsetsAttr);

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> shapeOperands;
  llvm::SMLoc shapeOperandsLoc;

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  llvm::SMLoc stridesOperandsLoc;
  // parse optional shape and strides, shape and strides should always come
  // together
  if (mlir::succeeded(parser.parseOptionalComma())) {
    // parse shape part, in form of [x, y]
    if (parser.parseLSquare())
      return mlir::failure();
    shapeOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(shapeOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return mlir::failure();

    if (parser.parseComma())
      return mlir::failure();

    // parse stride part, in form of [x, y]
    if (parser.parseLSquare())
      return mlir::failure();
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(stridesOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return mlir::failure();
  }

  if (parseOptionalAttrDict(parser, result, {"boundary_check", "mode"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseType(sourceRawTypes[0]))
    return mlir::failure();

  if (parser.parseArrow())
    return mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(offsetsOperands.size()),
                           static_cast<int32_t>(shapeOperands.size()),
                           static_cast<int32_t>(stridesOperands.size())}));

  mlir::Type indexType = parser.getBuilder().getIndexType();
  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return mlir::failure();
  if (parser.resolveOperands(offsetsOperands, indexType, offsetsOperandsLoc,
                             result.operands))
    return mlir::failure();
  if (parser.resolveOperands(shapeOperands, indexType, shapeOperandsLoc,
                             result.operands))
    return mlir::failure();
  if (parser.resolveOperands(stridesOperands, indexType, stridesOperandsLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void CreateNdDescOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getSource();
  printDynamicIndexList(printer, *this, getDynamicOffsets(),
                        getStaticOffsetsAttr());
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

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << ' ' << "{";
    printer << "mode = " << mode;
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getSourceType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTensorDescType();
}

mlir::LogicalResult CreateNdDescOp::verify() {
  auto mode = getMode();
  auto isScattered = getTensorDescType().getScattered();
  auto mapping = getTensorDescType().getMapping();

  if (isScattered) {
    return emitOpError("Encoding Attribute of TensorDesc is not expected for "
                       "non-scattered operators.\n");
  }

  if (mode == imex::xegpu::Mode::VC && mapping) {
    return emitOpError("Mapping attribute of TensorDesc is not expected "
                       "for VC mode operations.\n");
  }

  if (mode == imex::xegpu::Mode::SIMT && !mapping) {
    return emitOpError("Expecting SgMap attribute for SIMT mode operators.\n");
  }

  auto offsetRank = getOffsets().size();
  auto shapeRank = getShape().size();
  auto stridesRank = getStrides().size();
  auto baseRank = getRankOf(getSource()) ? getRankOf(getSource()) : 2;

  if (offsetRank != shapeRank || shapeRank != stridesRank ||
      shapeRank != baseRank)
    return emitOpError(
        "Expecting the rank of shape, strides, offsets and memref type "
        "should match with each other (they currently should be 2D).");

  return mlir::success();
}

xegpu::TensorDescType CreateNdDescOp::getTensorDescType() {
  return getTensorDesc().getType();
}

llvm::SmallVector<mlir::OpFoldResult> CreateNdDescOp::getOffsets() {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  auto dynamicOffsets = getDynamicOffsets(); // given by dynamic_offsets
                                             // variable
  auto staticOffsets = getStaticOffsets(); // given by static_offsets attribute

  // in case static_offsets is missing
  if (staticOffsets.size() == 0) {
    offsets.assign(dynamicOffsets.begin(), dynamicOffsets.end());
    return offsets;
  }

  for (size_t i = 0, j = 0; i < staticOffsets.size(); i++) {
    if (mlir::ShapedType::isDynamic(staticOffsets[i])) {
      assert(j < dynamicOffsets.size());
      offsets.push_back(dynamicOffsets[j++]);
    } else {
      auto ty = mlir::IndexType::get(getContext());
      auto attr = mlir::IntegerAttr::get(ty, staticOffsets[i]);
      offsets.push_back(attr);
    }
  }
  return offsets;
}

llvm::ArrayRef<int64_t> CreateNdDescOp::getStaticShape() {
  auto rank = getTensorDescType().getRank();
  static llvm::SmallVector<int64_t> dyn(rank, mlir::ShapedType::kDynamic);
  auto srcTy = llvm::dyn_cast_if_present<mlir::MemRefType>(getSourceType());
  if (srcTy)
    return srcTy.getShape();

  return dyn;
}

llvm::SmallVector<mlir::OpFoldResult> CreateNdDescOp::getShape() {
  llvm::SmallVector<mlir::OpFoldResult> shape;
  auto dynShape = getDynamicShape();
  if (dynShape.size()) {
    shape.append(dynShape.begin(), dynShape.end());
    return shape;
  }

  auto ty = llvm::dyn_cast_if_present<mlir::MemRefType>(getSourceType());
  if (ty && ty.hasStaticShape()) {
    for (auto dim : ty.getShape()) {
      auto attr =
          mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), dim);
      shape.push_back(attr);
    }
    return shape;
  }

  emitOpError("The shape information is missing.");
  llvm_unreachable("Unexpected error in CreateNdDescOp.\n");
}

llvm::ArrayRef<int64_t> CreateNdDescOp::getStaticStrides() {
  auto rank = getTensorDescType().getRank();
  static llvm::SmallVector<int64_t> dyn(rank, mlir::ShapedType::kDynamic);
  auto srcTy = llvm::dyn_cast_if_present<mlir::MemRefType>(getSourceType());
  if (srcTy) {
    auto [strides, offset] = mlir::getStridesAndOffset(srcTy);
    return strides;
  }
  return dyn;
}

llvm::SmallVector<mlir::OpFoldResult> CreateNdDescOp::getStrides() {
  llvm::SmallVector<mlir::OpFoldResult> strides;

  auto dynStrides = getDynamicStrides();
  if (dynStrides.size()) {
    strides.append(dynStrides.begin(), dynStrides.end());
    return strides;
  }

  auto ty = llvm::dyn_cast_if_present<mlir::MemRefType>(getSourceType());
  if (ty && ty.hasStaticShape()) {
    auto [staticStrides, offset] = mlir::getStridesAndOffset(ty);
    for (auto dim : staticStrides) {
      auto attr =
          mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), dim);
      strides.push_back(attr);
    }
    return strides;
  }
  emitOpError("The strides information is missing.");
  llvm_unreachable("Unexpected error in CreateNdDescOp.\n");
}

/// Return the element type of the TensorDesc
mlir::Type CreateNdDescOp::getElementType() {
  return getTensorDescType().getElementType();
}

/// Return the shape of the TensorDesc
llvm::ArrayRef<int64_t> CreateNdDescOp::getTensorDescShape() {
  return getTensorDescType().getShape();
}

mlir::ParseResult CreateDescOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceRawOperands);
  llvm::SMLoc sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  mlir::OpAsmParser::UnresolvedOperand offsetsRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> offsetsOperands(
      offsetsRawOperands);
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(offsetsRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result, {"chunk_size_per_lane", "mode"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseType(sourceRawTypes[0]))
    return mlir::failure();
  if (parser.parseComma())
    return mlir::failure();

  mlir::Type offsetsRawTypes[1];
  llvm::ArrayRef<mlir::Type> offsetsTypes(offsetsRawTypes);
  if (parser.parseType(offsetsRawTypes[0]))
    return mlir::failure();
  if (parser.parseArrow())
    return mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();

  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return mlir::failure();
  if (parser.resolveOperands(offsetsOperands, offsetsTypes, offsetsOperandsLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void CreateDescOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  bool printSep = false;
  auto chunk = getChunkSizePerLane();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getSource();
  printer << ",";
  printer << ' ';
  printer << getOffsets();

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || chunk != 1) {
    printer << ' ' << "{";
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << "mode = " << mode;
    printSep = true;
  }

  if (printDefaults || chunk != 1) {
    if (printSep)
      printer << "," << ' ';
    printer << "chunk_size_per_lane = " << chunk;
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || chunk != 1) {
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getSource().getType();
  printer << ",";
  printer << ' ';
  printer << getOffsets().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult CreateDescOp::verify() {
  auto mode = getMode();
  auto mapping = getTensorDesc().getType().getMapping();
  auto offsetTy = getOffsets().getType();
  auto tdescTy = getTensorDesc().getType();
  auto chunkSize = getChunkSizePerLane();

  if (mode == imex::xegpu::Mode::SIMT || mapping) {
    return emitOpError("CreateDescOp only support VC mode and mapping "
                       "attribute of TensorDesc is not expected.\n");
  }

  if (getRankOf(getSource()) > 2)
    return emitOpError(
        "Expecting the source is a 1D/2D memref or pointer (uint64_t).");

  if (!tdescTy.getScattered())
    return emitOpError(
        "Expecting the presence of ScatteredAttr for tensor descriptor.");

  // Infer the TensorDesc shape
  std::vector<int64_t> shape;
  if (llvm::isa<mlir::VectorType>(offsetTy)) {
    shape = llvm::dyn_cast<mlir::VectorType>(offsetTy).getShape().vec();
    if (shape.size() != 1)
      return emitOpError("Expecting the offset is a 1D vector.");
  }

  if (chunkSize != 1) {
    shape.push_back(chunkSize);
  }

  auto tdescShape = tdescTy.getShape();
  if (shape != tdescShape.vec()) {
    return emitOpError("Expecting dimensions of offsets is the same as the "
                       "tensor descriptor, or one less than.");
  }

  return mlir::success();
}

void CreateDescOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         TensorDescType TensorDesc, mlir::Value source,
                         mlir::Value offsets, uint32_t chunk_size_per_lane) {
  state.addOperands(source);
  state.addOperands(offsets);
  state.getOrAddProperties<Properties>().chunk_size_per_lane =
      builder.getIntegerAttr(builder.getIntegerType(32), chunk_size_per_lane);
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
  state.addTypes(TensorDesc);
}

void CreateDescOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         TensorDescType TensorDesc, mlir::Value source,
                         mlir::Value offsets,
                         mlir::IntegerAttr chunk_size_per_lane) {
  state.addOperands(source);
  state.addOperands(offsets);
  if (chunk_size_per_lane)
    state.getOrAddProperties<Properties>().chunk_size_per_lane =
        chunk_size_per_lane;
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
  state.addTypes(TensorDesc);
}

mlir::ParseResult LoadNDOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(
          parser, result,
          {"mode", "vnni_axis", "transpose", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();

  if (parser.parseArrow())
    return mlir::failure();

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseType(valueRawTypes[0]))
    return mlir::failure();

  result.addTypes(valueTypes);
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();

  return mlir::success();
}

void LoadNDOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  bool printSep = false;
  auto printDefaults = printDefaultValues();
  auto numAttrs = (*this)->getAttrs().size();

  printer << ' ';
  printer << getTensorDesc();

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << ' ' << "{";
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << "mode = " << mode;
    printSep = true;
  }

  if (getVnniAxisAttr()) {
    if (printSep)
      printer << "," << ' ';
    printer << "vnni_axis = " << getVnniAxis().value();
    printSep = true;
  }

  if (getTransposeAttr()) {
    if (printSep)
      printer << "," << ' ';
    printer << "transpose = ";
    getTransposeAttr().print(printer);
    printSep = true;
  }

  printCacheHintAttrs<LoadNDOp>(printer, *this, printSep);

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getValue().getType();
}

mlir::LogicalResult LoadNDOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getValueType();

  if (tdescTy.getRank() > 2)
    return emitOpError(
        "The TensorDesc for LoadNDOp should be a 2D/1D TensorDesc.");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  auto tdescElemTy = tdescTy.getElementType();
  auto valueElemTy = valueTy.getElementType();

  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  if (tdescTy.getRank() == 2) {
    // TODO: The following logic are architecture
    // dependent, pending to be moved out
    auto width = tdescTy.getShape()[1];
    auto height = tdescTy.getShape()[0];
    auto elemTyByteWidth = tdescElemTy.getIntOrFloatBitWidth() / 8;

    if (width < MIN_2D_BLOCK_WIDTH_IN_ELEMENTS ||
        width > MAX_2D_BLOCK_WIDTH_IN_ELEMENTS ||
        (width * elemTyByteWidth) % 4 != 0) {
      return emitOpError(
          "Invalid width size for 2D block load.  "
          "The specification expects the value to "
          "be in range [1, 64], and The the total "
          "data size (width * elemTyBytes) to be multiple of 4.\n");
    }

    if (height < MIN_2D_BLOCK_HEIGHT_IN_ELEMENTS ||
        height > MAX_2D_BLOCK_HEIGHT_IN_ELEMENTS) {
      return emitOpError("Invalid height size for 2D block load. The "
                         "specification expects the "
                         "value to be in range [1, 32].\n");
    }
  }

  auto mode = getMode();
  auto tdescShape = tdescTy.getShape().vec();
  auto valueShape = valueTy.getShape().vec();
  auto array_len = tdescTy.getArrayLength();

  if (mode == imex::xegpu::Mode::SIMT) {
    auto sgMap = tdescTy.getMapping();
    if (!sgMap) {
      return emitOpError(
          "Expecting SgMap attribute for SIMT mode operators.\n");
    }

    if (!verifyAndInferShape(tdescShape, sgMap)) {
      return emitOpError("Failed to infer the shape.")
             << "The new shape[i] should meet the following condistions "
                "for SubGroupMapAttr: "
             << "\n\ttdescShape[i] % mma_block_size[i] == 0 (if it has) && "
             << "\n\ttdescShape[i] % wi_layout[i] == 0 && "
             << "\n\ttdescShape[i] % wi_data[i] == 0 && "
             << "\n\t(tdescShape[i] % (wi_layout[i] * wi_data[i]) == 0 || "
             << "\n\t (wi_layout[i] * wi_data[i]) % tdescShape[i] == 0).\n";
    }
  }

  if (getTranspose()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() >= trans.size())
      transpose(trans, tdescShape);
    else
      emitWarning("Invalid transpose attr. It is ignored.");
  }

  if (getVnniAxis()) {
    auto axis = getVnniAxis().value();
    auto vnni_factor = valueShape.back();
    tdescShape[axis] /= vnni_factor;
    tdescShape.push_back(vnni_factor);
  }

  if (array_len > 1) {
    auto it = tdescShape.begin();
    tdescShape.insert(it, array_len);
  }

  if (tdescShape != valueShape)
    return emitOpError("Result shape doesn't match TensorDesc shape.")
           << "\nThe expected shape is " << makeString(tdescShape) << "."
           << "\nBut the given shape is " << makeString(valueShape) << "."
           << "\nIn VC mode, when VNNI is not enabled, the result should have "
           << "the same shape (or transposed shape if transpose is enabled) "
           << "as TensorDesc; \nwhen VNNI is enabled, the result should have "
           << "one more dimention than the TensorDesc, with last dimention "
           << "having vnni factor, \nbut having same number of total data "
           << "elements. The vnni factor are typically calculated as "
           << "simd_lane_width / elementTypeBitWidth. \nFor element type "
           << "having more than 32 bits, vnni shouldn't be used. \nIn SIMT "
           << "mode, the shape is derived from the mapping attributes.\n";
  return mlir::success();
}

mlir::ParseResult StoreNDOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  llvm::SMLoc valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"mode", "l1_hint", "l2_hint", "l3_hint"}, true))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseType(valueRawTypes[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();

  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return mlir::failure();

  return mlir::success();
}

void StoreNDOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  [[maybe_unused]] bool printSep = false;
  auto printDefaults = printDefaultValues();
  auto numAttrs = (*this)->getAttrs().size();

  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << ' ' << "{";
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << "mode = " << getMode();
    printSep = true;
  }

  printCacheHintAttrs<StoreNDOp>(printer, *this, true);

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getValue().getType();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult StoreNDOp::verify() {
  auto dstTy = getTensorDesc().getType();                              // Tile
  auto valTy = llvm::dyn_cast<mlir::VectorType>(getValue().getType()); // Vector

  if (dstTy.getRank() > 2)
    return emitOpError(
        "The TensorDesc for StoreNdOp should be a 2D TensorDesc.");

  if (dstTy.getArrayLength() > 1)
    return emitOpError("Store operators don't support array block yet.");

  if (!valTy)
    return emitOpError("Invalid value operand, it should be a VectorType.\n");

  auto dstElemTy = dstTy.getElementType();
  auto valElemTy = valTy.getElementType();

  if (dstElemTy != valElemTy) {
    return emitOpError("The elem type of value (vector) shape doesn't match "
                       "the elem type of memory (dst) shape.\n");
  }

  if (dstTy.getRank() == 2) { // TODO: The following logic are architecture
                              // dependent, pending to be moved
    // out
    auto width = dstTy.getShape()[1];
    auto height = dstTy.getShape()[0];
    auto elemTyByteWidth = dstElemTy.getIntOrFloatBitWidth() / 8;
    if (width < MIN_2D_BLOCK_WIDTH_IN_ELEMENTS ||
        width > MAX_2D_BLOCK_WIDTH_IN_ELEMENTS ||
        (width * elemTyByteWidth) % 4 != 0) {
      return emitOpError(
          "Invalid width size for 2D block write. "
          "The specification expects the value to "
          "be in range [1, 64], and The the total "
          "data size (width * elemTyBytes) to be multiple of 4.\n");
    }

    if (height < MIN_2D_BLOCK_HEIGHT_IN_ELEMENTS ||
        height > MAX_2D_BLOCK_HEIGHT_IN_ELEMENTS) {
      return emitOpError(
          "Invalid height size for 2D block write. The specification"
          "expects the value to be in range [1, 32].\n");
    }
  }

  auto mode = getMode();

  if (mode == imex::xegpu::Mode::VC) { // for VC mode, no attr attached
    if (dstTy.getShape() != valTy.getShape())
      return emitOpError("In VC mode, the value (vector) shape doesn't match "
                         "the memory (dst) shape.\n");
  } else {
    auto mapping = dstTy.getMapping();
    if (!mapping) {
      return emitOpError(
          "Expecting SgMap attribute for SIMT mode operators.\n");
    }

    imex::xegpu::SubGroupMapAttr sgMap;
    std::vector<int64_t> shape = dstTy.getShape().vec();

    sgMap = llvm::dyn_cast<imex::xegpu::SubGroupMapAttr>(mapping);

    if (!verifyAndInferShape(shape, sgMap)) {
      return emitOpError("Failed to infer the shape.")
             << "The new shape[i] should meet the following condistions "
                "for SubGroupMapAttr: "
             << "\n\ttdescShape[i] % mma_block_size[i] == 0 (if it has) && "
             << "\n\ttdescShape[i] % wi_layout[i] == 0 && "
             << "\n\ttdescShape[i] % wi_data[i] == 0 && "
             << "\n\t(tdescShape[i] % (wi_layout[i] * wi_data[i]) == 0 || "
             << "\n\t (wi_layout[i] * wi_data[i]) % tdescShape[i] == 0).\n";
    }

    if (shape != valTy.getShape().vec())
      return emitOpError(
          "In SIMT mode, the value (vector) shape doesn't match the memory"
          "(dst) shape as derived according to the mapping rule.\n");
  }
  return mlir::success();
}

mlir::ParseResult PrefetchNDOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;
  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"mode", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();
  return mlir::success();
}

void PrefetchNDOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  [[maybe_unused]] bool printSep = false;
  auto printDefaults = printDefaultValues();
  auto numAttrs = (*this)->getAttrs().size();
  printer << ' ';
  printer << getTensorDesc();

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << ' ' << "{";
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << "mode = " << getMode();
    printSep = true;
  }

  printCacheHintAttrs<PrefetchNDOp>(printer, *this, true);

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult DpasOp::verify() {

  int64_t lhsRank = getLhsType().getRank();
  int64_t rhsRank = getRhsType().getRank();
  mlir::Type lhsElemType = getLhsType().getElementType();
  mlir::Type rhsElemType = getRhsType().getElementType();
  mlir::Type resultElemType = getResultType().getElementType();

  // TODO: this is hardware specific, need to be moved out.
  if (!dpasSupportedTypes(lhsElemType, 0)) {
    return emitOpError("Unsupported src datatype for dpas op");
  }

  // TODO: this is hardware specific, need to be moved out.
  if (!dpasSupportedTypes(resultElemType, 1)) {
    return emitOpError("Unsupported result datatype for dpas op");
  }

  if (lhsElemType != rhsElemType) {
    return emitOpError("lhs and rhs element type does not match for dpas op");
  }

  if (getAcc()) {
    if (getAccType() != getResultType())
      return emitOpError("Accumulator and Result for dpas op should have the "
                         "same type (both shape and element type).");
  }

  // TODO: SIMT makes it harder to check semantic errors for DPAS op.
  // the only thing we can check seems to be vnni factor. But it
  // depends on hardware though.
  // if (!dpasSupportedShapes(*this)) {
  //   return emitOpError("Incorrect shapes for dpas op");
  // }

  if (lhsRank != rhsRank || lhsRank != 3) {
    return emitOpError(
        "lhs and rhs rank does not match for dpas op, or their rank is not 3.");
  }

  return mlir::success();
}

mlir::ParseResult LoadGatherOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;
  mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> maskOperands(
      maskRawOperands);
  llvm::SMLoc maskOperandsLoc;

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);
  mlir::Type maskRawTypes[1];
  llvm::ArrayRef<mlir::Type> maskTypes(maskRawTypes);
  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<mlir::Type> valueTypes(valueRawTypes);

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(
          parser, result,
          {"mode", "vnni_axis", "transpose", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return mlir::failure();

  if (parser.parseArrow())
    return mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return mlir::failure();

  result.addTypes(valueTypes);

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();

  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void LoadGatherOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  bool printSep = false;
  auto printDefaults = printDefaultValues();
  auto numAttrs = (*this)->getAttrs().size();

  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << ' ' << "{";
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << "mode = " << getMode();
    printSep = true;
  }

  if (getVnniAxisAttr()) {
    if (printSep)
      printer << "," << ' ';
    printer << "vnni_axis = " << getVnniAxis().value();
    printSep = true;
  }

  if (getTransposeAttr()) {
    if (printSep)
      printer << "," << ' ';
    printer << "transpose = ";
    getTransposeAttr().print(printer);
    printSep = true;
  }

  printCacheHintAttrs<LoadGatherOp>(printer, *this, printSep);

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ",";
  printer << ' ';
  printer << getMask().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getValue().getType();
}

mlir::LogicalResult LoadGatherOp::verify() {
  auto tdescTy = getTensorDescType();
  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!tdescTy.getScattered())
    return emitOpError(
        "LoadGatherOp only works on TensorDesc with ScatteredAttr.");

  auto getElementType = [&](mlir::Type type) -> mlir::Type {
    if (type.isIntOrIndexOrFloat())
      return type;
    else if (llvm::isa<mlir::VectorType>(type))
      return llvm::dyn_cast<mlir::VectorType>(type).getElementType();
    else if (llvm::isa<imex::xegpu::TensorDescType>(type))
      return llvm::dyn_cast<imex::xegpu::TensorDescType>(type).getElementType();
    assert(0 && "Unreachable !!!");
    return type;
  };

  auto tdescElemTy = getElementType(tdescTy);
  auto valueElemTy = getElementType(valueTy);
  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  auto getShape = [&](mlir::Type type, std::vector<int64_t> &shape) -> void {
    if (type.isIntOrIndexOrFloat())
      shape.push_back(1);
    else if (llvm::isa<mlir::VectorType>(type))
      shape = llvm::dyn_cast<mlir::VectorType>(type).getShape().vec();
    else
      assert(0 && "Unreachable !!!");
  };

  std::vector<int64_t> maskShape, valueShape;
  getShape(maskTy, maskShape);
  getShape(valueTy, valueShape);
  auto tdescShape = tdescTy.getShape().vec();

  if (tdescShape != maskShape)
    return emitOpError("Mask should have the same shape as TensorDesc.");

  auto mode = getMode();
  auto mapping = tdescTy.getMapping();
  if (mode == imex::xegpu::Mode::SIMT || mapping) {
    return emitOpError("LoadGatherOp only supports VC mode and mapping "
                       "attribute of TensorDesc is not expected.\n");
  }

  if (getTranspose()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() >= trans.size())
      transpose(trans, tdescShape);
    else
      emitWarning("Invalid transpose attr. It is ignored.");
  }

  if (getVnniAxis()) {
    auto axis = getVnniAxis().value();
    auto vnni_factor = valueShape.back();
    tdescShape[axis] /= vnni_factor;
    tdescShape.push_back(vnni_factor);
  }

  if (valueShape != tdescShape)
    return emitOpError(
        "Result shape doesn't match TensorDesc shape. when VNNI is not enabled,"
        "the result should have the same shape (or transposed shape if "
        "transpose is also enabled) as TensorDesc. When VNNI is enabled, "
        "the result should have one more dimention than the TensorDesc, "
        "with last dimention having vnni factor, but having same number of"
        "total data elements. The vnni factor are typically calculated as "
        "simd_lane_width/elementTypeBitWidth. For element type having "
        "more than 32 bits, vnni shouldn't be used.\n");

  return mlir::success();
}

void LoadGatherOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Type value, mlir::Value TensorDesc,
                         mlir::Value mask, mlir::IntegerAttr vnni_axis,
                         mlir::DenseI64ArrayAttr transpose,
                         CacheReadHintAttr l1_hint, CacheReadHintAttr l2_hint,
                         CacheReadHintAttr l3_hint) {
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  if (vnni_axis)
    state.getOrAddProperties<Properties>().vnni_axis = vnni_axis;

  if (transpose)
    state.getOrAddProperties<Properties>().transpose = transpose;

  if (l1_hint)
    state.getOrAddProperties<Properties>().l1_hint = l1_hint;

  if (l2_hint)
    state.getOrAddProperties<Properties>().l2_hint = l2_hint;

  if (l3_hint)
    state.getOrAddProperties<Properties>().l3_hint = l3_hint;

  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
  state.addTypes(value);
}

void LoadGatherOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Type value, mlir::Value TensorDesc,
                         mlir::Value mask, mlir::IntegerAttr vnni_axis,
                         mlir::DenseI64ArrayAttr transpose,
                         CacheReadHint l1_hint, CacheReadHint l2_hint,
                         CacheReadHint l3_hint) {
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  if (vnni_axis)
    state.getOrAddProperties<Properties>().vnni_axis = vnni_axis;

  if (transpose)
    state.getOrAddProperties<Properties>().transpose = transpose;

  state.getOrAddProperties<Properties>().l1_hint =
      CacheReadHintAttr::get(builder.getContext(), l1_hint);
  state.getOrAddProperties<Properties>().l2_hint =
      CacheReadHintAttr::get(builder.getContext(), l2_hint);
  state.getOrAddProperties<Properties>().l3_hint =
      CacheReadHintAttr::get(builder.getContext(), l3_hint);
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
  state.addTypes(value);
}

mlir::ParseResult StoreScatterOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;

  mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  llvm::SMLoc valueOperandsLoc;

  mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> maskOperands(
      maskRawOperands);
  llvm::SMLoc maskOperandsLoc;

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<mlir::Type> valueTypes(valueRawTypes);

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);

  mlir::Type maskRawTypes[1];
  llvm::ArrayRef<mlir::Type> maskTypes(maskRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"mode", "l1_hint", "l2_hint", "l3_hint"}, true))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return mlir::failure();

  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return mlir::failure();

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();

  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void StoreScatterOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  bool printSep = false;
  auto printDefaults = printDefaultValues();
  auto numAttrs = (*this)->getAttrs().size();

  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << ' ' << "{";
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << "mode = " << getMode();
    printSep = true;
  }

  printCacheHintAttrs<StoreScatterOp>(printer, *this, printSep);

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getValue().getType();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ",";
  printer << ' ';
  printer << getMask().getType();
}

mlir::LogicalResult StoreScatterOp::verify() {
  auto valueTy = getValueType();
  auto tdescTy = getTensorDescType();
  auto maskTy = getMaskType();

  if (!tdescTy.getScattered())
    return emitOpError("Invalid TensorDesc. StoreScatterOp only works on "
                       "TensorDescs with ScatteredAttr.");

  if (tdescTy.getArrayLength() > 1)
    return emitOpError("Store operators don't support array block yet.");

  std::vector<int64_t> valueShape, maskShape;
  auto getShape = [&](mlir::Type type, std::vector<int64_t> &shape) -> void {
    if (type.isIntOrIndexOrFloat())
      shape.push_back(1);
    else if (llvm::isa<mlir::VectorType>(type))
      shape = llvm::dyn_cast<mlir::VectorType>(type).getShape().vec();
    else
      assert(0 && "Unreachable !!!");
  };

  getShape(valueTy, valueShape);
  getShape(maskTy, maskShape);

  if (valueShape != maskShape) {
    return emitOpError("Mask and value should have the same shape/size");
  }

  auto tdescShape = tdescTy.getShape().vec();

  auto mode = getMode();
  auto mapping = tdescTy.getMapping();

  if (mode != imex::xegpu::Mode::VC || mapping) {
    return emitOpError("StoreScatterOp only supports VC mode and mapping "
                       "attribute of TensorDesc is not expected.\n");
  }

  if (tdescShape != valueShape) {
    return emitOpError("TensorDesc shape and value shape doesn't match. ")
           << "The expected/derived value shape is: " << makeString(tdescShape)
           << ".\nMask and value should have the same shape/size as "
              "TensorDesc.\n";
  }

  return mlir::success();
}

void StoreScatterOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value value,
                           mlir::Value TensorDesc, mlir::Value mask,
                           CacheWriteHintAttr l1_hint,
                           CacheWriteHintAttr l2_hint,
                           CacheWriteHintAttr l3_hint) {
  state.addOperands(value);
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  if (l1_hint) {
    state.getOrAddProperties<Properties>().l1_hint = l1_hint;
  }
  if (l2_hint) {
    state.getOrAddProperties<Properties>().l2_hint = l2_hint;
  }
  if (l3_hint) {
    state.getOrAddProperties<Properties>().l3_hint = l3_hint;
  }
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
}

void StoreScatterOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value value,
                           mlir::Value TensorDesc, mlir::Value mask,
                           CacheWriteHint l1_hint, CacheWriteHint l2_hint,
                           CacheWriteHint l3_hint) {
  state.addOperands(value);
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  state.getOrAddProperties<Properties>().l1_hint =
      CacheWriteHintAttr::get(builder.getContext(), l1_hint);
  state.getOrAddProperties<Properties>().l2_hint =
      CacheWriteHintAttr::get(builder.getContext(), l2_hint);
  ;
  state.getOrAddProperties<Properties>().l3_hint =
      CacheWriteHintAttr::get(builder.getContext(), l3_hint);
  ;
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
}

mlir::ParseResult PrefetchOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;
  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"mode", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return mlir::failure();
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();
  return mlir::success();
}

void PrefetchOp::print(mlir::OpAsmPrinter &printer) {
  auto mode = getMode();
  bool printSep = false;
  auto printDefaults = printDefaultValues();
  auto numAttrs = (*this)->getAttrs().size();

  printer << ' ';
  printer << getTensorDesc();

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << ' ' << "{";
  }

  if (printDefaults || mode != imex::xegpu::Mode::SIMT) {
    printer << "mode = " << getMode();
    printSep = true;
  }

  printCacheHintAttrs<PrefetchOp>(printer, *this, printSep);

  if (printDefaults || mode != imex::xegpu::Mode::SIMT || numAttrs > 1) {
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult PrefetchOp::verify() {
  auto mode = getMode();
  auto tdescTy = getTensorDesc().getType();
  auto mapping = tdescTy.getMapping();

  if (tdescTy.getScattered())
    return emitOpError("Invalid TensorDesc. PrefetchOp only works on "
                       "TensorDescs with ScatteredAttr.");

  if (mode != imex::xegpu::Mode::VC || mapping) {
    return emitOpError("PrefetchOp only supports VC mode. and mapping "
                       "attribute of TensorDesc is not expected.\n");
  }

  return mlir::success();
}

void PrefetchOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       mlir::Value TensorDesc, CacheReadHintAttr l1_hint,
                       CacheReadHintAttr l2_hint, CacheReadHintAttr l3_hint) {
  state.addOperands(TensorDesc);
  if (l1_hint)
    state.getOrAddProperties<Properties>().l1_hint = l1_hint;

  if (l2_hint)
    state.getOrAddProperties<Properties>().l2_hint = l2_hint;

  if (l3_hint)
    state.getOrAddProperties<Properties>().l3_hint = l3_hint;

  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
}

void PrefetchOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       mlir::Value TensorDesc, CacheReadHint l1_hint,
                       CacheReadHint l2_hint, CacheReadHint l3_hint) {
  state.addOperands(TensorDesc);
  state.getOrAddProperties<Properties>().l1_hint =
      CacheReadHintAttr::get(builder.getContext(), l1_hint);
  state.getOrAddProperties<Properties>().l2_hint =
      CacheReadHintAttr::get(builder.getContext(), l2_hint);
  state.getOrAddProperties<Properties>().l3_hint =
      CacheReadHintAttr::get(builder.getContext(), l3_hint);
  ;
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
}

mlir::LogicalResult UpdateOffsetOp::verify() {
  auto srcTy = getTensorDesc().getType();
  auto offTy = getOffsets().getType();
  auto resTy = getResult().getType();

  if (srcTy != resTy)
    return emitOpError(
        "The result should have the same type"
        "(shape and encoding attribute) as the input TensorDesc.");

  auto shape = srcTy.getShape();

  if (!srcTy.getScattered()) {
    return emitOpError("Invalid TensorDesc. UpdateOffsetOp only works on "
                       "TensorDescs with ScatteredAttr.");
  }

  auto vecTy = llvm::dyn_cast<mlir::VectorType>(offTy);
  if (!vecTy || vecTy.getRank() != 1)
    return emitOpError("The offset should be an 1D vector.\n");

  if (shape[0] != vecTy.getShape()[0])
    return emitOpError(
        "The offset should have same length as the dim-0 of TensorDesc.");

  return mlir::success();
}

void UpdateOffsetOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Type result,
                           mlir::Value TensorDesc, mlir::Value offsets) {
  state.addOperands(TensorDesc);
  state.addOperands(offsets);
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
  state.addTypes(result);
}

mlir::LogicalResult UpdateNDOffsetOp::verify() {
  // number of offsets specified must match the rank of the tensor descriptor
  if (getTensorDesc().getType().getRank() != (int64_t)getOffsets().size()) {
    return emitOpError("Invalid number of offsets.");
  }
  return mlir::success();
}

void InvokeSIMDOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::SymbolRefAttr callee, mlir::TypeRange results,
                         imex::xegpu::ArgTypeAttr argType,
                         mlir::ValueRange operands) {
  state.addOperands(operands);
  state.addAttribute("argType", argType);
  state.addAttribute("callee", callee);
  state.addTypes(results);
}

void InvokeSIMDOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::StringAttr callee, mlir::TypeRange results,
                         imex::xegpu::ArgTypeAttr argType,
                         mlir::ValueRange operands) {
  build(builder, state, mlir::SymbolRefAttr::get(callee), results, argType,
        operands);
}

void InvokeSIMDOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         llvm::StringRef callee, mlir::TypeRange results,
                         imex::xegpu::ArgTypeAttr argType,
                         mlir::ValueRange operands) {
  build(builder, state, mlir::StringAttr::get(builder.getContext(), callee),
        results, argType, operands);
}

mlir::LogicalResult AtomicRMWOp::verify() {
  auto mode = getMode();
  if (mode != imex::xegpu::Mode::VC) {
    return emitOpError("AtomicRMWOp only work on VC mode.\n");
  }
  return mlir::success();
}

void AtomicRMWOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Type result, imex::xegpu::AtomicRMWKindAttr kind,
                        mlir::Value tensorDesc, mlir::Value mask,
                        mlir::Value value) {
  state.addOperands(tensorDesc);
  state.addOperands(mask);
  if (value)
    state.addOperands(value);
  state.getOrAddProperties<Properties>().kind = kind;
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
  state.addTypes(result);
}

void AtomicRMWOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Type result, imex::xegpu::AtomicRMWKind kind,
                        mlir::Value tensorDesc, mlir::Value mask,
                        mlir::Value value) {
  state.addOperands(tensorDesc);
  state.addOperands(mask);
  if (value)
    state.addOperands(value);
  state.getOrAddProperties<Properties>().kind =
      AtomicRMWKindAttr::get(builder.getContext(), kind);
  state.getOrAddProperties<Properties>().mode =
      ModeAttr::get(builder.getContext(), Mode::VC);
  state.addTypes(result);
}

} // namespace xegpu
} // namespace imex

#include <imex/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/XeGPU/IR/XeGPU.cpp.inc>
