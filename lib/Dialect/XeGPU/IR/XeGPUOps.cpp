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

#include <imex/Dialect/XeGPU/IR/XeGPUOps.h>
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

// static bool vnniVerifier(size_t axis, llvm::ArrayRef<int64_t> tdescShape,
//                          llvm::ArrayRef<int64_t> valueShape,
//                          size_t elemTyBitWidth) {
//   bool isValid = valueShape.size() == tdescShape.size() + 1;

//   for (size_t i = 0; i < tdescShape.size(); i++) {
//     if ((i == axis && valueShape[i] * valueShape.back() != tdescShape[i]) ||
//         (i != axis && valueShape[i] != tdescShape[i]))
//       isValid = false;
//   }

//   const static size_t xeSIMDLaneBitWidth = 32;
//   auto vnni_factor = valueShape.back();

//   isValid &= vnni_factor == xeSIMDLaneBitWidth / elemTyBitWidth;

//   return isValid;
// }

static void transpose(llvm::ArrayRef<int64_t> trans,
                      std::vector<int64_t> &shape) {
  std::vector<int64_t> old = shape;
  for (size_t i = 0; i < trans.size(); i++)
    shape[i] = old[trans[i]];
};

static bool isMappingAttr(mlir::Attribute attr) {
  return attr && (llvm::isa<imex::xegpu::SubGroupMapAttr>(attr) ||
                  llvm::isa<imex::xegpu::WorkGroupMapAttr>(attr) ||
                  llvm::isa<imex::xegpu::XeMapAttr>(attr));
}

bool dpasSupportedTypes(mlir::Type type, bool isResult) {
  if (isResult) {
    if (type.isF32() || type.isInteger(32))
      return true;
    else
      return false;
  } else {
    if (type.isF16() || type.isBF16() || type.isInteger(8))
      return true;
    else
      return false;
  }
}

// bool dpasSupportedShapes(DpasOp op) {
//   mlir::Type lhsElemType = op.getLhsType().getElementType();
//   // TODO: handle dynamic shape cast
//   auto lhsShape = op.getLhsType().cast<mlir::ShapedType>().getShape();
//   auto rhsShape = op.getRhsType().cast<mlir::ShapedType>().getShape();
//   // Retrieve 2D shapes(MxK * KxN) from 3D. Verify this
//   auto m = lhsShape[0];
//   auto k = lhsShape[1] * lhsShape[2];
//   auto n = rhsShape[1];

//   if ((lhsElemType.isF16() || lhsElemType.isBF16()) && m <= MAX_TM_SIZE &&
//       n == TN_SIZE && k == TK_SIZE_FOR_D16) {
//     return true;
//   }

//   if (lhsElemType.isInteger(8) && m <= MAX_TM_SIZE && n == TN_SIZE &&
//       k == TK_SIZE_FOR_D8) {
//     return true;
//   }

//   return false;
// }

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
      return ::mlir::failure();

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

mlir::ParseResult CreateNdDescOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {

  // parse the source operand
  mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceRawOperands);
  llvm::SMLoc sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();

  // parse the offset operand, in format of [x, y]
  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  mlir::DenseI64ArrayAttr static_offsetsAttr;
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
    return ::mlir::failure();
  result.addAttribute("static_offsets", static_offsetsAttr);

  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> shapeOperands;
  llvm::SMLoc shapeOperandsLoc;

  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  llvm::SMLoc stridesOperandsLoc;
  // parse optional shape and strides, shape and strides should always come
  // together
  if (::mlir::succeeded(parser.parseOptionalComma())) {
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
      return ::mlir::failure();
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(stridesOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return ::mlir::failure();
  }

  if (parseOptionalAttrDict(parser, result, {"boundary_check", "mode"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseArrow())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(offsetsOperands.size()),
                           static_cast<int32_t>(shapeOperands.size()),
                           static_cast<int32_t>(stridesOperands.size())}));

  mlir::Type indexType = parser.getBuilder().getIndexType();
  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, indexType, offsetsOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(shapeOperands, indexType, shapeOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, indexType, stridesOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateNdDescOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  printDynamicIndexList(printer, *this, getOffsets(), getStaticOffsetsAttr());
  if (!getShape().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getShape();
    printer << "]";
  }

  if (!getStrides().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getStrides();
    printer << "]";
  }

  printer << ' ' << "{";
  printer << "mode = " << getMode();
  printer << "," << ' ';
  printer << "boundary_check = " << getBoundaryCheck();
  printer << "}";

  printer << ' ' << ":";
  printer << ' ';
  printer << getSource().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult CreateNdDescOp::verify() {
  auto mode = getMode();
  auto encoding = getTensorDesc().getType().getEncoding();

  if (mode == imex::xegpu::Mode::SIMT && !isMappingAttr(encoding)) {
    return emitOpError("Expecting either SgMap, WgMap or XeMap attribute for "
                       "SIMT mode operators.\n");
  }

  // it is invalid to have both dynamic and static shape
  if (!(hasDynamicShape() ^ hasStaticShape()))
    return emitOpError("It is invalid to have both or none of dynamic shape "
                       "and static shape. Only one of them is needed.");

  if (getOffsetsRank() != getShapeRank() ||
      getShapeRank() != getStridesRank() ||
      (isMemRef(getSource()) && getRankOf(getSource()) != getOffsetsRank()))
    return emitOpError("Expecting the rank of shape, strides and offsets "
                       "should match with each other.");

  return mlir::success();
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
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> offsetsOperands(
      offsetsRawOperands);
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(offsetsRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result, {"chunk_size_per_lane", "mode"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  mlir::Type offsetsRawTypes[1];
  llvm::ArrayRef<mlir::Type> offsetsTypes(offsetsRawTypes);
  if (parser.parseType(offsetsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, offsetsTypes, offsetsOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateDescOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  printer << ",";
  printer << ' ';
  printer << getOffsets();

  printer << ' ' << "{";
  printer << "mode = " << getMode();
  printer << "," << ' ';
  printer << "chunk_size_per_lane = " << getChunkSizePerLane();
  printer << "}";

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
  if (getRankOf(getSource()) > 2)
    return emitOpError(
        "Expecting the source is a 2D/1D memref or pointer (uint64_t).");

  std::vector<int64_t> shape;

  auto offsetTy = getOffsets().getType();
  auto tdescTy = getTensorDesc().getType();
  auto chunkSize = getChunkSizePerLane();

  auto tdescShape = tdescTy.getShape();

  if (llvm::isa<mlir::VectorType>(offsetTy)) {
    shape = llvm::dyn_cast<mlir::VectorType>(offsetTy).getShape().vec();
    if (shape.size() > 2)
      return emitOpError(
          "Expecting the offset is either a 2D/1D vector (for VC) "
          "or scalar (for SIMT).");
  }

  if (offsetTy.isIndex() || chunkSize != 1) {
    shape.push_back(chunkSize);
  }

  if (shape != tdescShape.vec()) {
    return emitOpError("Expecting dimensions of offsets is the same as the "
                       "tensor descriptor, or one less than.");
  }

  if (!tdescTy.getEncoding())
    return emitOpError(
        "Expecting the presence of scattered attribute for tensor descriptor.");

  return mlir::success();
}

mlir::ParseResult LoadNDOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parseOptionalAttrDict(
          parser, result,
          {"mode", "vnni_axis", "transpose", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseArrow())
    return ::mlir::failure();

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  result.addTypes(valueTypes);
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();

  return mlir::success();
}

void LoadNDOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getTensorDesc();

  if ((*this)->getAttrs().size()) {
    printer << ' ' << "{";
    printer << "mode = " << getMode();
    if (getVnniAxisAttr()) {
      printer << "," << ' ';
      printer << "vnni_axis = " << getVnniAxis().value();
    }

    if (getTransposeAttr()) {
      printer << "," << ' ';
      printer << "transpose = ";
      getTransposeAttr().print(printer);
    }

    printCacheHintAttrs<LoadNDOp>(printer, *this, true);

    printer << "}";
  }
  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getValue().getType();
}

// mlir::LogicalResult CreateNbarrierOp::verify() {
//   llvm::dbgs() << "\nOp: " << getValueAsString(*this)
//                << "\n\tnum producers: " << getNumProducers()
//                << "\n\tnum consumers: " << getNumConsumers()
//                << "\n\n";
//   return mlir::success();
// }

mlir::LogicalResult LoadNDOp::verify() {
  auto tdescTy = getTensorDesc().getType();
  auto valueTy = llvm::dyn_cast<mlir::VectorType>(getValue().getType());

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

  if (tdescTy.getRank() == 2) { // TODO: The following logic are architecture
                                // dependent, pending to be moved
    // out
    auto width = tdescTy.getShape()[1];
    auto height = tdescTy.getShape()[0];
    auto elemTyByteWidth = tdescElemTy.getIntOrFloatBitWidth() / 8;

    if (width < MIN_2D_BLOCK_WIDTH_IN_ELEMENTS ||
        width > MAX_2D_BLOCK_WIDTH_IN_ELEMENTS ||
        (width * elemTyByteWidth) % 4 != 0) {
      return emitOpError("Invalid width size for 2D block load.  \
                          The specification expects the value to \
                          be in range [1, 64], and The the total \
                          data size (width * elemTyBytes) to be multiple of 4.\n");
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

  if (mode == imex::xegpu::Mode::SIMT) {
    imex::xegpu::WorkGroupMapAttr wgMap;
    imex::xegpu::SubGroupMapAttr sgMap;

    auto encoding = tdescTy.getEncoding();
    if (!isMappingAttr(encoding)) {
      return emitOpError("Expecting either SgMap, WgMap or XeMap attribute for "
                         "SIMT mode operators.\n");
    }

    if (auto xeMapAttr = llvm::dyn_cast<imex::xegpu::XeMapAttr>(encoding)) {
      wgMap = xeMapAttr.getWg();
      sgMap = xeMapAttr.getSg();
    } else {
      wgMap = llvm::dyn_cast<imex::xegpu::WorkGroupMapAttr>(encoding);
      sgMap = llvm::dyn_cast<imex::xegpu::SubGroupMapAttr>(encoding);
    }

    if (wgMap) {
      auto sgData = wgMap.getSgData();
      auto sgLayout = wgMap.getSgLayout();
      for (size_t i = 0; i < sgData.size(); i++) {
        if (tdescShape[i] % sgLayout[i] != 0 ||
            tdescShape[i] % sgData[i] != 0 ||
            tdescShape[i] % (sgLayout[i] * sgData[i]) != 0)
          return emitOpError("Invalid WorkGroupMapAttr. It should meet the "
                             "following conditions: "
                             "tdescShape[i] % sgLayout[i] == 0 && "
                             "tdescShape[i] % sgData[i] == 0 && "
                             "tdescShape[i] % (sgLayout[i] *sgData[i]) == 0");
        tdescShape[i] /= sgLayout[i];
      }
    }

    if (sgMap) {
      auto blockSize = sgMap.getMmaBlockSize();
      auto wiLayout = sgMap.getWiLayout();
      auto wiData = sgMap.getWiData();
      for (size_t i = 0; i < blockSize.size(); i++) {
        if (tdescShape[i] % blockSize[i] != 0 ||
            blockSize[i] % wiLayout[i] != 0 || blockSize[i] % wiData[i] != 0 ||
            blockSize[i] % (wiLayout[i] * wiData[i]) != 0) {
          return emitOpError("Invalid SubGroupMapAttr. It should meet the "
                             "following conditions: "
                             "tdescShape[i] % blockSize[i] == 0 && "
                             "blockSize[i] % wiLayout[i] == 0 && "
                             "blockSize[i] % wiData[i] == 0 && "
                             "blockSize[i] % (wiLayout[i] * wiData[i]) == 0 ");
        }
      }

      for (size_t i = 0; i < wiLayout.size(); i++) {
        if (tdescShape[i] % wiData[i] != 0 ||
            tdescShape[i] % (wiLayout[i] * wiData[i]) != 0) {
          return emitOpError("Invalid SubGroupMapAttr. It should meet the "
                             "following conditions: "
                             "tdescShape[i] % wiData[i] == 0 && "
                             "tdescShape[i] % (wiLayout[i] * wiData[i]) == 0 ");
        }
        tdescShape[i] /= wiLayout[i];
      }
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

  if (tdescShape != valueShape)
    return emitOpError(
        "Result shape doesn't match TensorDesc shape."
        "The expected shape is " +
        makeString(tdescShape) +
        ", while "
        "the given shape is " +
        makeString(valueShape) +
        ". "
        "In VC mode, when VNNI is not enabled, the result should have the same "
        "shape (or transposed shape if transpose is also enabled) as "
        "TensorDesc; "
        "when VNNI is enabled, the result should have one more dimention than "
        "the "
        "TensorDesc, with last dimention having vnni factor, but having same "
        "number "
        "of total data elements. The vnni factor are typically calculated as "
        "simd_lane_width / elementTypeBitWidth. "
        "For element type having more than 32 bits, vnni shouldn't be used. "
        "In SIMT mode, the shape is derived from the mapping attributes.\n");
  return mlir::success();
}

::mlir::ParseResult StoreNDOp::parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  llvm::SMLoc valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"mode", "l1_hint", "l2_hint", "l3_hint"}, true))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  return ::mlir::success();
}

void StoreNDOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();
  if ((*this)->getAttrs().size()) {
    printer << ' ' << "{";
    printer << "mode = " << getMode();
    printCacheHintAttrs<StoreNDOp>(printer, *this, true);
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
      return emitOpError("Invalid width size for 2D block write. \
                          The specification expects the value to \
                          be in range [1, 64], and The the total \
                          data size (width * elemTyBytes) to be multiple of 4.\n");
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
    auto encoding = dstTy.getEncoding();
    if (!isMappingAttr(encoding)) {
      return emitOpError("Expecting either SgMap, WgMap or XeMap attribute for "
                         "SIMT mode operators.\n");
    }

    imex::xegpu::WorkGroupMapAttr wgMap;
    imex::xegpu::SubGroupMapAttr sgMap;
    std::vector<int64_t> shape = dstTy.getShape().vec();

    if (auto xeMapAttr = llvm::dyn_cast<imex::xegpu::XeMapAttr>(encoding)) {
      wgMap = xeMapAttr.getWg();
      sgMap = xeMapAttr.getSg();
    } else {
      wgMap = llvm::dyn_cast<imex::xegpu::WorkGroupMapAttr>(encoding);
      sgMap = llvm::dyn_cast<imex::xegpu::SubGroupMapAttr>(encoding);
    }

    if (wgMap) {
      auto sgData = wgMap.getSgData();
      auto sgLayout = wgMap.getSgLayout();
      for (size_t i = 0; i < sgData.size(); i++) {
        if (shape[i] % sgLayout[i] != 0 || shape[i] % sgData[i] != 0 ||
            shape[i] % (sgLayout[i] * sgData[i]) != 0)
          return emitOpError("Invalid WorkGroupMapAttr. It should meet the "
                             "following conditions: "
                             "tdescShape[i] % sgLayout[i] == 0 && "
                             "tdescShape[i] % sgData[i] == 0 && "
                             "tdescShape[i] % (sgLayout[i] *sgData[i]) == 0");
        shape[i] /= sgLayout[i];
      }
    }

    if (sgMap) {
      auto blockSize = sgMap.getMmaBlockSize();
      auto wiLayout = sgMap.getWiLayout();
      auto wiData = sgMap.getWiData();
      for (size_t i = 0; i < shape.size(); i++) {
        if (blockSize[i] % (wiLayout[i] * wiData[i]) != 0 ||
            blockSize[i] % wiLayout[i] != 0 || blockSize[i] % wiData[i] != 0 ||
            shape[i] % blockSize[i] != 0) {
          return emitOpError("Invalid SubGroupMapAttr. It should meet the "
                             "following conditions: "
                             "tdescShape[i] % blockSize[i] == 0 && "
                             "blockSize[i] % wiLayout[i] == 0 && "
                             "blockSize[i] % wiData[i] == 0 && "
                             "blockSize[i] % (wiLayout[i] * wiData[i]) == 0 ");
        }
      }

      for (size_t i = 0; i < wiLayout.size(); i++) {
        if (shape[i] % wiData[i] != 0 ||
            shape[i] % (wiLayout[i] * wiData[i]) != 0) {
          return emitOpError("Invalid SubGroupMapAttr. It should meet the "
                             "following conditions: "
                             "tdescShape[i] % wiData[i] == 0 && "
                             "tdescShape[i] % (wiLayout[i] * wiData[i]) == 0 ");
        }
        shape[i] /= wiLayout[i];
      }
    }

    if (shape != valTy.getShape().vec())
      return emitOpError(
          "In SIMT mode, the value (vector) shape doesn't match the memory"
          "(dst) shape as derived according to the mapping rule.\n");
  }
  return mlir::success();
}

::mlir::ParseResult PrefetchNDOp::parse(::mlir::OpAsmParser &parser,
                                        ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;
  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"mode", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrefetchNDOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getTensorDesc();
  // printer.printOptionalAttrDict((*this)->getAttrs());
  if ((*this)->getAttrs().size()) {
    printer << ' ' << "{";
    printer << "mode = " << getMode();
    printCacheHintAttrs<PrefetchNDOp>(printer, *this, true);
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

  if (lhsRank < 3) {
    return emitOpError("dpas op requires 3d vector. Rank is not 3");
  }

  return mlir::success();
}

::mlir::ParseResult LoadGatherOp::parse(::mlir::OpAsmParser &parser,
                                        ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;
  mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> maskOperands(
      maskRawOperands);
  llvm::SMLoc maskOperandsLoc;

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  mlir::Type maskRawTypes[1];
  llvm::ArrayRef<mlir::Type> maskTypes(maskRawTypes);
  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(
          parser, result,
          {"mode", "vnni_axis", "transpose", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  result.addTypes(valueTypes);

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadGatherOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();
  if ((*this)->getAttrs().size()) {
    printer << ' ' << "{";

    printer << "mode = " << getMode();
    if (getVnniAxisAttr())
      printer << ", vnni_axis = " << getVnniAxis().value();

    if (getTransposeAttr()) {
      printer << ", transpose = ";
      getTransposeAttr().print(printer);
    }

    printCacheHintAttrs<LoadGatherOp>(printer, *this, true);

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
  auto tdescTy = getTensorDesc().getType();
  auto maskTy = getMask().getType();
  auto valueTy = getValue().getType();

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
        "transpose"
        "is also enabled) as TensorDesc. When VNNI is enabled, the result "
        "should"
        "have one more dimention than the TensorDesc, with last dimention "
        "having"
        "vnni factor, but having same number of total data elements. The vnni "
        "factor are typically calculated as simd_lane_width / "
        "elementTypeBitWidth."
        "For element type having more than 32 bits, vnni shouldn't be used.\n");

  return ::mlir::success();
}

::mlir::ParseResult StoreScatterOp::parse(::mlir::OpAsmParser &parser,
                                          ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;

  mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  llvm::SMLoc valueOperandsLoc;

  mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> maskOperands(
      maskRawOperands);
  llvm::SMLoc maskOperandsLoc;

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);

  mlir::Type maskRawTypes[1];
  llvm::ArrayRef<mlir::Type> maskTypes(maskRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"mode", "l1_hint", "l2_hint", "l3_hint"}, true))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();

  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreScatterOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();
  if ((*this)->getAttrs().size()) {
    printer << ' ' << "{";
    printer << "mode = " << getMode();
    printCacheHintAttrs<StoreScatterOp>(printer, *this, true);
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

::mlir::LogicalResult StoreScatterOp::verify() {
  auto valueTy = getValue().getType();
  auto tdescTy = getTensorDesc().getType();
  auto maskTy = getMask().getType();

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

  if (tdescTy.getShape().vec() != maskShape || valueShape != maskShape) {
    return emitOpError(
        "Mask and value should have the same shape/size as TensorDesc."
        "Mask and Value can be scalar if TensorDesc is in form of "
        "TensorDesc<1xf16>.");
  }
  return ::mlir::success();
}

::mlir::LogicalResult UpdateOffsetOp::verify() {
  auto srcTy = getTensorDesc().getType();
  auto offTy = getOffsets().getType();
  auto resTy = getResult().getType();

  if (srcTy != resTy)
    return emitOpError(
        "The result should have the same type"
        "(shape and encoding attribute) as the input TensorDesc.");

  auto shape = srcTy.getShape();
  auto encoding = srcTy.getEncoding();

  if (!encoding || !llvm::isa<imex::xegpu::ScatteredAttr>(encoding)) {
    return emitOpError(
        "Invalid TensorDesc, it should have a scattered attribute.");
  }

  // For VC mode with chunkSize > 1. For chunkSize == 1, it is hard to
  // distinguish between VC and SIMT mode by only looking at updateOffsetOp
  // itself. So current verifier skipped these two cases.
  if (shape.size() == 2) {
    if (!llvm::isa<mlir::VectorType>(offTy))
      return emitOpError(
          "Based on TensorDesc shape, it is an VC tensor descriptor, "
          "in which the offset should be an 1D vector.");

    auto vecTy = llvm::dyn_cast<mlir::VectorType>(offTy);
    if (vecTy.getRank() != 1)
      return emitOpError("The index should be an 1D vector Type for VC mode "
                         "tensor descriptor.");

    if (shape[0] != vecTy.getShape()[0])
      return emitOpError("For VC Mode TensorDesc. The offset should have same"
                         "length as the dim-0 of TensorDesc.");
  }

  return ::mlir::success();
}

::mlir::LogicalResult UpdateNDOffsetOp::verify() {
  // number of offsets specified must match the rank of the tensor descriptor
  if (getTensorDesc().getType().getRank() != getOffsets().size()) {
    return emitOpError("invalid number of offsets.");
  }
  return ::mlir::success();
}
} // namespace xegpu
} // namespace imex

#include <imex/Dialect/XeGPU/IR/XeGPUOpsEnums.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/XeGPU/IR/XeGPUOps.cpp.inc>
