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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace imex {
namespace xetile {

bool TileBase::hasRank() const { return true; }

llvm::ArrayRef<int64_t> TileBase::getShape() const {
  return cast<TileType>().getShape();
}

TileBase TileBase::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                             Type elementType) const {
  return TileType::get(shape.value_or(getShape()), elementType);
}

static mlir::LogicalResult parseShape(mlir::AsmParser &parser,
                                      llvm::SmallVector<int64_t> &shape,
                                      mlir::Type &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);
  return mlir::success();
}

static void printShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                       mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
}

mlir::LogicalResult InitTileOp::verify() {

  // number of offsets must be 2 because init_tile creates 2D tiles
  // dynamic_offsets is always a subset of offsets, so checking this is
  // sufficient
  if (getStaticOffsets().size() != 2) {
    return emitOpError("number of offsets must be 2");
  }

  // if the source is a memref and has static shape, then dynamic shape and
  // strides arguments must not be present
  if (isSourceMemRef() && sourceMemRefHasStaticShape() &&
      (hasDynamicStrides() || hasDynamicShape())) {
    return emitOpError("dynamic shape or strides are not allowed with a static "
                       "shaped memref as source");
  }

  // if the source is a memref with dynamic shape, then a 2D dynamic shape
  // argument must be present
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getDynamicShape().size() != 2) {
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic shape argument missing or it is not 2D");
  }

  // if the source is a memref with dynamic shape, then a 2D dynamic strides
  // argument must be present
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getDynamicStrides().size() != 2) {
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic strides argument missing or it is not 2D");
  }

  // if the source is an address, the dynamic shape must be 2D
  if (isSourceInteger() && getDynamicShape().size() != 2) {
    return emitOpError("address is used as source but dynamic shape argument "
                       "is missing or it is not 2D");
  }

  // if the source is an address, dynamic strides must be 2D
  if (isSourceInteger() && getDynamicStrides().size() != 2) {
    return emitOpError("address is used as source but dynamic strides argument "
                       "is missing or it is not 2D");
  }

  return mlir::success();
}

void InitTileOp::build(::mlir::OpBuilder &builder,
                       ::mlir::OperationState &state,
                       xetile::TileType resultType, ::mlir::Value source,
                       ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
                       ::llvm::ArrayRef<::mlir::NamedAttribute> attrs) {
  ::llvm::SmallVector<int64_t> staticOffsets;
  ::llvm::SmallVector<::mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, resultType, source, dynamicOffsets, staticOffsets,
        ::mlir::ValueRange({}),  /* empty dynamic shape*/
        ::mlir::ValueRange({})); /* empty dynamic strides*/
  state.addAttributes(attrs);
}

void InitTileOp::build(::mlir::OpBuilder &builder,
                       ::mlir::OperationState &state,
                       xetile::TileType resultType, ::mlir::Value source,
                       ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
                       ::llvm::ArrayRef<::mlir::Value> dynamic_shape,
                       ::llvm::ArrayRef<::mlir::Value> dynamic_strides,
                       ::llvm::ArrayRef<::mlir::NamedAttribute> attrs) {
  ::llvm::SmallVector<int64_t> staticOffsets;
  ::llvm::SmallVector<::mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, resultType, source, dynamicOffsets, staticOffsets,
        dynamic_shape, dynamic_strides);
  state.addAttributes(attrs);
}

mlir::LogicalResult LoadTileOp::verify() {
  int64_t outputRank =
      llvm::cast<mlir::VectorType>(getValue().getType()).getRank();

  auto innerBlocks = getInnerBlocksAttr();
  auto transpose = getTransposeAttr();

  // if load_tile operation in blocked format only support 2D blocks
  if (innerBlocks && innerBlocks.size() != 2) {
    return emitOpError("inner_blocks must be two dimensional");
  }

  // if blocked load_tile load is specified output must be 4-dimensional
  if (innerBlocks && outputRank != 4) {
    return emitOpError(
        "output must be 4-dimensional if inner_blocks is specified");
  }

  if (transpose && transpose.size() != 2) {
    return emitOpError("transpose must be two dimensional");
  }

  return mlir::success();
}

mlir::LogicalResult TileMMAOp::verify() {
  int64_t aRank = getAType().getRank();
  int64_t bRank = getBType().getRank();

  mlir::Type aElemType = getAType().getElementType();
  mlir::Type bElemType = getBType().getElementType();

  // the two vector inputs to tile mma must have same rank
  if (aRank != bRank)
    return emitOpError("rank mismatch in tile mma inputs");

  // the two vector inputs must have the same element type
  if (aElemType != bElemType)
    return emitOpError("element type mismatch in tile mma inputs");

  return mlir::success();
}

void XeTileDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/XeTile/IR/XeTileOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/XeTile/IR/XeTileOps.cpp.inc>
      >();
}

} // namespace xetile
} // namespace imex

#include <imex/Dialect/XeTile/IR/XeTileOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOps.cpp.inc>
