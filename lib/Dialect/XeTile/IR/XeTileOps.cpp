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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
  auto baseTy = getBaseType();

  // base memref must either have a static shape or a strided layout
  // otherwise, we can not get the shape info to create the tile
  auto stridedLayout =
      ::llvm::dyn_cast<::mlir::StridedLayoutAttr>(baseTy.getLayout());
  if (!baseTy.hasStaticShape() && !stridedLayout) {
    return emitOpError("base memref does not have a static shape or stride "
                       "layout information.");
  }

  // offsets must be 2D.
  int numOffsets = getNumOfStaticOffsets() + getOffsets().size();
  if (numOffsets != 2) {
    return emitOpError("offsets of the init_tile must be 2D.");
  }

  return mlir::success();
}

mlir::LogicalResult LoadTileOp::verify() {
  int64_t outputRank =
      llvm::cast<mlir::VectorType>(getResult().getType()).getRank();

  auto innerBlocks = getInnerBlocksAttr();

  // if load_tile operation in blocked format only support 2D blocks
  if (innerBlocks && innerBlocks.size() != 2) {
    return emitOpError("inner_blocks must be two dimensional if specified");
  }

  // if blocked load_tile load is specified output must be 4-dimensional
  if (innerBlocks && outputRank != 4) {
    return emitOpError(
        "output must be 4-dimensional if inner_blocks is specified");
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
