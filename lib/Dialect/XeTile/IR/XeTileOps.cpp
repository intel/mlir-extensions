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
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>

#define DEBUG_TYPE "xetile-ops"

namespace imex {
namespace xetile {

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

// Helper to check if given OpFoldResult is a constant.
static bool isConstantIndex(mlir::OpFoldResult value) {
  // If the value is an attribute, then it is a constant.
  if (llvm::isa<mlir::Attribute>(value))
    return true;
  return llvm::cast<mlir::Value>(value)
             .getDefiningOp<mlir::arith::ConstantOp>() != nullptr;
}

// check whether the given shape can be perfectly distributed to each subgroup.
// If so return true, otherwise return false.
static bool isEvenDistributed(llvm::ArrayRef<int64_t> shape,
                              xetile::WorkGroupMapAttr attr) {
  assert(attr && "workgroup map attribute is missing.");
  auto data = attr.getSgData().asArrayRef();
  auto layout = attr.getSgLayout().asArrayRef();
  for (auto [s, d, l] : llvm::zip_equal(shape, data, layout)) {
    // check s % (d * l) != 0
    if (s % d != 0 || (s / d) % l != 0)
      return false;
  }
  return true;
}

mlir::LogicalResult InitTileOp::verify() {
  auto tileTy = getType();

  // Check for memory space validity.
  if (getSourceMemorySpaceAsInt() !=
      static_cast<unsigned int>(tileTy.getMemorySpaceAsInt()))
    return emitOpError(
        "memory space of the tile doesn't match with the source.");

  // for scattered TileType
  if (auto indices = getIndices()) {
    auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSourceType());
    if (!srcTy || srcTy.getRank() > 1)
      return emitOpError("Expecting a 0-D or 1-D memref as source.");

    if (!tileTy.getScatterAttr())
      return emitOpError("Expecting a scattered TileType.");

    // TODO: temoprary disable it in favor of 4D representation of
    // blocking pass
    // if (tileTy.getShape() != indices.getType().getShape())
    //   return emitOpError("Shape mismatch between indices and result tile.");

    return mlir::success();
  }

  // for block TileType

  // If the source is a memref and has static shape, then size and stride
  // arguments must not be present.
  if (isSourceMemRef() && sourceMemRefHasStaticShape() && hasSizeArgs())
    return emitOpError("dynamic sizes are not allowed with a static "
                       "shaped memref as source");

  // Checks that memory access parameters are of valid rank.
  if ((getMixedSizes().size() != getMixedStrides().size()) ||
      (getMixedStrides().size() != getMixedOffsets().size()) ||
      (static_cast<int64_t>(getMixedSizes().size()) < tileTy.getRank()))
    return emitOpError(
        "memref with a dynamic shape or raw address is used as source but "
        "dynamic shape argument missing or it is not of valid rank");

  auto order = tileTy.getOrder();
  bool rowMajor = (order[0] == 1 && order[1] == 0);

  if (isSourceMemRef() && sourceMemRefHasStaticShape()) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(getSourceType());

    // Checks for memrefs with format:
    // clang-format off
    // memref<[shape], strided<[strides], offsets:[offset]>>
    // clang-format on
    llvm::SmallVector<int64_t, 4> strides;
    auto shape = getSourceMemrefStaticShape();
    int64_t offset;
    if (mlir::succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      int64_t rank = memrefType.getRank();
      if (rowMajor &&
          !((strides[rank - 2] == shape[rank - 1]) && (strides[rank - 1] == 1)))
        return emitOpError(
            "memref operand is expected to have a row-major layout");

      if (!rowMajor &&
          !((strides[rank - 2] == 1) && (strides[rank - 1] == shape[rank - 2])))
        return emitOpError(
            "memref operand is expected to have a column-major layout");
      return mlir::success();
    }

    // Checks for memrefs with affine maps :
    // clang-format off
    // memref<[shape], affine_map<(d0, d1) -> (d1, d0)>>
    // clang-format on
    if (rowMajor && !(memrefType.getLayout().isIdentity())) {
      // No affine map means it's using the default row-major layout
      return emitOpError(
          "memref operand is expected to have a row-major layout");
    }
    if (!rowMajor) {
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
    auto dynamicShape = getMixedSizes();
    auto dynamicStrides = getMixedStrides();

    if (dynamicShape.size() == 0 || dynamicStrides.size() == 0) {
      return emitOpError("dynamic shape and strides must not be empty");
    }

    // Check if all shape and stride values are constant.
    if (!llvm::all_of(dynamicShape, isConstantIndex) ||
        !llvm::all_of(dynamicStrides, isConstantIndex)) {
      LLVM_DEBUG(llvm::dbgs() << "Assuming user has verified the layout\n");
      return mlir::success();
    }

    auto shapeDim1 = getConstantIntValue(dynamicShape[1]).value();
    auto strideDim0 = getConstantIntValue(dynamicStrides[0]).value();
    auto strideDim1 = getConstantIntValue(dynamicStrides[1]).value();

    // checks for layouts where source is not memref and just an address
    if (rowMajor && (strideDim0 == 1 && strideDim1 == shapeDim1)) {
      return emitOpError(
          "memref operand is expected to have a row-major layout");
    }

    if (!rowMajor && !(strideDim0 == 1 && strideDim1 == shapeDim1)) {
      return emitOpError(
          "memref operand is expected to have a column-major layout");
    }
  } else if (isSourceMemRef() && !sourceMemRefHasStaticShape())
    LLVM_DEBUG(llvm::dbgs() << "Assuming user has verified the layout\n");

  return mlir::success();
}

void InitTileOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       xetile::TileType resultType, mlir::Value source,
                       llvm::ArrayRef<mlir::OpFoldResult> offsets) {
  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, resultType, source, dynamicOffsets,
        mlir::ValueRange({}), /* empty dynamic sizes*/
        mlir::ValueRange({}), /* empty dynamic strides*/
        mlir::DenseI64ArrayAttr::get(builder.getContext(),
                                     staticOffsets), /* static offsets*/
        mlir::DenseI64ArrayAttr(),                   /* empty sttaic sizes*/
        mlir::DenseI64ArrayAttr(),                   /* empty static strides*/
        nullptr);
}

void InitTileOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       xetile::TileType resultType, mlir::Value source,
                       llvm::ArrayRef<mlir::OpFoldResult> offsets,
                       llvm::ArrayRef<mlir::OpFoldResult> sizes,
                       llvm::ArrayRef<mlir::OpFoldResult> strides) {
  llvm::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  llvm::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  build(builder, state, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides,
        mlir::DenseI64ArrayAttr::get(builder.getContext(), staticOffsets),
        mlir::DenseI64ArrayAttr::get(builder.getContext(), staticSizes),
        mlir::DenseI64ArrayAttr::get(builder.getContext(), staticStrides),
        nullptr);
}

void InitTileOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       xetile::TileType resultType,
                       mlir::TypedValue<mlir::MemRefType> source,
                       mlir::TypedValue<mlir::VectorType> indices) {
  auto type = source.getType();
  assert(type.getRank() <= 1 && "source must be a 1D memref.");
  build(builder, state, resultType, source, {} /* offsets */, {} /* sizes */,
        {} /*strides */, {} /* static offsets*/, {} /* static sizes*/,
        {} /* static strides */, indices);
}

mlir::LogicalResult StoreTileOp::verify() {
  auto tileTy = getTile().getType();
  auto attr = tileTy.getWgMap();
  if (attr && !isEvenDistributed(tileTy.getShape(), attr))
    return emitOpError("data is not evenly distributed to each subgroup.");
  return mlir::success();
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

  auto checkMmaShapes = [](llvm::ArrayRef<int64_t> &A,
                           llvm::ArrayRef<int64_t> &B,
                           llvm::ArrayRef<int64_t> &Out) -> bool {
    return A[1] == B[0] && Out[0] == A[0] && Out[1] == B[1];
  };

  // check mma shape
  if (aRank == 2 && !checkMmaShapes(aShape, bShape, outShape))
    return emitOpError(
        "incompatible A, B and output sizes for 2D tile mma op. "
        "2D tile mma should have the shape (m x k) x (k x n) = (m x n).");

  // optional input C must have the same shape as output
  if (getC() &&
      llvm::cast<mlir::VectorType>(getC().getType()).getShape() != outShape)
    return emitOpError("input C must have the same shape as output.");

  return mlir::success();
}

mlir::LogicalResult TransposeOp::verify() {
  auto srcShape = getVector().getType().getShape();
  auto resShape = getResult().getType().getShape();
  auto permutation = getPermutation();

  auto transposeShape = srcShape.vec();
  for (auto [i, j] : llvm::enumerate(permutation)) {
    if (j >= static_cast<int64_t>(srcShape.size()))
      return emitOpError("permutation index out of bounds");
    transposeShape[i] = srcShape[j];
  }

  if (transposeShape != resShape.vec())
    return emitOpError("Incorrect transpose permutation");

  return mlir::success();
}

mlir::LogicalResult ReductionOp::verify() {
  auto dims = getReductionDims();
  auto resShape = getResult().getType().getShape();
  if (getReductionSize() > 0) {
    if (dims.size() > 1)
      // When reduction size is specified,
      // only a single dimension can be reduced.
      return emitOpError(
          "when reduction size is specified, only a single reduction "
          "dimension is allowed.");
    auto srcTy = getSource().getType();
    if (srcTy.getRank() != 2)
      return emitOpError(
          "when reduction size is specified, source must be a 2D vector.");
    auto redDim = dims.front();
    if (resShape[redDim] !=
        srcTy.getShape()[redDim] / static_cast<int64_t>(getReductionSize()))
      return emitOpError(
          "reduction size does not match the expected size of the result "
          "dimension after reduction.");
  } else {
    for (auto i : dims)
      if (resShape[i] != 1)
        return emitOpError("reduction dimension of result must have size 1");
  }
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

mlir::LogicalResult ConvertLayoutOp::verify() {
  auto attr = getWgMapSourceAttr();
  auto shape = getSource().getType().getShape();
  if (attr && !isEvenDistributed(shape, attr))
    return emitOpError("data is not evenly distributed to each subgroup.");
  return mlir::success();
}

} // namespace xetile
} // namespace imex

#include <imex/Dialect/XeTile/IR/XeTileOpsEnums.cpp.inc>
#define GET_OP_CLASSES
using namespace mlir;
#include <imex/Dialect/XeTile/IR/XeTileOps.cpp.inc>
