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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
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
  if (value.is<mlir::Attribute>())
    return true;
  return value.get<mlir::Value>().getDefiningOp<mlir::arith::ConstantOp>() !=
         nullptr;
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

    if (tileTy.getShape() != indices.getType().getShape())
      return emitOpError("Shape mismatch between indices and result tile.");

    return mlir::success();
  }

  // for block TileType

  // If the source is a memref and has static shape, then size and stride
  // arguments must not be present.
  if (isSourceMemRef() && sourceMemRefHasStaticShape() && hasSizeArgs())
    return emitOpError("dynamic sizes are not allowed with a static "
                       "shaped memref as source");

  // If the source is a memref with dynamic sizes, then dynamic size
  // arguments must be present.
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getMixedSizes().size() != 2)
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic shape argument missing or it is not 2D");

  // If the source is a memref with dynamic sizes, then a dynamic stride
  // arguments must be present.
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getMixedStrides().size() != 2)
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic strides argument missing or it is not 2D");

  // if the source is an address, the dynamic sizes must be 2D
  if (isSourceInteger() && getMixedSizes().size() != 2)
    return emitOpError("address is used as source but dynamic shape argument "
                       "is missing or it is not 2D");

  // if the source is an address, dynamic strides must be 2D
  if (isSourceInteger() && getMixedStrides().size() != 2)
    return emitOpError("address is used as source but dynamic strides argument "
                       "is missing or it is not 2D");

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
    if (mlir::succeeded(
            mlir::getStridesAndOffset(memrefType, strides, offset))) {
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
      llvm::dbgs() << "Assuming user has verified the layout\n";
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
    llvm::dbgs() << "Assuming user has verified the layout\n";

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
  auto vecShape = getResult().getType().getShape();

  // inner_blocks may or maynot be present in this op.
  auto innerBlocks = mlir::DenseI64ArrayAttr();
  if (encoding)
    innerBlocks = mlir::dyn_cast<xetile::XeTileAttr>(encoding).getInnerBlocks();

  // if inner_blocks is not present in the tile_attr, the output of the load
  // must be 2D and tile shape and vector output shape must match
  if (innerBlocks == mlir::DenseI64ArrayAttr())
    if (!vecShape.equals(tileShape))
      return emitOpError("Output shape must match the tile shape.");

  if (innerBlocks != mlir::DenseI64ArrayAttr() && innerBlocks.size() > 0) {
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

  if (innerBlocks != mlir::DenseI32ArrayAttr() && innerBlocks.size() > 0) {
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
using namespace mlir;
#include <imex/Dialect/XeTile/IR/XeTileOps.cpp.inc>
