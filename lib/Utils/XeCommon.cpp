//===- XeCommon.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements some routines used by Xe related dialects.
///
//===----------------------------------------------------------------------===//
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <unordered_set>

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/XeCommon.h"
#include "llvm/Support/FormatVariadic.h"

namespace imex {

bool isColMajorOrder(mlir::DenseI32ArrayAttr order) {
  return (order == mlir::DenseI32ArrayAttr::get(order.getContext(), {0, 1}));
}

int getHeightForSLMBlock(llvm::ArrayRef<int64_t> shape, int width,
                         int vnniFactor, bool colMajor) {
  if (colMajor) {
    // for col-major, the scattered load/store will be used, and the width
    // will be mapped to simd lanes and height will be mapped to ChunkSize.
    for (auto h : getSupportedChunkSizes(width)) {
      h *= vnniFactor;
      if (shape[0] % h == 0)
        return h;
    }
  } else {
    // for row-major, the 1D block load/store will be used, the chunk size
    // is the whole block size, thus the height = chunkSize/width.
    for (auto chunk : getSupportedChunkSizes(1)) {
      auto h = chunk / width;
      h *= vnniFactor;
      if (chunk % width == 0 && h && shape[0] % h == 0)
        return h;
    }
  }
  return 0;
}

bool isSupportedOptimalSLMAccess(xetile::TileType tileTy) {
  // 1D load/store supports maximumly 64 elements, and scattered load/store
  // (used for transposed cases) supports maximumly 128 elements (16x8, since
  // we fixed block width to 16, which is mapped to simd16). For simplicity,
  // we start with simple cases, that can be evenly divided by the maximum
  // capacity of one instruction.

  const int width = 16;
  auto memSpace = tileTy.getMemorySpaceAsInt();
  auto shape = tileTy.getShape();
  auto vnni = getVnniFactor(tileTy.getElementType());

  if (memSpace == 3 && shape[1] % width == 0 && shape[0] % vnni == 0) {
    auto colMajor = isColMajorOrder(tileTy.getOrder());
    auto h = getHeightForSLMBlock(shape, width, vnni, colMajor);
    return h != 0;
  }
  return false;
}

static llvm::SmallVector<int64_t>
getVNNIShuffleIndices(mlir::VectorType srcType) {
  auto numElements = srcType.getNumElements();
  llvm::SmallVector<int64_t> ret(numElements, 0);
  auto dstType = getPackedType(srcType);
  auto dstShape = dstType.getShape();
  // Convert from contiguous layout to VNNI packed, e.g. from
  // `vector<16x16xf16>` to `vector<8x16x2xf16>`.
  // To arrange the data in VNNI format, the shuffle indices must satisfy
  // following mapping.
  // [i, j, k] => i * dstShape[1] * dstShape[2] + j + k * dstShape[1]
  int shuffleIndex = 0;
  for (unsigned i = 0; i < dstShape[0]; ++i) {
    for (unsigned j = 0; j < dstShape[1]; ++j) {
      for (unsigned k = 0; k < dstShape[2]; ++k) {
        ret[shuffleIndex++] =
            i * dstShape[1] * dstShape[2] + j + k * dstShape[1];
      }
    }
  }
  return ret;
}

int getVnniFactor(mlir::Type elemTy) {
  int vnni = 1;
  if (elemTy.isIntOrFloat())
    vnni = std::max<int>(32 / elemTy.getIntOrFloatBitWidth(), 1);
  return vnni;
}

mlir::VectorType getPackedType(mlir::VectorType vecTy) {
  auto shape = vecTy.getShape().vec();
  auto factor = getVnniFactor(vecTy.getElementType());
  unsigned axis = shape.size() == 3 ? 1 : 0;

  // Only 2D/3D vector supported and The vector size
  // must be divisible by the factor
  if ((shape.size() != 2 && shape.size() != 3) || !factor ||
      shape[axis] % factor != 0)
    return nullptr;

  shape.emplace_back(factor);
  shape[axis] /= factor;
  return mlir::VectorType::get(shape, vecTy.getElementType());
}

std::pair<mlir::Value, mlir::Operation *>
applyVnniTransform(mlir::OpBuilder &builder,
                   mlir::TypedValue<mlir::VectorType> src) {
  assert(src && "value must be non-null");
  auto loc = src.getLoc();
  auto srcTy = src.getType();
  auto elems = srcTy.getNumElements();
  auto elemTy = srcTy.getElementType();
  auto linearVecTy = mlir::VectorType::get(elems, elemTy);
  auto root = builder.create<mlir::vector::ShapeCastOp>(loc, linearVecTy, src);
  auto mask = getVNNIShuffleIndices(srcTy);
  auto shuffle = builder.create<mlir::vector::ShuffleOp>(loc, root, root, mask);
  auto packedTy = getPackedType(srcTy);
  auto cast = builder.create<mlir::vector::ShapeCastOp>(loc, packedTy, shuffle);
  // for convenience of load+transpose optimization, add packed attribute
  // to indicate these ops are used to do vnni transform.
  root.getOperation()->setAttr("packed", builder.getUnitAttr());
  shuffle.getOperation()->setAttr("packed", builder.getUnitAttr());
  cast.getOperation()->setAttr("packed", builder.getUnitAttr());

  return {cast, root};
}

llvm::SmallVector<int> getSupportedChunkSizes(int simdlanes) {
  if (simdlanes == 1)
    return {64, 32, 16, 8, 4, 3, 2, 1};
  return {8, 4, 3, 2, 1};
}

llvm::SmallVector<int64_t> getOperandIndices(mlir::Operation *op, mlir::Value operand) {
  llvm::SmallVector<int64_t> resIndices;
  for (auto [i, value] : llvm::enumerate(op->getOperands())) {
    if (operand == value)
      resIndices.push_back(i);
  }
  return resIndices;
}

int getResultIndex(mlir::Operation *op, mlir::Value value) {
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    if (result == value)
      return index;
  }
  return -1;
}

llvm::SmallVector<mlir::BlockArgument> getArgsForOperand(mlir::scf::ForOp &op,
                                     mlir::Value operand) {
  llvm::SmallVector<mlir::BlockArgument> argList;
  auto indices = getOperandIndices(op, operand);
  auto numControls = op.getNumControlOperands();
  for (auto idx: indices) {
    assert(idx >= static_cast<int>(numControls));
    argList.push_back(op.getRegionIterArg(idx - numControls));
  }
  return argList;
}

bool isSupportedModule(mlir::gpu::GPUModuleOp mod) {
  bool hasTileTyInFuncTy = false;
  mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::gpu::GPUFuncOp op) {
    auto funcTy = op.getFunctionType();
    hasTileTyInFuncTy |= std::any_of(
        funcTy.getInputs().begin(), funcTy.getInputs().end(),
        [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
    hasTileTyInFuncTy |= std::any_of(
        funcTy.getResults().begin(), funcTy.getResults().end(),
        [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
  });

  return hasTileTyInFuncTy == false;
}

mlir::ValueRange buildUnrealizedCast(mlir::OpBuilder &builder,
                                     mlir::TypeRange resultTypes,
                                     mlir::ValueRange inputs) {
  mlir::Location loc = builder.getUnknownLoc();
  if (!inputs.empty())
    loc = inputs.front().getLoc();
  auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, resultTypes, inputs);
  return castOp->getResults();
}

/// @brief
/// We have to use flattened i32 for intrinsic calls like llvm_genx_raw_send2_*,
/// hence we need to encode vectors with arbitrary datatypes as i32.
/// keepF16 = true: when the vector element type is f16, it disables flattening
/// it to i32.
std::pair<std::string, mlir::VectorType>
encodeVectorType(mlir::ConversionPatternRewriter &rewriter,
                 mlir::VectorType type, bool use64bitData, bool enforceInteger,
                 bool keepF16) {
  mlir::Type srcElemType = type.getElementType();
  assert((srcElemType.isF16() || srcElemType.isBF16() || srcElemType.isF32() ||
          srcElemType.isInteger(8) || srcElemType.isInteger(16) ||
          srcElemType.isInteger(32) || srcElemType.isInteger(64)) &&
         "Unsupported vector element type.");
  const uint32_t srcBitWidth = srcElemType.getIntOrFloatBitWidth();
  mlir::Type resElemType = rewriter.getI32Type();
  if (!enforceInteger) {
    if (srcElemType.isF32() || (keepF16 && srcElemType.isF16())) {
      resElemType = srcElemType;
    }
    if (use64bitData) {
      resElemType = rewriter.getI64Type();
    }
  }
  const uint32_t resBitWidth = resElemType.getIntOrFloatBitWidth();
  const uint32_t resVecSize =
      (type.getNumElements() * srcBitWidth) / resBitWidth;
  mlir::VectorType resVecType = mlir::VectorType::get(resVecSize, resElemType);
  std::string resStr =
      llvm::formatv("v{0}{1}{2}", resVecSize,
                    ((resElemType.isF32() || (keepF16 && resElemType.isF16()))
                         ? 'f'
                         : 'i'),
                    resBitWidth)
          .str();
  return {resStr, resVecType};
}

unsigned encodeDataum(mlir::Type type) {
  switch (type.getIntOrFloatBitWidth()) {
  case 8:
    return 1;
  case 16:
    return 2;
  case 32:
    return 3;
  case 64:
    return 4;
  default:
    assert(0 && "add more support");
    return 0;
  }
}

unsigned encodeOpcode(mlir::arith::AtomicRMWKind kind) {
  unsigned encode = 0;
  switch (kind) {
  case mlir::arith::AtomicRMWKind::addf:
    encode = 19;
    break;
  case mlir::arith::AtomicRMWKind::addi:
    encode = 12;
    break;
  case mlir::arith::AtomicRMWKind::assign:
    encode = 10;
    break;
  // case mlir::arith::AtomicRMWKind::maxf:
  //   encode = 22;
  //   break;
  case mlir::arith::AtomicRMWKind::maxs:
    encode = 15;
    break;
  case mlir::arith::AtomicRMWKind::maxu:
    encode = 17;
    break;
  // case mlir::arith::AtomicRMWKind::minf:
  //   encode = 21;
  //   break;
  case mlir::arith::AtomicRMWKind::mins:
    encode = 14;
    break;
  case mlir::arith::AtomicRMWKind::minu:
    encode = 16;
    break;
  case mlir::arith::AtomicRMWKind::ori:
    encode = 25;
    break;
  case mlir::arith::AtomicRMWKind::andi:
    encode = 24;
    break;
  case mlir::arith::AtomicRMWKind::mulf:
  case mlir::arith::AtomicRMWKind::muli:
    assert(0 && "Atomic operation not supported!");
    break;
  default:
    assert(0 && "to be supported");
    break;
  }
  return encode;
}

mlir::TypedValue<mlir::VectorType> stack(mlir::Value vecUp, mlir::Value vecDown,
                                         mlir::Location loc,
                                         mlir::OpBuilder &builder) {
  auto vecUpTy = llvm::cast<mlir::VectorType>(vecUp.getType());
  auto vecDownTy = llvm::cast<mlir::VectorType>(vecDown.getType());
  assert(vecUpTy.getRank() == 2 && vecDownTy.getRank() == vecUpTy.getRank() &&
         "only supports 2D vectors.");
  assert(vecUpTy.getShape()[1] == vecDownTy.getShape()[1] &&
         "Operands of stack() do not have the same number of columns.");

  llvm::SmallVector<int64_t> mask(vecUpTy.getShape()[0] +
                                  vecDownTy.getShape()[0]);
  std::iota(mask.begin(), mask.end(), 0);
  auto op = builder.create<mlir::vector::ShuffleOp>(loc, vecUp, vecDown, mask);
  return op;
}

// generate linearized shuffle mask for concat.
static llvm::SmallVector<int64_t>
getShuffleMask(llvm::ArrayRef<int64_t> shape1, llvm::ArrayRef<int64_t> shape2) {
  assert(shape1.size() == shape2.size() && shape1.size() <= 2 &&
         "only 1D/2D shape are supported.");
  assert(shape1.drop_back() == shape2.drop_back() &&
         "the row dim of the shapes should match.");
  int64_t size1 = std::accumulate(shape1.begin(), shape1.end(), 1,
                                  std::multiplies<int64_t>());
  int64_t size2 = std::accumulate(shape2.begin(), shape2.end(), 1,
                                  std::multiplies<int64_t>());
  llvm::SmallVector<int64_t> mask(size1 + size2);
  auto rows = shape1.size() == 1 ? 1 : shape1[0];
  auto cols1 = shape1.size() == 1 ? shape1[0] : shape1[1];
  auto cols2 = shape2.size() == 1 ? shape2[0] : shape2[1];
  for (int64_t i = 0; i < rows; i++) {
    int64_t s = i * (cols1 + cols2);
    int64_t m = s + cols1;
    int64_t e = m + cols2;
    int64_t v1 = i * cols1;
    int64_t v2 = size1 + i * cols2;
    std::iota(mask.begin() + s, mask.begin() + m, v1);
    std::iota(mask.begin() + m, mask.begin() + e, v2);
  }
  return mask;
}

mlir::TypedValue<mlir::VectorType> concat(mlir::Value lhs, mlir::Value rhs,
                                          mlir::Location loc,
                                          mlir::OpBuilder &builder) {
  auto lhsTy = llvm::cast<mlir::VectorType>(lhs.getType());
  auto rhsTy = llvm::cast<mlir::VectorType>(rhs.getType());

  assert(lhsTy.getShape()[0] == lhsTy.getShape()[0] &&
         "Operands of concat() do not have the same number of rows.");
  assert(lhsTy.getRank() <= 2 && rhsTy.getRank() == lhsTy.getRank() &&
         "Currently concat only works on 1D/2D vector.");

  auto elemTy = lhsTy.getElementType();
  auto leftSize = lhsTy.getNumElements();
  auto leftShape = lhsTy.getShape();
  auto leftFlatTy = mlir::VectorType::get({lhsTy.getNumElements()}, elemTy);

  auto rightSize = rhsTy.getNumElements();
  auto rightShape = rhsTy.getShape();
  auto rightFlatTy = mlir::VectorType::get({rhsTy.getNumElements()}, elemTy);

  auto newShape = lhsTy.getRank() == 1
                      ? llvm::SmallVector<int64_t>({leftSize + rightSize})
                      : llvm::SmallVector<int64_t>(
                            {leftShape[0], leftShape[1] + rightShape[1]});
  auto castLeft =
      builder.create<mlir::vector::ShapeCastOp>(loc, leftFlatTy, lhs);
  auto castRight =
      builder.create<mlir::vector::ShapeCastOp>(loc, rightFlatTy, rhs);
  auto mask = getShuffleMask(leftShape, rightShape);
  auto shuffleOp =
      builder.create<mlir::vector::ShuffleOp>(loc, castLeft, castRight, mask);
  auto targetTy = mlir::VectorType::get(newShape, elemTy);
  auto newOp =
      builder.create<mlir::vector::ShapeCastOp>(loc, targetTy, shuffleOp);
  return newOp;
}

// A wrapper function to merge small vectors into a big one. It takes a
// range of mlir::Value objects with mlir::VectorType, and merge them
// into a big vector using the provided transformation function.
mlir::Value packVectorsWith(mlir::ValueRange ins, PackFuncTy op,
                            mlir::Location loc, mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value> shuffleOps(ins.begin(), ins.end());
  while (shuffleOps.size() > 1) {
    auto curr = shuffleOps;
    shuffleOps.clear();
    size_t currPairStartIdx{0};
    while (currPairStartIdx < curr.size() - 1) {
      size_t leftIdx{currPairStartIdx++};
      size_t rightIdx{currPairStartIdx++};
      auto newOp = op(curr[leftIdx], curr[rightIdx], loc, builder);
      shuffleOps.push_back(newOp);
    }
    if (currPairStartIdx < curr.size()) {
      assert(currPairStartIdx == curr.size() - 1);
      shuffleOps.push_back(curr[curr.size() - 1]);
    }
  }
  return shuffleOps[0];
}

/// Checks if the given `type` is a 1-D vector type that requires VectorAnyINTEL
/// capability. In other words, the vector size is not supported by SPIR-V.
/// SPIR-V only supports 2, 3, 4, 8, 16 elements (8 and 16 with Vector16
/// capability).
bool isVectorAnyINTELType(mlir::Type type) {
  std::unordered_set<int64_t> spirvSupportedSizes = {2, 3, 4, 8, 16};
  auto vecType = mlir::dyn_cast<mlir::VectorType>(type);
  return vecType && vecType.getRank() == 1 &&
         (spirvSupportedSizes.find(vecType.getNumElements()) ==
          spirvSupportedSizes.end());
}

// convert OpFoldResult to Value by replacing integer
// attributes with arith::ConstantOps. It also performs
// simple type conversions
mlir::Value getValueOrConstantOp(mlir::OpFoldResult ofr, mlir::Location loc,
                                 mlir::PatternRewriter &rewriter,
                                 mlir::Type type) {
  if (ofr.is<mlir::Value>())
    return ofr.get<mlir::Value>();

  auto intAttr = llvm::cast<mlir::IntegerAttr>(ofr.get<mlir::Attribute>());

  if (type)
    intAttr = mlir::IntegerAttr::get(type, intAttr.getInt());

  return rewriter.create<mlir::arith::ConstantOp>(loc, intAttr);
}

llvm::SmallVector<mlir::Value> getStridesOrOffsetsOrShapesInValueType(
    mlir::PatternRewriter &rewriter,
    ::llvm::SmallVector<mlir::OpFoldResult> mixedOSS, mlir::Location loc) {
  llvm::SmallVector<mlir::Value> valueVec;
  // auto mixedStrides = op.getMixedStrides();
  for (size_t i = 0; i < mixedOSS.size(); i++) {
    auto oss = getValueOrConstantOp(mixedOSS[i], loc, rewriter,
                                    rewriter.getIndexType());
    valueVec.push_back(oss);
  }
  return valueVec;
}

} // namespace imex
