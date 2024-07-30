//===- XeTileOpConversion.h - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements ConversionPatterns for XeTileOps, used in XeTileToXeGPU
/// conversion, converting the XeTile dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#include "XeTileOpConversion.h"
#include "ArithOpConversion.h"
#include "SCFOpConversion.h"
#include "imex/Utils/XeArch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include <algorithm>
#include <cassert>
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h>
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

namespace imex {

using mlir::vector::ExtractOp;
using mlir::vector::ExtractStridedSliceOp;
using mlir::vector::ShapeCastOp;
using mlir::vector::ShuffleOp;

using VectorTypedValue = mlir::TypedValue<mlir::VectorType>;
using funcTy = VectorTypedValue(mlir::Value, mlir::Value, mlir::Location,
                                mlir::PatternRewriter &);

// Combine vectors vertically while keeping the logical data layout.
// As an example, given two vectors (2x4xf16) p and q, it will merge
// them in to a 4x4xf16 vector.
//  p1, p2, p3, p4            p1, p2, p3, p4
//  p5, p6, p7, p8            p5, p6, p7, p8
//                     ==>    q1, q2, q3, q4
//  q1, q2, q3, q4            q5, q6, q7, q8
//  q5, q6, q7, q8
static VectorTypedValue stack(mlir::Value v1, mlir::Value v2,
                              mlir::Location loc,
                              mlir::PatternRewriter &rewriter) {
  // LLVM requires operands of a shuffle op has the same type.
  assert(v1.getType() == v2.getType() &&
         "Operands of shuffle should have the same type.");
  auto vecTy = llvm::cast<mlir::VectorType>(v1.getType());
  assert(vecTy.getRank() == 2 && "only supports 2D vectors.");
  auto shape = vecTy.getShape();
  llvm::SmallVector<int64_t> mask(shape[0] + shape[0]);
  std::iota(mask.begin(), mask.end(), 0);
  auto op = rewriter.create<ShuffleOp>(loc, v1, v2, mask);
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

// merge vectors horizontally while keep the logical data layout.
// 1 2 3 4   +    10 11 12   =   1 2 3 4 10 11 12
// 5 6 7 8        13 14 15       5 6 7 8 13 14 15
// since there is no direct op in mlir exists, we will
// using ShapeCast and Shuffle to mimic it. It comes with
// cost of complex shuffle masks. the mask for the above one
// will be like this: 0 1 2 3  8  9 10
//                    4 5 6 7 11 12 13
VectorTypedValue concat(mlir::Value v1, mlir::Value v2, mlir::Location loc,
                        mlir::PatternRewriter &rewriter) {
  // LLVM requires operands of shuffle op has the same type
  auto vecTy = llvm::cast<mlir::VectorType>(v1.getType());
  assert(v1.getType() == v2.getType() &&
         "Operands doesn't have the same type!");
  assert(vecTy.getRank() <= 2 &&
         "Currently concat only works on 1D/2D vector.");
  auto size = vecTy.getNumElements();
  auto shape = vecTy.getShape();
  auto newShape = vecTy.getRank() == 1
                      ? llvm::SmallVector<int64_t>({size * 2})
                      : llvm::SmallVector<int64_t>({shape[0], shape[1] * 2});
  auto elemTy = vecTy.getElementType();
  auto flatTy = mlir::VectorType::get({size}, elemTy);
  auto cast1 = rewriter.create<ShapeCastOp>(loc, flatTy, v1);
  auto cast2 = rewriter.create<ShapeCastOp>(loc, flatTy, v2);
  auto mask = getShuffleMask(shape, shape);
  auto shuffleOp = rewriter.create<ShuffleOp>(loc, cast1, cast2, mask);
  auto targetTy = mlir::VectorType::get(newShape, elemTy);
  auto newOp = rewriter.create<ShapeCastOp>(loc, targetTy, shuffleOp);
  return newOp;
}

// A wrapper function to merge small vectors into a big one. It takes a
// range of mlir::Value objects with mlir::VectorType, and merge them
// into a big vector using the provided transformation function.
mlir::Value mergeVectorsWrapper(mlir::ValueRange ins,
                                std::function<funcTy> transFunc,
                                mlir::Location loc,
                                XeOneToNPatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Value> shuffleOps(ins.begin(), ins.end());
  while (shuffleOps.size() > 1) {
    auto curr = shuffleOps;
    assert(curr.size() % 2 == 0 && "The size should be divisible by 2.");
    shuffleOps.clear();
    for (size_t i = 0; i + 1 < curr.size(); i += 2) {
      auto newOp = transFunc(curr[i], curr[i + 1], loc, rewriter);
      shuffleOps.push_back(newOp);
    }
  }
  return shuffleOps[0];
}

// a unified function lowering Unpack and Pack ops.
static llvm::SmallVector<mlir::Value>
lowerUnpackOrPack(XeOneToNPatternRewriter &rewriter, mlir::Operation *op,
                  mlir::ValueRange inputs, mlir::DenseI64ArrayAttr inBlkSizes,
                  mlir::DenseI64ArrayAttr outBlkSizes,
                  llvm::ArrayRef<int64_t> inGrids,
                  llvm::ArrayRef<int64_t> outGrids, bool isVnniFormat = false,
                  bool isForDPASB = false) {

  // handle based on the dim0, and save results into intermediates
  llvm::SmallVector<mlir::Value> intermediates(outGrids[0] * inGrids[1]);
  if (inBlkSizes[0] == outBlkSizes[0]) { // do nothing
    intermediates = inputs;
  } else if (inBlkSizes[0] < outBlkSizes[0]) { // stack on dim 0
    // `nums` small vectors will be stacked into one big vector
    auto nums = inGrids[0] / outGrids[0];
    llvm::SmallVector<mlir::Value> valSet;
    for (auto j = 0; j < inGrids[1]; j++) {
      for (auto i = 0; i < inGrids[0]; i++) {
        auto idx = i * inGrids[1] + j;
        valSet.push_back(inputs[idx]);
        if (valSet.size() == (size_t)nums) {
          auto newOp =
              mergeVectorsWrapper(valSet, stack, op->getLoc(), rewriter);
          intermediates[i / nums * inGrids[1] + j] = newOp;
          valSet.clear();
        }
      }
    }
  } else { // do extract on dim0 using vector::ExtractStridedSliceOp
    // intermediates.resize(outGrids[0] * inGrids[1]);
    llvm::SmallVector<int64_t> blkSizes({outBlkSizes[0], inBlkSizes[1]});
    // if the vnni transform applied, vector shape
    // and offset need to be adjusted accordingly.
    if (isVnniFormat) {
      auto vnniAxis = isForDPASB ? 0 : 1;
      auto factor = mlir::cast<mlir::VectorType>(inputs.front().getType())
                        .getShape()
                        .back();
      blkSizes[vnniAxis] /= factor;
    }
    // each vector will be horizonally cut into `nums` subvectors
    auto nums = outGrids[0] / inGrids[0];
    llvm::SmallVector<int64_t> strides({1, 1});
    for (auto i = 0; i < inGrids[0]; i++) {
      for (auto j = 0; j < inGrids[1]; j++) {
        auto startPos = i * nums * inGrids[1] + j;
        auto v = inputs[i * inGrids[1] + j];
        for (auto k = 0; k < nums; k++) {
          llvm::SmallVector<int64_t> offsets({k * blkSizes[0], 0});
          auto newOp = rewriter.create<ExtractStridedSliceOp>(
              op->getLoc(), v, offsets, blkSizes, strides);
          auto idx = startPos + k * inGrids[1];
          intermediates[idx] = newOp;
        }
      }
    }
  }

  // handle intermediates based on the dim1, and save results into newOps
  llvm::SmallVector<mlir::Value> newOps;
  llvm::SmallVector<int64_t> interGrids = {outGrids[0], inGrids[1]};
  if (inBlkSizes[1] == outBlkSizes[1]) {
    // do nothing since they have the same size
    newOps = intermediates;
  } else if (inBlkSizes[1] < outBlkSizes[1]) {
    // doing concat since blkSZ of input vector is smaller
    // `nums` of small vectors will be concated into a big one
    size_t nums = inGrids[1] / outGrids[1];
    llvm::SmallVector<mlir::Value> valSet;
    for (auto i = 0; i < interGrids[0]; i++) {
      for (auto j = 0; j < interGrids[1]; j++) {
        valSet.push_back(intermediates[i * interGrids[1] + j]);
        if (valSet.size() == nums) {
          auto newOp =
              mergeVectorsWrapper(valSet, concat, op->getLoc(), rewriter);
          newOps.push_back(newOp);
          valSet.clear();
        }
      }
    }
  } else { // doing extract on dim 1
    llvm::SmallVector<int64_t> blkSizes({outBlkSizes[0], outBlkSizes[1]});
    // if vnni transform applied, vector shape
    // and offset needs to adjusted accordingly.
    if (isVnniFormat) {
      auto vnniAxis = isForDPASB ? 0 : 1;
      auto factor = mlir::cast<mlir::VectorType>(inputs.front().getType())
                        .getShape()
                        .back();
      blkSizes[vnniAxis] /= factor;
    }
    llvm::SmallVector<int64_t> strides({1, 1});
    auto nums = outGrids[1] / interGrids[1];
    for (auto i = 0; i < interGrids[0]; i++) {
      for (auto j = 0; j < interGrids[1]; j++) {
        auto v = intermediates[i * interGrids[1] + j];
        for (int64_t k = 0; k < nums; k++) {
          llvm::SmallVector<int64_t> offsets({0, k * blkSizes[1]});
          auto newOp = rewriter.create<ExtractStridedSliceOp>(
              op->getLoc(), v, offsets, blkSizes, strides);
          newOps.push_back(newOp);
        }
      }
    }
  }

  return newOps;
}

// It lowers a pair of Unpack and Pack operators at a time.
// the pattern first matchs TileUnpackOp, and finds its TilePackOp
// user. It can avoid some vector shuffle and extract ops by
// looking at the target block size (innerBlock from TilePackOp)
// directly. It requires 1-1 mapping of UnpackOp and PackOp, which
// should be enforced by a separate pass.
class SgTileUnpackOpPattern : public XeOneToNConversion<xetile::TileUnpackOp> {
  using XeOneToNConversion<xetile::TileUnpackOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::TileUnpackOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

    auto inputs = adaptor.getInVec();
    auto inTy = op.getInVec().getType();
    auto inGrids = inTy.getShape().take_front(2);
    auto inBlkSizes = op.getInnerBlocksAttr();

    // specific attention needed for vectors in vnni format,
    // which is applied to load for dpas.
    auto loadOp = op.getInVec().getDefiningOp<xetile::LoadTileOp>();
    auto elemTy = op.getInVec().getType().getElementType();
    bool isDpasB = loadOp && isForDPASB(loadOp);
    bool isVnniFormat =
        isDpasB && elemTy.isIntOrFloat() && elemTy.getIntOrFloatBitWidth() < 32;

    // the default grids used as outGrids when unpack is not paired with a pack
    int64_t defautlOutGrids[2] = {1, 1};
    llvm::ArrayRef<int64_t> outGrids;
    mlir::DenseI64ArrayAttr outBlkSizes;
    auto packOp = llvm::dyn_cast<xetile::TilePackOp>(*(op->user_begin()));
    if (op->hasOneUse() && packOp) { // lower the Unpack and Pack pair
      auto outTy = packOp.getOutVec().getType();
      outGrids = outTy.getShape().take_front(2);
      outBlkSizes = packOp.getInnerBlocksAttr();
    } else { // lower the Unpack only
      auto outTy = op.getOutVec().getType();
      outGrids = llvm::ArrayRef<int64_t>(defautlOutGrids, 2);
      auto ctx = op.getContext();
      outBlkSizes = mlir::DenseI64ArrayAttr::get(ctx, outTy.getShape());
    }

    // TODO: logically it is to do concat, but the data is in vnni format
    // which breaks the concat logic, it transforms concat into stack.
    if (isVnniFormat && (inBlkSizes[1] < outBlkSizes[1])) {
      return op->emitOpError("[Unexpected rare case]: ")
             << "It rarly happens that we need to do concat on vnni "
             << "transformed vectors (which is 3D instead of 2D). "
             << "It is essentially a stack on the 2nd dim, and is "
             << "not implemented yet.\n";
    }

    rewriter.setInsertionPoint(op);
    auto newOps =
        lowerUnpackOrPack(rewriter, op, inputs, inBlkSizes, outBlkSizes,
                          inGrids, outGrids, isVnniFormat, isDpasB);

    if (op->hasOneUse() && packOp) { // lowered Unpack and Pack as pair
      rewriter.replaceOp(packOp, newOps);
      rewriter.eraseOp(op);
    } else { // lowering unpack only
      rewriter.replaceOp(op, newOps);
    }
    return mlir::success();
  }
};

class SgTilePackOpPattern : public XeOneToNConversion<xetile::TilePackOp> {
  using XeOneToNConversion<xetile::TilePackOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::TilePackOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto input = op.getInVec();
    auto defOp = input.getDefiningOp<xetile::TileUnpackOp>();
    // Unpack and Pack appeared as a pair, it should be handled
    // by UnpackOpPattern in this case.
    if (defOp && defOp->hasOneUse())
      return mlir::failure();

    auto inTy = op.getInVec().getType();
    auto inGrids = llvm::SmallVector<int64_t>({1, 1});
    auto inBlkSizes =
        mlir::DenseI64ArrayAttr::get(op.getContext(), inTy.getShape());

    auto outTy = op.getOutVec().getType();
    auto outGrids = outTy.getShape().take_front(2);
    auto outBlkSizes = op.getInnerBlocksAttr();

    auto newOps = lowerUnpackOrPack(rewriter, op, {input}, inBlkSizes,
                                    outBlkSizes, inGrids, outGrids);

    // it is simple one-to-one mapping
    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

// A helper to compute the right array length given the inner block width,
// and the tile width, as well as the element type. Both inner block width
// and tile width are in number of elements. It is computed based on hardware
// constraints (on PVC): array_lenght * inner_block_width * sizeof(elemTy) <=
// 256 bits. So, if tile width is larger than 256/sizeof(elemTy), the maximum
// supported array_length will be used.
int getBlockArrayLength(mlir::Operation *op, mlir::Type elemTy,
                        int inner_block_width, int tile_width) {
  auto uArch = std::make_shared<XePVCuArch>();
  auto elemBits = elemTy.getIntOrFloatBitWidth();
  auto params = uArch->get2DLoadConfig(op, elemBits, false, false);
  assert(mlir::succeeded(params) && "Invalid Config Params");

  llvm::SmallVector<int> supportedArrLen = params->array_length;
  int restriction = std::min(params->restriction, tile_width);

  int result = 1;
  for (auto len : supportedArrLen) {
    if (len * inner_block_width <= restriction)
      result = len;
  }
  return result;
}

// It rewrites a XeTile::init_tile into one or more mlir::xegpu::create_nd_desc
// It is one of start points of generating 1:N values.
class SgInitTileOpPattern : public XeOneToNConversion<xetile::InitTileOp> {
  using XeOneToNConversion<xetile::InitTileOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    mlir::Value source = op.getSource();
    auto tileTy = op.getType();
    auto innerBlocks = tileTy.getInnerBlocks();
    auto shape = llvm::to_vector(tileTy.getShape());
    auto indexType = rewriter.getIndexType();

    if (tileTy.getRank() != 2)
      return op.emitOpError("The tile shape should be 2D.");

    if (!innerBlocks || innerBlocks.size() != 2)
      return op.emitOpError("Missing valid innerBlock for the tile in op.");

    bool hasColMajorTraversal =
        tileTy.getOrder().asArrayRef() == mlir::ArrayRef({0, 1});
    // Need to make a copy, so we can swap values.
    auto innerBlk = llvm::to_vector(innerBlocks.asArrayRef());
    // If order is [1, 0] (source memref is col-major), we need to swap the
    // shape and innerBlocks because XeGPU ops only support row-major tile
    // creation.
    if (hasColMajorTraversal) {
      assert(op.isSourceMemRef() && op.sourceMemRefHasStaticShape() &&
             "Expecting a static shape source memref.");
      std::swap(innerBlk[0], innerBlk[1]);
      std::swap(shape[0], shape[1]);
    }

    // using array_length for load if dim1 of innerBlocks is smaller than
    // dim1 of shape.
    auto elemTy = tileTy.getElementType();
    auto array_length =
        isForLoad(op) && shape[1] > innerBlk[1]
            ? getBlockArrayLength(op, elemTy, innerBlk[1], shape[1])
            : 1;

    // If the source memref is col-major we need to convert into a row-major
    // because at XeGPU level we only support row-major memrefs. Also
    // array_length must be 1 if col-major memrefs are used as the source.
    if (hasColMajorTraversal) {
      array_length = 1;
      // create a memref.reinterpret_cast to convert col-major to row-major
      auto rowMajorSourceShape =
          swapLastTwoElements(op.getSourceMemrefStaticShape());
      auto rowMajorSourceStrides = defaultStrides(rowMajorSourceShape);
      int64_t rowMajorSourceOffset = 0;
      auto newMemRefTy = mlir::MemRefType::get(rowMajorSourceShape, elemTy);
      source = rewriter.create<mlir::memref::ReinterpretCastOp>(
          loc, newMemRefTy, source, rowMajorSourceOffset, rowMajorSourceShape,
          rowMajorSourceStrides);
    }

    auto width = array_length * innerBlk[1];

    llvm::SmallVector<int64_t, 2> blocks(
        {shape[0] / innerBlk[0], shape[1] / width});

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

    // For col-major memref initial offsets need to be swapped.
    auto offsetsY = offsets.pop_back_val();
    auto offsetsX = offsets.pop_back_val();
    if (hasColMajorTraversal)
      std::swap(offsetsX, offsetsY);

    auto tDescTy = mlir::xegpu::TensorDescType::get(
        innerBlk, elemTy, false /*scattered*/, array_length,
        mlir::xegpu::MemoryScope::Global, true /*boundary_check*/);

    auto createIndexConstant = [&](mlir::Type type, int64_t value) {
      auto attr = rewriter.getIndexAttr(value);
      return rewriter.create<mlir::arith::ConstantOp>(loc, type, attr);
    };

    rewriter.setInsertionPoint(op);

    llvm::SmallVector<mlir::Value> xegpuOps(blocks[0] * blocks[1],
                                            mlir::Value());
    for (int i = 0; i < blocks[0]; i++) {
      for (int j = 0; j < blocks[1]; j++) {
        auto subOffX = createIndexConstant(indexType, (innerBlk[0] * i));
        auto subOffY = createIndexConstant(indexType, (width * j));
        auto tDescOffsetX =
            rewriter.createOrFold<mlir::arith::AddIOp>(loc, subOffX, offsetsX);
        auto tDescOffsetY =
            rewriter.createOrFold<mlir::arith::AddIOp>(loc, subOffY, offsetsY);
        mlir::SmallVector<mlir::OpFoldResult> tDescOffsets = llvm::to_vector<4>(
            llvm::map_range(offsets, [](mlir::Value v) -> mlir::OpFoldResult {
              return v;
            }));
        tDescOffsets.push_back(tDescOffsetX);
        tDescOffsets.push_back(tDescOffsetY);

        // TODO: this needs improvement, it assumes the source is static
        // memeref.
        if (auto MemRefTypedSource =
                mlir::cast<mlir::TypedValue<mlir::MemRefType>>(source)) {
          auto createNdOp = rewriter.create<mlir::xegpu::CreateNdDescOp>(
              op.getLoc(), tDescTy /*resultTy*/, MemRefTypedSource /*source*/,
              tDescOffsets /*offsets*/);
          // col-major source memref requires creating the tiles in transposed
          // order
          if (hasColMajorTraversal)
            xegpuOps[blocks[0] * j + i] = createNdOp;
          else
            xegpuOps[blocks[1] * i + j] = createNdOp;
        } else {
          return mlir::failure();
        }
      }
    }

    rewriter.replaceOp(op, xegpuOps);
    return mlir::success();
  }
};

static mlir::xegpu::CachePolicy
translateCachePolicy(imex::xetile::CachePolicyAttr val) {
  if (!val)
    return mlir::xegpu::CachePolicy::CACHED;

  switch (val.getValue()) {
  case imex::xetile::CachePolicy::CACHED:
    return mlir::xegpu::CachePolicy::CACHED;
  case imex::xetile::CachePolicy::UNCACHED:
    return mlir::xegpu::CachePolicy::UNCACHED;
  case imex::xetile::CachePolicy::STREAMING:
    return mlir::xegpu::CachePolicy::STREAMING;
  case imex::xetile::CachePolicy::READ_INVALIDATE:
    return mlir::xegpu::CachePolicy::READ_INVALIDATE;
  case imex::xetile::CachePolicy::WRITE_BACK:
    return mlir::xegpu::CachePolicy::WRITE_BACK;
  case imex::xetile::CachePolicy::WRITE_THROUGH:
    return mlir::xegpu::CachePolicy::WRITE_THROUGH;
  }
  llvm_unreachable("Invalid CachePolicy value");
}

// It lowers a XeTile::prefetch_tile into one or more mlir::xegpu::prefetch_2d.
// The adaptor will provide the set of xegpu.create_nd_desc lowered for
// its input tile.
struct SgPrefetchTileOpPattern
    : public XeOneToNConversion<xetile::PrefetchTileOp> {
  using XeOneToNConversion<xetile::PrefetchTileOp>::XeOneToNConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    auto tiles = adaptor.getTile();
    auto innerBlocks = tileTy.getInnerBlocks();

    if (tileTy.getRank() != 2)
      return mlir::failure();

    if (!innerBlocks || innerBlocks.size() != 2)
      return mlir::failure();

    auto shape = tileTy.getShape();
    auto expectedNumTensorDescs =
        (shape[0] / innerBlocks[0]) * (shape[1] / innerBlocks[1]);
    if (expectedNumTensorDescs != (int64_t)tiles.size()) {
      op.emitOpError("Failed to lower LoadTileOp because shape[0] * shape[1] "
                     "!= sources.size().");
      return mlir::failure();
    }

    auto getCachePolicy = [&](imex::xetile::CachePolicyAttr val) {
      return mlir::xegpu::CachePolicyAttr::get(op.getContext(),
                                               translateCachePolicy(val));
    };

    auto L1 = getCachePolicy(op.getL1HintAttr());
    auto L2 = getCachePolicy(op.getL2HintAttr());
    auto L3 = getCachePolicy(op.getL3HintAttr());

    for (auto tile : tiles) {
      rewriter.create<mlir::xegpu::PrefetchNdOp>(op.getLoc(), tile, L1, L2, L3);
    }

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

// It lowers XeTile::load_tile into one or more mlir::xegpu::load_2d
// The adaptor will provide the set of xegpu.create_nd_desc lowered for
// its input tile.
struct SgLoadTileOpPattern : public XeOneToNConversion<xetile::LoadTileOp> {
  using XeOneToNConversion<xetile::LoadTileOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto tileTy = op.getSource().getType();
    auto blockSZ = tileTy.getInnerBlocks();

    // It expects the tile has been tiled using blocking pass
    if (!blockSZ)
      return mlir::failure();

    auto elemTy = tileTy.getElementType();
    auto sources = adaptor.getSource();

    auto ctx = op.getContext();
    auto L1 = mlir::xegpu::CachePolicyAttr::get(
        ctx, mlir::xegpu::CachePolicy::CACHED);
    auto L2 = mlir::xegpu::CachePolicyAttr::get(
        ctx, mlir::xegpu::CachePolicy::CACHED);
    auto L3 = mlir::xegpu::CachePolicyAttr::get(
        ctx, mlir::xegpu::CachePolicy::CACHED);

    mlir::UnitAttr vnniAttr = nullptr;
    mlir::IntegerAttr transposeBitWidthAttr;
    // TODO: move these two into architecture abstracture in future.
    const int SIMD_WIDTH_IN_BITS = 32;
    int factor = SIMD_WIDTH_IN_BITS / elemTy.getIntOrFloatBitWidth();
    // TODO: use uArch for this?
    auto isLowPrecision = [](unsigned int width) -> bool {
      bool isPowerOf2 = (width & (width - 1)) == 0;
      return isPowerOf2 & (width < 32) & (width > 1);
    };
    if (isForDPASB(op) && factor > 1)
      vnniAttr = mlir::UnitAttr::get(ctx);

    mlir::DenseI64ArrayAttr transposeAttr;
    auto srcOrder = tileTy.getOrder();
    if (srcOrder.asArrayRef() == mlir::ArrayRef({1, 0})) {
      // Nothing to do
    } else if (srcOrder.asArrayRef() == mlir::ArrayRef({0, 1})) {
      auto elemWidth = elemTy.getIntOrFloatBitWidth();
      if (elemWidth == 32) {
        transposeAttr = rewriter.getDenseI64ArrayAttr({1, 0});
      } else if (isLowPrecision(elemWidth) && vnniAttr) {
        transposeAttr = rewriter.getDenseI64ArrayAttr({1, 0});
        transposeBitWidthAttr = rewriter.getI32IntegerAttr(32);
        vnniAttr = nullptr;
      } else {
        return ((mlir::PatternRewriter &)rewriter)
            .notifyMatchFailure(op, "Unsupported element type for transpose");
      }
    } else {
      return ((mlir::PatternRewriter &)rewriter)
          .notifyMatchFailure(op, "Unsupported order");
    }

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<::mlir::Value> xegpuOps;
    for (auto src : sources) {
      auto tdescTy = llvm::dyn_cast<mlir::xegpu::TensorDescType>(src.getType());
      assert(tdescTy && "Expecting a TensorDescType value for load_tile.");
      auto shape = tdescTy.getShape().vec();
      auto array_length = tdescTy.getArrayLength();

      if (transposeAttr)
        std::swap(shape[0], shape[1]);

      if (vnniAttr || transposeBitWidthAttr) {
        const int axis = 0;
        shape[axis] /= factor;
        shape.push_back(factor);
      }

      if (array_length != 1)
        shape.insert(shape.begin(), array_length);

      auto vectorTy = mlir::VectorType::get(shape, elemTy);
      auto ldOp = rewriter.create<mlir::xegpu::LoadNdOp>(
          op.getLoc(), vectorTy, src, vnniAttr, transposeAttr,
          transposeBitWidthAttr, L1, L2, L3);
      if (array_length == 1) {
        xegpuOps.push_back(ldOp);
      } else {
        for (auto i = 0; i < array_length; i++) {
          auto extractOp = rewriter.create<ExtractOp>(op.getLoc(), ldOp, i);
          xegpuOps.push_back(extractOp);
        }
      }
    }

    rewriter.replaceOp(op, xegpuOps);
    return mlir::success();
  }
};

// It lowers a XeTile::store_tile into one or more mlir::xegpu::store_2d
// The adaptor will provide the set of xegpu.create_nd_desc lowered for
// its input tile, and similar to its input vector value.
struct SgStoreTileOpPattern : public XeOneToNConversion<xetile::StoreTileOp> {
  using XeOneToNConversion<xetile::StoreTileOp>::XeOneToNConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto tiles = adaptor.getTile();
    auto values = adaptor.getValue();

    if (tiles.size() != values.size()) {
      return op.emitOpError("[Failed to lower the StoreOp]")
             << "tile and value size doesn't match."
             << "tiles: " << tiles.size() << ", "
             << "values: " << values.size() << "\n";
    }

    auto context = op.getContext();
    auto WRITE_BACK = mlir::xegpu::CachePolicy::WRITE_BACK;
    auto L1 = mlir::xegpu::CachePolicyAttr::get(context, WRITE_BACK);
    auto L2 = mlir::xegpu::CachePolicyAttr::get(context, WRITE_BACK);
    auto L3 = mlir::xegpu::CachePolicyAttr::get(context, WRITE_BACK);
    for (size_t i = 0; i < tiles.size(); i++)
      rewriter.create<mlir::xegpu::StoreNdOp>(op.getLoc(), values[i], tiles[i],
                                              L1, L2, L3);

    rewriter.eraseOp(op);
    return ::mlir::success();
  }
};

// It lowers a XeTile::tile_mma into one or more mlir::xegpu::dpas
// The adaptor provides new inputs for each old input.
struct SgTileMMAOpPattern : public XeOneToNConversion<xetile::TileMMAOp> {
  using XeOneToNConversion<xetile::TileMMAOp>::XeOneToNConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

    auto aShape = op.getAType().getShape();
    auto bShape = op.getBType().getShape();

    if (aShape.size() != 4 || bShape.size() != 4) {
      op.emitOpError() << "Operand A and B for mma should be 4d.\n";
      return mlir::failure();
    }

    if (aShape[3] != bShape[2] || aShape[1] != bShape[0]) {
      op.emitOpError() << "A and B size doesn't match. A should be m x k, and "
                          "B should be k x n";
      return mlir::failure();
    }

    uint64_t M = aShape[0];
    uint64_t K = aShape[1];
    uint64_t N = bShape[1];

    auto loc = op.getLoc();
    auto AValues = adaptor.getA();
    auto BValues = adaptor.getB();
    auto CValues = adaptor.getC();

    auto elemTy = op.getOutput().getType().getElementType();
    auto subCTy = mlir::VectorType::get({aShape[2], bShape[3]}, elemTy);

    mlir::SmallVector<mlir::Value> xegpuOps;
    for (uint64_t i = 0; i < M; i++) {
      for (uint64_t j = 0; j < N; j++) {
        mlir::Value tmpC;
        if (op.getC())
          tmpC = CValues[i * N + j]; // init with acc
        for (uint64_t k = 0; k < K; k++) {
          auto aVec = AValues[i * K + k];
          auto bVec = BValues[k * N + j];
          tmpC = rewriter.create<mlir::xegpu::DpasOp>(
              loc, subCTy /*result*/, aVec /*lhs*/, bVec /*rhs*/, tmpC /*acc*/);
        }
        xegpuOps.push_back(tmpC);
      }
    }
    rewriter.replaceOp(op, xegpuOps);
    return mlir::success();
  }
};

struct SgUpdateTileOffsetOpPattern
    : public XeOneToNConversion<xetile::UpdateTileOffsetOp> {
  using XeOneToNConversion<xetile::UpdateTileOffsetOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto offsetX = op.getOffsetX();
    auto offsetY = op.getOffsetY();
    auto tiles = adaptor.getTile();

    bool hasColMajorTraversal =
        op.getTile().getType().getOrder().asArrayRef() ==
        mlir::ArrayRef({0, 1});

    llvm::SmallVector<mlir::Value> newOps;
    int64_t kDynamics[2] = {mlir::ShapedType::kDynamic,
                            mlir::ShapedType::kDynamic};
    for (const auto &tile : tiles) {
      // if the traversal is col-major, we need to reverse the offsets at XeGPU
      // level because only row-major traversal is supported.
      auto xegpuTile = rewriter.create<mlir::xegpu::UpdateNdOffsetOp>(
          op.getLoc(), tile.getType(), tile,
          hasColMajorTraversal ? mlir::ValueRange({offsetY, offsetX})
                               : mlir::ValueRange({offsetX, offsetY}),
          llvm::ArrayRef<int64_t>(kDynamics, 2));
      newOps.push_back(xegpuTile);
    }
    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

extern llvm::SmallVector<mlir::Value>
lowerOuterReduction(mlir::ValueRange sources, llvm::ArrayRef<int64_t> shape,
                    mlir::vector::CombiningKind kind, mlir::Location loc,
                    mlir::Type elemTy, XeOneToNPatternRewriter &rewriter);

extern llvm::SmallVector<mlir::Value>
lowerInnerReductionWithIntraVectorShuffles(mlir::ValueRange sources,
                                           llvm::ArrayRef<int64_t> shape,
                                           mlir::vector::CombiningKind kind,
                                           mlir::Location loc,
                                           mlir::Type elemTy,
                                           XeOneToNPatternRewriter &rewriter);

extern llvm::SmallVector<mlir::Value> lowerInnerReductionWithVectorReduction(
    mlir::ValueRange sources, llvm::ArrayRef<int64_t> shape,
    mlir::vector::CombiningKind kind, mlir::Location loc, mlir::Type elemTy,
    XeOneToNPatternRewriter &rewriter);

struct SgTileReduceOpPattern : public XeOneToNConversion<xetile::ReduceOp> {
  using XeOneToNConversion<xetile::ReduceOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::ReduceOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto srcTy = op.getSource().getType();
    auto elemTy = srcTy.getElementType();
    auto dims = op.getReductionDim();
    // its input should be a 4D vector, and has 2 reduction dims,
    // otherwise run the blocking pass first.
    if (dims.size() != 2 || srcTy.getRank() != 4)
      return mlir::failure();

    auto loc = op.getLoc();
    auto shape = srcTy.getShape();
    auto sources = adaptor.getSource();

    rewriter.setInsertionPoint(op);
    // doing reduction on outer dimension
    if (dims[0] == 0 && dims[1] == 2) {
      auto intermediates = lowerOuterReduction(sources, shape, op.getKind(),
                                               loc, elemTy, rewriter);
      rewriter.replaceOp(op, intermediates);
      return mlir::success();
    }

    // doing reduction on inner dimension, otherwise it is not supported.
    assert(dims[0] == 1 && dims[1] == 3 && "unsupported reduction operation.");

    auto intermediates = lowerInnerReductionWithIntraVectorShuffles(
        sources, shape, op.getKind(), loc, elemTy, rewriter);
    llvm::SmallVector<mlir::Value> newOps;
    {
      // intermediate is a vector of values with type of vector<shape[3]xf16>,
      // each value represents a portion of the reduced value. For example,
      // for vector<32x4x1x16> with reduction on dim 1 and dim 3. the
      // intermediate values will be two vectors of vector<16xf16>. The values
      // in the first vector represents the reduction result of the first 16
      // rows. Here we will extract each value and splat it to a vector<1x1xf16>
      // as results to their consumers.
      for (auto v : intermediates) {
        auto targetTy = mlir::VectorType::get({1, 1}, elemTy);
        for (auto i = 0; i < shape[3]; i++) {
          auto pos = rewriter.create<mlir::arith::ConstantOp>(
              op.getLoc(), rewriter.getI32IntegerAttr(i));
          auto extractOp =
              rewriter.create<mlir::vector::ExtractElementOp>(loc, v, pos);
          auto splatOp = rewriter.create<mlir::vector::SplatOp>(
              op.getLoc(), targetTy, extractOp);
          newOps.push_back(splatOp);
        }
      }
    }
    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

// A transpose op for a larger vector will be lowered into multiple
// explicit transpose ops for smaller vectors and the order/use of
// these these new transpose ops are transposed too. For example:
// xetile.transpose %1, [1, 0]: vector<16x48> -> vector<48x16> will
// be lowered into 6 transpose ops on vector<8x16> assuming the smaller
// vector shape is 8x16. So it will from:
// |--------------|--------------|--------------|
// |  0: 8x16     |  1: 8x16     |  2: 8x16     |
// |--------------|--------------|--------------|
// |  3: 8x16     |  4: 8x16     |  5: 8x16     |
// |--------------|--------------|--------------|
//
// to:
//
// |--------------|--------------|
// |  0: 16x8     |  3: 16x8     |
// |--------------|--------------|
// |  1: 16x8     |  4: 16x8     |
// |--------------|--------------|
// |  2: 16x8     |  5: 16x8     |
// |--------------|--------------|
// (the number before `:` is the id of the block)

template <typename OpTy>
struct SgTransposeOpPattern : public XeOneToNConversion<OpTy> {
  using XeOneToNConversion<OpTy>::XeOneToNConversion;
  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OpAdaptor = typename OpTy::template GenericAdaptor<RangeT>;

  mlir::LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto resType = op.getResult().getType();
    if (resType.getRank() != 4)
      return ((mlir::PatternRewriter &)rewriter)
          .notifyMatchFailure(op, "Expected a 4D vector");

    auto srcVectors = adaptor.getVector();
    auto shape = resType.getShape();
    if (shape[0] * shape[1] != static_cast<int64_t>(srcVectors.size()))
      return ((mlir::PatternRewriter &)rewriter)
          .notifyMatchFailure(op, "Invalid shape");

    auto permutation = op.getPermutation();
    auto outerPerm = permutation.take_front(2);
    int64_t innerPerm[2] = {permutation[2] - 2, permutation[3] - 2};

    auto newResType =
        mlir::VectorType::get(shape.take_back(2), resType.getElementType());

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value> results;
    for (auto i : llvm::seq<size_t>(0, shape[0])) {
      for (auto j : llvm::seq<size_t>(0, shape[1])) {
        size_t ij[2] = {i, j};
        auto idx = ij[outerPerm[1]] + shape[outerPerm[1]] * ij[outerPerm[0]];
        mlir::Value arg = srcVectors[idx];
        mlir::Value res = rewriter.create<mlir::vector::TransposeOp>(
            loc, newResType, arg, innerPerm);
        results.emplace_back(res);
      }
    }
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

bool isLegalElementWiseOp(mlir::Operation *op) {
  auto res = op->getResult(0);
  auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
  if (resType && resType.getRank() > 2)
    return false;
  return true;
}

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
struct ElementWiseOpPattern : public XeOneToNConversion<Op> {

  using XeOneToNConversion<Op>::XeOneToNConversion;
  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OpAdaptor = typename Op::template GenericAdaptor<RangeT>;

  mlir::LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {

    auto res = op.getResult();
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    // non-vector ops, or 1D/2D vector ops generated during lowering.
    if (!resType || resType.getRank() <= 2)
      return mlir::failure();

    // For non 2D vector ops, we expect 4D vector ops only
    if (resType.getRank() != 4) {
      op.emitOpError() << "type is not 4D vector";
      return mlir::failure();
    }

    auto shape = resType.getShape();
    auto newTy =
        mlir::VectorType::get({shape[2], shape[3]}, resType.getElementType());

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

    llvm::SmallVector<mlir::Value> newOps;
    for (int i = 0; i < shape[0] * shape[1]; i++) {
      auto newOp = createOp<Op, numOperands>(rewriter, op.getLoc(), operand, i);
      newOp->getResult(0).setType(newTy);
      newOps.push_back(newOp);
    }

    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

template <typename CastOp>
struct TypecastOpPattern : public XeOneToNConversion<CastOp> {
  using XeOneToNConversion<CastOp>::XeOneToNConversion;
  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OpAdaptor = typename CastOp::template GenericAdaptor<RangeT>;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto out = mlir::dyn_cast<mlir::VectorType>(op.getType());
    if (!out || out.getRank() != 4)
      return mlir::failure();

    auto shape = out.getShape();
    auto vecTy =
        mlir::VectorType::get({shape[2], shape[3]}, out.getElementType());
    auto inputs = adaptor.getIn();
    llvm::SmallVector<mlir::Value> newOps;
    for (auto in : inputs) {
      auto newOp = rewriter.create<CastOp>(op.getLoc(), vecTy, in);
      newOps.push_back(newOp);
    }
    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

struct SgBroadcastOpPattern : public XeOneToNConversion<xetile::BroadcastOp> {
  using XeOneToNConversion<xetile::BroadcastOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::BroadcastOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto resultTy = op.getResult().getType();
    auto resultShape = resultTy.getShape();
    auto dstType = mlir::VectorType::get(resultShape.take_back(2),
                                         resultTy.getElementType());

    auto dim = op.getBroadcastDim();
    assert(dim.size() == 2 && "Expecting 2D broadcast dim.");

    llvm::SmallVector<mlir::Value> newOps;
    if (dim[0] == 0 && dim[1] == 2) {
      // clang-format off
      // broadcast along the first dim, we simply need to replicate the source.
      // For example, for
      //    xetile.broadcast %src [0]: vector<1x64xf16> -> vector<32x64xf16>
      // After blocking (assuming block size = [1, 16]) and lowering to xegpu,
      // its input values (source) will be a vector of values with type <1x16xf16>
      // and size = 4, which can be viewed as:
      // | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      // so we need to replicate it 32 times (resultShape[0]) to get final results:
      //  0: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      //  ......
      // 31: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      // clang-format on
      for (auto i = 0; i < resultShape[0]; i++)
        newOps.append(adaptor.getSource().begin(), adaptor.getSource().end());
    } else if (dim[0] == 1 && dim[1] == 3) {
      // clang-format off
      // broadcast along the second dim, we use both splatOp and replicates.
      // For example: xetile.broadcast %src [1]: vector<32x1xf16> ->
      // vector<32x64xf16>. After blocking (assuming block size = [1, 16]) and
      // lowering to xegpu, the input value (source) will be a vector of values
      // with type <1x1xf16> and size = 32, which can be viewed as:
      //    0: | vector<1x1xf16> |
      //           ...
      //   31: | vector<1x1xf16> |
      // first, splatOp is used to broadcast the value of vector<1x1xf16> to
      // vector<1x16xf16>
      //    0: | vector<1x16xf16> |
      //           ...
      //   31: | vector<1x16xf16> |
      // and then we replicate the splatOp 4 times (resultShape[1]) to get the
      // final results:
      //    0: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      //           ...
      //   31: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      // clang-format on
      for (auto src : adaptor.getSource()) {
        auto ty = mlir::dyn_cast<mlir::VectorType>(src.getType());
        assert(ty && ty.getNumElements() == 1 &&
               "Expecting a <1x1xelemty> vector type.");
        auto ext = rewriter.create<mlir::vector::ExtractOp>(
            op.getLoc(), src, llvm::ArrayRef<int64_t>({0, 0}));
        auto splatOp =
            rewriter.create<mlir::vector::SplatOp>(op.getLoc(), dstType, ext);
        newOps.append(resultShape[1], splatOp);
      }
    } else {
      return mlir::failure();
    }
    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

void populateXeTileOpConversionPatterns(imex::XeOneToNTypeConverter &converter,
                                        mlir::RewritePatternSet &patterns,
                                        TileUsageAnalysis &analysis) {
  patterns.insert<SgInitTileOpPattern, SgPrefetchTileOpPattern,
                  SgTileUnpackOpPattern, SgTilePackOpPattern,
                  SgLoadTileOpPattern, SgStoreTileOpPattern, SgTileMMAOpPattern,
                  SgUpdateTileOffsetOpPattern,
                  SgTransposeOpPattern<mlir::vector::TransposeOp>,
                  SgTransposeOpPattern<xetile::TransposeOp>,
                  SgBroadcastOpPattern, SgTileReduceOpPattern>(
      patterns.getContext(), converter, analysis);
  patterns.insert<ElementWiseOpPattern<mlir::arith::NegFOp, 1>,
                  ElementWiseOpPattern<mlir::math::ExpOp, 1>,
                  ElementWiseOpPattern<mlir::math::SinOp, 1>,
                  ElementWiseOpPattern<mlir::math::CosOp, 1>,
                  ElementWiseOpPattern<mlir::math::SqrtOp, 1>,
                  ElementWiseOpPattern<mlir::math::TanhOp, 1>,
                  ElementWiseOpPattern<mlir::math::LogOp, 1>,
                  ElementWiseOpPattern<mlir::math::RsqrtOp, 1>,
                  ElementWiseOpPattern<mlir::math::ErfOp, 1>,
                  ElementWiseOpPattern<mlir::arith::AddFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::RemFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::DivFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::MulFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::MaximumFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::MinimumFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::SubFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::XOrIOp, 2>,
                  ElementWiseOpPattern<mlir::math::PowFOp, 2>,
                  ElementWiseOpPattern<mlir::arith::SelectOp, 3>>(
      patterns.getContext(), converter, analysis);
  patterns.insert<TypecastOpPattern<mlir::arith::ExtFOp>,
                  TypecastOpPattern<mlir::arith::ExtSIOp>,
                  TypecastOpPattern<mlir::arith::ExtUIOp>,
                  TypecastOpPattern<mlir::arith::FPToSIOp>,
                  TypecastOpPattern<mlir::arith::FPToUIOp>,
                  TypecastOpPattern<mlir::arith::IndexCastOp>,
                  TypecastOpPattern<mlir::arith::IndexCastUIOp>,
                  TypecastOpPattern<mlir::arith::SIToFPOp>,
                  TypecastOpPattern<mlir::arith::UIToFPOp>,
                  TypecastOpPattern<mlir::arith::TruncFOp>,
                  TypecastOpPattern<mlir::arith::TruncIOp>>(
      patterns.getContext(), converter, analysis);
}

} // namespace imex
