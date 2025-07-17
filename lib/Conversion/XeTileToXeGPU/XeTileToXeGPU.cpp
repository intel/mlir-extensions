//===- XeTileToXeGPU.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeTileToXeGPU conversion, converting the XeTile
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/Passes.h>

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h"
#include "imex/Utils/XeArch.h"
#include "imex/Utils/XeCommon.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <memory>

namespace imex {
#define GEN_PASS_DEF_CONVERTXETILETOXEGPU
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

using namespace mlir;
namespace imex {

// Converts an Attribute representing memory space to xegpu::MemorySpaceAttr.
// It currently only supports memory space represented as integer attribute.
// TODO: improve it to support other types of memory space attributes, e.g.,
// gpu::MemorySpaceAttr, and spirv::MemorySpaceAttr, etc.
static xegpu::MemorySpaceAttr convertMemorySpace(Attribute attr) {
  auto space = xegpu::MemorySpace::Global; // default to global value
  if (auto IntAttr = dyn_cast_if_present<IntegerAttr>(attr)) {
    space = IntAttr.getInt() == 3 ? xegpu::MemorySpace::SLM
                                  : xegpu::MemorySpace::Global;
  }
  return attr ? xegpu::MemorySpaceAttr::get(attr.getContext(), space)
              : xegpu::MemorySpaceAttr();
}

static xegpu::CachePolicy
translateCachePolicy(imex::xetile::CachePolicyAttr val,
                     xegpu::CachePolicy defaultVal) {
  if (!val)
    return defaultVal;

  switch (val.getValue()) {
  case imex::xetile::CachePolicy::CACHED:
    return xegpu::CachePolicy::CACHED;
  case imex::xetile::CachePolicy::UNCACHED:
    return xegpu::CachePolicy::UNCACHED;
  case imex::xetile::CachePolicy::STREAMING:
    return xegpu::CachePolicy::STREAMING;
  case imex::xetile::CachePolicy::READ_INVALIDATE:
    return xegpu::CachePolicy::READ_INVALIDATE;
  case imex::xetile::CachePolicy::WRITE_BACK:
    return xegpu::CachePolicy::WRITE_BACK;
  case imex::xetile::CachePolicy::WRITE_THROUGH:
    return xegpu::CachePolicy::WRITE_THROUGH;
  }
  llvm_unreachable("Invalid CachePolicy value");
}

template <typename OpTy>
static auto
getCachePolicy(OpTy op,
               xegpu::CachePolicy defaultVal = xegpu::CachePolicy::CACHED) {

  auto getCachePolicyAttr = [&](imex::xetile::CachePolicyAttr val) {
    return xegpu::CachePolicyAttr::get(op.getContext(),
                                       translateCachePolicy(val, defaultVal));
  };

  auto L1 = getCachePolicyAttr(op.getL1HintAttr());
  auto L2 = getCachePolicyAttr(op.getL2HintAttr());
  auto L3 = getCachePolicyAttr(op.getL3HintAttr());

  return std::make_tuple(L1, L2, L3);
}

// It converts a VectorType value to a 1D vector of 32-bit element type,
// using shapecast and bitcast operations, e.g., vector<4x4xf16> ->
// vector<8xi32>.
static Value convertTo1D32BitVector(Value value, Location loc,
                                    ConversionPatternRewriter &rewriter) {
  auto vecTy = dyn_cast<VectorType>(value.getType());
  if (!vecTy)
    return value;

  auto elemTy = vecTy.getElementType();
  auto shapecastTy = VectorType::get(vecTy.getNumElements(), elemTy);

  if (shapecastTy != vecTy) {
    value = rewriter.create<vector::ShapeCastOp>(loc, shapecastTy, value);
  }

  auto vnni = getVnniFactor(elemTy);
  if (vnni > 1) {
    elemTy = isa<IntegerType>(elemTy) ? (Type)rewriter.getI32Type()
                                      : (Type)rewriter.getF32Type();
    auto castTy = VectorType::get(vecTy.getNumElements() / vnni, elemTy);
    value = rewriter.create<vector::BitCastOp>(loc, castTy, value);
  }
  return value;
}

// This method is essentially to insert ops to do vnni transformation
// on the given value, and returns the value after transformation.
// If the value is only has one use, which is store to slm, it is
// marked as potentialFoldable. Then if value is produced by a LoadNdOp,
// and the loadNdOp doesn't have packedAttr, it will fold the vnni
// transformation with the LoadNdOp, instead of inserting extra ops.
static Value convertToPackedVector(Value value, Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   bool potentialFoldable = false) {
  auto vecTy = dyn_cast<VectorType>(value.getType());
  if (!vecTy)
    return value;

  auto packedTy = getPackedType(vecTy);
  if (packedTy != vecTy) {

    auto defOp = value.getDefiningOp<xegpu::LoadNdOp>();
    if (defOp && potentialFoldable && !defOp.getPackedAttr()) {
      rewriter.startOpModification(defOp);
      defOp.setPacked(true);
      value = defOp.getResult();
      value.setType(packedTy);
      rewriter.finalizeOpModification(defOp);
    } else {
      auto typedValue = dyn_cast<TypedValue<VectorType>>(value);
      value = applyVnniTransform(rewriter, typedValue).first;
    }

    auto elemTy = vecTy.getElementType();

    // shape cast packed type (3D vector) to 2D vector, are required by bitcast
    auto shape = packedTy.getShape();
    vecTy = VectorType::get({shape[0], shape[1] * shape[2]}, elemTy);
    value = rewriter.create<vector::ShapeCastOp>(loc, vecTy, value);

    // cast to 32-bit data, use i32 for intergers and f32 for floats.
    elemTy = isa<IntegerType>(elemTy) ? (Type)rewriter.getI32Type()
                                      : (Type)rewriter.getF32Type();
    vecTy = VectorType::get(packedTy.getShape().take_front(2), elemTy);
    if (vecTy != packedTy)
      value = rewriter.create<vector::BitCastOp>(loc, vecTy, value);
  }
  return value;
}

// a helper utility to perform binary operation on OpFoldResult.
// If both a and b are attributes, it will simply return the result.
// Otherwise, the corresponding arith op will be generated, and an
// contant op will be created if one of them is an attribute.
template <typename ArithOp, template <typename S> class MathOp>
OpFoldResult genBinOp(OpFoldResult a, OpFoldResult b, Location loc,
                      ConversionPatternRewriter &rewriter) {
  OpFoldResult result;
  if (isa<Attribute>(a) && isa<Attribute>(b)) {
    auto aAttr = cast<Attribute>(a);
    auto bAttr = cast<Attribute>(b);
    auto aVal = cast<IntegerAttr>(aAttr).getInt();
    auto bVal = cast<IntegerAttr>(bAttr).getInt();
    result = rewriter.getIndexAttr(MathOp<int64_t>()(aVal, bVal));
  } else {
    auto aVal = getValueOrCreateConstantIndexOp(rewriter, loc, a);
    auto bVal = getValueOrCreateConstantIndexOp(rewriter, loc, b);
    result = rewriter.create<ArithOp>(loc, aVal, bVal).getResult();
  }
  return result;
}

// a helper utility to perform division operation on OpFoldResult and int64_t.
#define div(a, b)                                                              \
  genBinOp<arith::DivSIOp, std::divides>(a, rewriter.getIndexAttr(b), loc,     \
                                         rewriter)

// a helper utility to perform reminder operation on OpFoldResult and int64_t.
#define rem(a, b)                                                              \
  genBinOp<arith::RemSIOp, std::modulus>(a, rewriter.getIndexAttr(b), loc,     \
                                         rewriter)

// a helper utility to perform multiply operation on OpFoldResult and int64_t.
#define mul(a, b)                                                              \
  genBinOp<arith::MulIOp, std::multiplies>(a, rewriter.getIndexAttr(b), loc,   \
                                           rewriter)

// a helper utility to perform addition operation on two OpFoldResult.
#define add(a, b) genBinOp<arith::AddIOp, std::plus>(a, b, loc, rewriter)

// convert init_tile to xegpu::CreateNdDescOp if the tile is for
// blocked load/store on global memory, otherwise, convert it to
// xegpu::CreateDescOp.
class InitOpPattern final : public OpConversionPattern<xetile::InitTileOp> {
public:
  using OpConversionPattern<xetile::InitTileOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tileTy = op.getType();
    auto source = adaptor.getSource();

    auto converter = getTypeConverter();
    auto tdescTy = converter->convertType<xegpu::TensorDescType>(tileTy);
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace =
        memSpaceAttr ? memSpaceAttr.getValue() : xegpu::MemorySpace::Global;

    // here it cannot use tdescTy.isScattered(), because block tile on SLM could
    // be lowered into scattered TensorDesc too.
    auto scatterAttr = tileTy.getScatterAttr();
    bool isScattered = scatterAttr ? scatterAttr.getValue() : false;

    Value newOp;
    if (isScattered) {
      auto idxTy =
          VectorType::get(tdescTy.getNumElements(), rewriter.getIndexType());
      auto indices =
          rewriter.create<vector::ShapeCastOp>(loc, idxTy, op.getIndices());
      newOp =
          rewriter.create<xegpu::CreateDescOp>(loc, tdescTy, source, indices);
    } else if (memSpace == xegpu::MemorySpace::Global) {
      newOp = rewriter.create<xegpu::CreateNdDescOp>(
          loc, tdescTy, source, op.getOffsets(), op.getSizes(), op.getStrides(),
          op.getConstOffsetsAttr(), op.getConstSizesAttr(),
          op.getConstStridesAttr());
    } else { // Lowering for blocked tiles on SLM.
      // For simplicity, it currently restricts the SLM to be a MemRef with
      // static shape.
      if (!op.isSourceMemRef() || !op.sourceMemRefHasStaticShape())
        return failure();

      // compute the base address of a SLM block for a tile, given tile
      // offset [y, x], SLM shape [H, W], and slm block size [BH, BW].
      // The block id the tile belongs to is:
      //     [id_y, id_x] = [y/BH, x/BW]
      // And the base address of the block is:
      //     (id_y * W/BW + id_x) * BH * BW.
      auto getBlockBase =
          [&](llvm::ArrayRef<OpFoldResult> tileOffsets,
              llvm::ArrayRef<int64_t> slmBlock,
              llvm::ArrayRef<int64_t> slmShape) -> OpFoldResult {
        int64_t H = slmShape[0], W = slmShape[1];
        int64_t BH = slmBlock[0], BW = slmBlock[1];
        int64_t blockDims[2] = {H / BH, W / BW};

        OpFoldResult blkIdy = blockDims[0] == 1 ? rewriter.getIndexAttr(0)
                                                : div(tileOffsets[0], BH);
        OpFoldResult blkIdx = blockDims[1] == 1 ? rewriter.getIndexAttr(0)
                                                : div(tileOffsets[1], BW);

        auto blkId = add(mul(blkIdy, blockDims[1]), blkIdx);
        auto blockSize = BH * BW;
        return mul(blkId, blockSize);
      };

      auto slmShape = op.getSourceMemrefStaticShape().vec();
      auto tileShape = tileTy.getShape().vec();

      // TODO: limits the 2nd dim to be 16 for both store_tile and load_tile
      // so that the blocking scheme can work for both. An advanced analysis
      // is needed to make it general.
      // also make sure 1st dim of tile divides 1st dim of SLM shape, such
      // that SLM can be logically organized as slmShape[1]/16 blocks, and
      // each block is of shape [slmShape[0], 16]
      if (tileShape[1] != 16 || slmShape[0] % tileShape[0] != 0)
        return failure();

      auto vnni = getVnniFactor(tileTy.getElementType());
      auto slmBlock = llvm::SmallVector<int64_t>({slmShape[0], 16});
      auto offsets = op.getMixedOffsets();

      // Todo: adjust the shape and offset based on vnni. Logically, for
      // the scattered view, it needs to adjust the 1st dim, but they are
      // actually equivalent in current supported cases. It may need
      // reevaluation when to support more general cases.
      slmBlock[1] /= vnni;
      slmShape[1] /= vnni;
      if (isColMajorOrder(tileTy.getOrder())) { // lower to scattered TensorDesc
        // Generate the scattered offsets for the tile, given tileOffsets = [y,
        // x], tileShape = [h, w], slmShape = [H, W], and slm block size = [BH,
        // BW] (currently, [BH, BW] is fixed to [H, 16]). the base address,
        // blockAddr, of the SLM block is computed using getBlockBase. the
        // in-block offset is:
        //    inBlockOffset = [y % BH, x % BW].
        // the linearized in-block offset is therefore:
        //    linearInBlockOffset = inBlockOffset[0] * Bw + inBlockOffset[1].
        // the linearized offset in SLM is:
        //    linearOffset = blockAddr + linearInBlockOffset.
        // Each row of the tile is accessed by a SIMD lane, and the relative
        // offsets for each lane is fixed to
        //    cst = [0, BW, 2 * BW, ... (h-1) * BW].
        // Thus the final offset for each lane is:
        //    linearOffset + cst.
        auto getScatteredOffsets =
            [&](llvm::ArrayRef<OpFoldResult> tileOffsets,
                llvm::ArrayRef<int64_t> tileShape,
                llvm::ArrayRef<int64_t> slmBlock,
                llvm::ArrayRef<int64_t> slmShape) -> Value {
          auto blockAddr = getBlockBase(tileOffsets, slmBlock, slmShape);
          int64_t BH = slmBlock[0], BW = slmBlock[1];
          auto x = rem(tileOffsets[1], BW);
          auto y = rem(tileOffsets[0], BH);
          y = mul(y, BW);
          // TODO: current logic assumes tileOffsets[1] % slmBlock[1] == 0.
          // But it is good to add a runtime assert to check this. Unfortunatly,
          // we don't have support for lowering cr.assertOp to spirv yet.
          auto base = add(add(blockAddr, y), x);

          // cst is to store the constant offsets for each row in the tile.
          // e.g., [0, 16, 32, 48, ...]
          auto stride = BW;
          llvm::SmallVector<int64_t> cst(tileShape[0]);
          std::generate(cst.begin(), cst.end(),
                        [stride, i = 0]() mutable { return stride * i++; });
          auto cstOffsets = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIndexVectorAttr(cst));
          auto baseValue = getValueOrCreateConstantIndexOp(rewriter, loc, base);
          auto idxTy = VectorType::get(tileShape[0], rewriter.getIndexType());
          auto splat = rewriter.create<vector::SplatOp>(loc, baseValue, idxTy);
          auto result = add(splat.getResult(), cstOffsets.getResult());
          return getValueOrCreateConstantIndexOp(rewriter, loc, result);
        };
        //    %t = init_tile %slm [y, x]: memref<m x n x.., strided, slm>
        //                                -> tile<32x16x.., ordered, slm>
        // is equivalent to
        //    %t = init_tile %slm [y, x]: memref<n x m x.., slm>
        //                                -> tile<16x32x.., slm>
        std::reverse(offsets.begin(), offsets.end());
        std::reverse(tileShape.begin(), tileShape.end());
        offsets[1] = div(offsets[1], vnni);
        newOp = rewriter.create<xegpu::CreateDescOp>(
            loc, tdescTy, source,
            getScatteredOffsets(offsets, tileShape, slmBlock, slmShape));
      } else { // lower to 1D block TenssorDesc
        // Generate a linearized offset for a tile, given tileOffsets = [y, x]
        // slmShape = [H, W], and slmBlock = [BH, BW]. It first uses getBlocBase
        // to compute the blockAddr, and the inBlockOffset is:
        //           inBlockOffset: y % BH * BW.
        // The final linearized offset is:
        //           blockAddr + inBlockOffset.
        // TODO: the design assumes x % BW == 0, and it is good to add a runtime
        // assert to check this. Unfortunatly, there is not support for lowering
        // cr.assertOp to spirv yet.
        auto getLinearOffset =
            [&](llvm::ArrayRef<OpFoldResult> tileOffsets,
                llvm::ArrayRef<int64_t> slmBlock,
                llvm::ArrayRef<int64_t> slmShape) -> OpFoldResult {
          auto blockAddr = getBlockBase(tileOffsets, slmBlock, slmShape);
          auto y = rem(tileOffsets[0], slmBlock[0]);
          auto inBlockOffset = mul(y, slmBlock[1]);
          return add(blockAddr, inBlockOffset);
        };

        offsets[1] = div(offsets[1], vnni);
        newOp = rewriter.create<xegpu::CreateNdDescOp>(
            loc, tdescTy, dyn_cast<TypedValue<MemRefType>>(source),
            getLinearOffset(offsets, slmBlock, slmShape));
      }
    }
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// convert update_tile_offset to xegpu::UpdateNdOffsetOp if the tile
// is for blocked load/store on global memory, otherwise, convert it to
// xegpu::UpdateOffsetOp.
class UpdateOpPattern : public OpConversionPattern<xetile::UpdateTileOffsetOp> {
public:
  using OpConversionPattern<xetile::UpdateTileOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    auto scatterAttr = tileTy.getScatterAttr();
    auto isScattered = scatterAttr ? scatterAttr.getValue() : false;
    auto tdesc = adaptor.getTile();
    if (isScattered) {
      auto indicesTy = op.getIndices().getType();
      auto flatTy = VectorType::get(indicesTy.getNumElements(),
                                    indicesTy.getElementType());
      auto indices = rewriter.create<vector::ShapeCastOp>(op.getLoc(), flatTy,
                                                          adaptor.getIndices());
      rewriter.replaceOpWithNewOp<xegpu::UpdateOffsetOp>(op, tdesc.getType(),
                                                         tdesc, indices);
    } else {
      auto x = op.getOffsetX();
      auto y = op.getOffsetY();
      int64_t kDynamics[2] = {ShapedType::kDynamic, ShapedType::kDynamic};
      rewriter.replaceOpWithNewOp<xegpu::UpdateNdOffsetOp>(
          op, tdesc.getType(), tdesc, ValueRange({x, y}),
          llvm::ArrayRef<int64_t>(kDynamics, 2));
    }

    return success();
  }
};

// convert prefetch_tile to xegpu::PrefetchNdOp if the tile is for
// blocked load/store on global memory, or xegpu::PrefetchOp if the
// tile is for scattered op on global memory. Otherwise, drop it.
class PrefetchOpPattern : public OpConversionPattern<xetile::PrefetchTileOp> {
public:
  using OpConversionPattern<xetile::PrefetchTileOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace =
        memSpaceAttr ? memSpaceAttr.getValue() : xegpu::MemorySpace::Global;
    auto scatterAttr = tileTy.getScatterAttr();
    auto isScattered = scatterAttr ? scatterAttr.getValue() : false;

    auto [L1, L2, L3] = getCachePolicy(op);

    // Accesses to SLM doesn't need to be prefetched.
    if (memSpace == xegpu::MemorySpace::SLM)
      rewriter.eraseOp(op);

    auto tile = adaptor.getTile();
    if (isScattered) {
      rewriter.replaceOpWithNewOp<xegpu::PrefetchOp>(op, tile, L1, L2, L3);
    } else {
      rewriter.replaceOpWithNewOp<xegpu::PrefetchNdOp>(op, tile, L1, L2, L3);
    }

    return success();
  }
};

// convert load_tile to xegpu::LoadNdOp if the tile is for blocked
// load, or xegpu::LoadGatherOp if the tile is for scattered load
// (for SLM access).
class LoadOpPattern : public OpConversionPattern<xetile::LoadTileOp> {
public:
  using OpConversionPattern<xetile::LoadTileOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tileTy = op.getTile().getType();
    auto arrayLengthAttr = tileTy.getArrayLength();
    auto arrayLength = arrayLengthAttr ? arrayLengthAttr.getInt() : 1;
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace =
        memSpaceAttr ? memSpaceAttr.getValue() : xegpu::MemorySpace::Global;

    // TODO: enable SLM support for col-major access
    if (memSpace == xegpu::MemorySpace::SLM &&
        isColMajorOrder(tileTy.getOrder()))
      return failure();

    auto vecTy = dyn_cast<VectorType>(op.getType(0));

    if (memSpace == xegpu::MemorySpace::SLM) {
      auto elemTy = vecTy.getElementType();
      auto vnni = getVnniFactor(elemTy);
      if (vnni > 1) {
        elemTy = isa<IntegerType>(elemTy) ? (Type)rewriter.getI32Type()
                                          : (Type)rewriter.getF32Type();
      }
      if (!isColMajorOrder(tileTy.getOrder())) {
        vecTy = VectorType::get(vecTy.getNumElements() / vnni, elemTy);
      }
    } else if (arrayLength > 1) {
      llvm::SmallVector<int64_t> shape({arrayLength});
      shape.append(vecTy.getShape().begin(), vecTy.getShape().end());
      vecTy = VectorType::get(shape, vecTy.getElementType());
    }

    auto [L1, L2, L3] = getCachePolicy(op);
    auto packAttr = UnitAttr();
    auto transAttr = DenseI64ArrayAttr();
    auto bitWidthAttr = IntegerAttr();
    auto ldOp = rewriter.create<xegpu::LoadNdOp>(loc, vecTy, adaptor.getTile(),
                                                 packAttr, transAttr,
                                                 bitWidthAttr, L1, L2, L3);

    llvm::SmallVector<Value> results({ldOp.getResult()});
    if (memSpace == xegpu::MemorySpace::SLM) {
      if (!isColMajorOrder(tileTy.getOrder())) {
        auto value = results.pop_back_val();
        auto elemTy = tileTy.getElementType();
        auto castTy = VectorType::get(tileTy.getNumElements(), elemTy);
        if (castTy != vecTy)
          value = rewriter.create<vector::BitCastOp>(loc, castTy, value);
        if (castTy != op.getType(0))
          value =
              rewriter.create<vector::ShapeCastOp>(loc, op.getType(0), value);
        results.push_back(value);
      } else {
        return failure();
      }
    } else if (arrayLength > 1) {
      auto value = results.pop_back_val();
      for (auto i = 0; i < arrayLength; ++i) {
        auto extractOp = rewriter.create<vector::ExtractOp>(loc, value, i);
        results.push_back(extractOp.getResult());
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

// convert xetile.load to xegpu::LoadGatherOp.
class GatherOpPattern : public OpConversionPattern<xetile::LoadGatherOp> {
public:
  using OpConversionPattern<xetile::LoadGatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::LoadGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getValue().getType();
    auto elemTy = type.getElementType();
    auto ldTy = VectorType::get(type.getNumElements(), elemTy);
    auto maskTy =
        VectorType::get(type.getNumElements(), rewriter.getIntegerType(1));
    auto [L1, L2, L3] = getCachePolicy(op);
    auto mask =
        rewriter.create<vector::ShapeCastOp>(loc, maskTy, adaptor.getMask());
    auto ldOp = rewriter.create<xegpu::LoadGatherOp>(
        loc, ldTy, adaptor.getTile(), mask, L1, L2, L3);
    auto v = rewriter.create<vector::ShapeCastOp>(loc, op.getType(), ldOp);
    rewriter.replaceOp(op, v);
    return success();
  }
};

// convert store_tile to xegpu::StoreNdOp if the tile is for blocked store.
// Otherwise, convert it to xegpu::StoreScatterOp (for SLM access).
class StoreOpPattern : public OpConversionPattern<xetile::StoreTileOp> {
public:
  using OpConversionPattern<xetile::StoreTileOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tileTy = op.getTile().getType();
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace =
        memSpaceAttr ? memSpaceAttr.getValue() : xegpu::MemorySpace::Global;

    auto [L1, L2, L3] = getCachePolicy(op, xegpu::CachePolicy::WRITE_BACK);
    auto value = adaptor.getValue();

    auto order = tileTy.getOrder();
    if (memSpace == xegpu::MemorySpace::SLM && isColMajorOrder(order)) {
      // Since the low-level instruction works on 32-bits data, the data need to
      // be vnni transformed and bitcasted. e.g., vector<4x4xf16> ->
      // vector<2x4xf32>
      value = convertToPackedVector(value, loc, rewriter,
                                    op.getValue().hasOneUse());
      auto maskTy = VectorType::get(tileTy.getShape()[1], rewriter.getI1Type());
      auto mask = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(maskTy, rewriter.getBoolAttr(true)));

      if (tileTy.getRank() > 1) {
        SmallVector<int64_t> permutation = llvm::to_vector(
            llvm::reverse(llvm::seq<int64_t>(tileTy.getRank())));
        value = rewriter.create<vector::TransposeOp>(loc, value, permutation);
      }

      rewriter.replaceOpWithNewOp<xegpu::StoreScatterOp>(
          op, value, adaptor.getTile(), mask, L1, L2, L3);
    } else {
      // Since the low-level instruction works on 1D vector of 32-bits data, the
      // data to be stored need to be linearized and bitcasted.
      // e.g., vector<4x4xf16> -> vector<8xf32>
      if (memSpace == xegpu::MemorySpace::SLM)
        value = convertTo1D32BitVector(value, loc, rewriter);
      rewriter.replaceOpWithNewOp<xegpu::StoreNdOp>(
          op, value, adaptor.getTile(), L1, L2, L3);
    }
    return success();
  }
};

// convert xetile.store to xegpu::StoreScatterOp.
class ScatterOpPattern : public OpConversionPattern<xetile::StoreScatterOp> {
public:
  using OpConversionPattern<xetile::StoreScatterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::StoreScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.getValue();
    auto tdesc = adaptor.getTile();
    auto mask = adaptor.getMask();

    auto tileTy = op.getTile().getType();
    auto numElems = tileTy.getNumElements();
    auto valTy = VectorType::get(numElems, tileTy.getElementType());
    auto maskTy = VectorType::get(numElems, rewriter.getIntegerType(1));
    auto [L1, L2, L3] = getCachePolicy(op, xegpu::CachePolicy::WRITE_BACK);
    mask = rewriter.create<vector::ShapeCastOp>(op.getLoc(), maskTy, mask);
    value = rewriter.create<vector::ShapeCastOp>(op.getLoc(), valTy, value);
    rewriter.replaceOpWithNewOp<xegpu::StoreScatterOp>(op, value, tdesc, mask,
                                                       L1, L2, L3);
    return success();
  }
};

class AtomicRMWOpPattern : public OpConversionPattern<xetile::AtomicRMWOp> {
public:
  using OpConversionPattern<xetile::AtomicRMWOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getValue().getType();
    auto elemTy = type.getElementType();
    auto value = adaptor.getValue();
    auto valTy = VectorType::get(type.getNumElements(), elemTy);
    auto maskTy =
        VectorType::get(type.getNumElements(), rewriter.getIntegerType(1));
    llvm::SmallVector<bool> maskValues(type.getNumElements(), true);
    auto maskAttr = DenseElementsAttr::get(maskTy, maskValues);
    Value mask = rewriter.create<arith::ConstantOp>(loc, maskTy, maskAttr);
    value = rewriter.create<vector::ShapeCastOp>(loc, valTy, value);
    auto rmwOp = rewriter.create<xegpu::AtomicRMWOp>(
        loc, valTy, op.getKind(), adaptor.getTile(), mask, value);
    auto v = rewriter.create<vector::ShapeCastOp>(loc, op.getType(), rmwOp);
    rewriter.replaceOp(op, v);
    return success();
  }
};

// convert xetile.mma to xegpu::DpasOp.
class MMAOpPattern : public OpConversionPattern<xetile::TileMMAOp> {
public:
  using OpConversionPattern<xetile::TileMMAOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xegpu::DpasOp>(op, op.getType(), adaptor.getA(),
                                               adaptor.getB(), adaptor.getC());
    return success();
  }
};

// convert xetile.broadcast. It will be coverted to vector::BroadcastOp
// if the broadcast dim is 0; otherwise, it will be converted to
// vector::InsertStridedSliceOp.
class BroadcastOpPattern : public OpConversionPattern<xetile::BroadcastOp> {
public:
  using OpConversionPattern<xetile::BroadcastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = op.getResult().getType();
    auto dim = op.getBroadcastDim();
    if (dim.size() != 1 || resTy.getRank() != 2 || dim[0] > 1)
      return rewriter.notifyMatchFailure(
          op, "Only support reduction with one broadcast dim on 2D vector.");

    if (dim[0] == 0) {
      auto srcTy = op.getSource().getType();
      auto newTy =
          VectorType::get(srcTy.getNumElements(), srcTy.getElementType());
      auto cast = rewriter.create<vector::ShapeCastOp>(op.getLoc(), newTy,
                                                       adaptor.getSource());
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, resTy, cast);
      return success();
    }

    if (dim[0] == 1) {
      auto srcTy = op.getSource().getType();
      auto elemTy = srcTy.getElementType();
      auto attr = elemTy.isInteger()
                      ? (Attribute)rewriter.getIntegerAttr(elemTy, 0)
                      : (Attribute)rewriter.getFloatAttr(elemTy, 0.0);

      Value result = rewriter.create<arith::ConstantOp>(
          op.getLoc(), resTy, DenseElementsAttr::get(resTy, attr));

      for (int64_t j = 0; j < resTy.getShape()[1]; j++) {
        result = rewriter.create<vector::InsertStridedSliceOp>(
            op.getLoc(), adaptor.getSource(), result,
            llvm::ArrayRef<int64_t>({0, j}), llvm::ArrayRef<int64_t>({1, 1}));
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    return failure();
  }
};

// convert xetile.reduce to vector::MultiDimReductionOp.
class ReduceOpPattern : public OpConversionPattern<xetile::ReductionOp> {
public:
  using OpConversionPattern<xetile::ReductionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::ReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = op.getResult().getType();
    auto newTy =
        VectorType::get(resTy.getNumElements(), resTy.getElementType());
    auto acc = rewriter.create<arith::ConstantOp>(
        op.getLoc(), newTy, DenseElementsAttr::get(newTy, 0));
    auto result = rewriter.replaceOpWithNewOp<vector::MultiDimReductionOp>(
        op, op.getType(), op.getKindAttr(), adaptor.getSource(), acc,
        op.getReductionDimsAttr());
    rewriter.replaceOp(op, result);
    return success();
  }
};

// convert xetile.transpose to vector::TransposeOp.
class TransposeOpPattern : public OpConversionPattern<xetile::TransposeOp> {
public:
  using OpConversionPattern<xetile::TransposeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xetile::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(op, adaptor.getVector(),
                                                     adaptor.getPermutation());
    return success();
  }
};

// update SCF ForOp, majory convert the argument type from TileType to
// TensorDescType
class SCFForOpPattern : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                             op.getUpperBound(), op.getStep(),
                                             adaptor.getInitArgs());
    Block *newBlock = newOp.getBody();
    // remove the terminator of the new block
    if (newBlock->mightHaveTerminator())
      rewriter.eraseOp(newBlock->getTerminator());

    Block *block = op.getBody();
    TypeConverter::SignatureConversion mapping(block->getNumArguments());
    for (auto [i, ty] : llvm::enumerate(newBlock->getArgumentTypes()))
      mapping.addInputs(i, ty);
    block = rewriter.applySignatureConversion(block, mapping);
    rewriter.mergeBlocks(block, newBlock, newBlock->getArguments());
    rewriter.replaceOp(op, newOp);
    return success();
  };
};

// update SCF YieldOp, update the operands
class SCFYieldOpPattern : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getResults());
    return success();
  }
};

// TODO: this is a temporary solution to support memref::ViewOp.
// Since the upstream doesn't have lowering pattern for converting
// memref::ViewOp to SPIRV, so here we convert it with alloc instead.
// But it requires every alloc just has one view. It should be removed
// after enable the support in MemrefToSPIRV.
class MemRefViewOpPattern final : public OpConversionPattern<memref::ViewOp> {
public:
  using OpConversionPattern<memref::ViewOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto memrefTy = op.getType();

    if (!memrefTy.hasStaticShape() || converter->isLegal(memrefTy))
      return failure();

    // for simplicity, make sure source is an alloc op, and only has one use,
    // otherwise skip it, since it is hard to guarantee the correctness.
    auto src = op.getSource();
    if (!isa<memref::AllocOp>(src.getDefiningOp()) || !src.hasOneUse())
      return failure();

    auto newTy = converter->convertType<MemRefType>(memrefTy);
    auto alignmentAttr = rewriter.getI64IntegerAttr(32);
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newTy, alignmentAttr);
    return success();
  }
};

class MemRefReinterpretCastOpPattern final
    : public OpConversionPattern<memref::ReinterpretCastOp> {
public:
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    op->setOperand(0, adaptor.getSource());
    rewriter.finalizeOpModification(op);
    return success();
  }
};

class MemRefTransposeOpPattern final
    : public OpConversionPattern<memref::TransposeOp> {
public:
  using OpConversionPattern<memref::TransposeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }
};

class XeTileConversionTarget : public ConversionTarget {
public:
  explicit XeTileConversionTarget(MLIRContext &context,
                                  std::shared_ptr<XeuArchInterface> ptruArch)
      : ConversionTarget(context) {

    this->uArchInterface = ptruArch;

    addDynamicallyLegalOp<xegpu::DpasOp>([&](Operation *op) -> bool {
      return (uArchInterface && succeeded(uArchInterface->isLegalDpasOp(op)));
    });

    addDynamicallyLegalOp<xegpu::LoadNdOp>([&](Operation *op) -> bool {
      return (uArchInterface && succeeded(uArchInterface->isLegalLoad2dOp(op)));
    });

    addDynamicallyLegalOp<xegpu::StoreNdOp>([&](Operation *op) -> bool {
      return (uArchInterface &&
              succeeded(uArchInterface->isLegalStore2dOp(op)));
    });

    addDynamicallyLegalOp<xegpu::PrefetchNdOp>([&](Operation *op) -> bool {
      return (uArchInterface &&
              succeeded(uArchInterface->isLegalPrefetch2dOp(op)));
    });

    addDynamicallyLegalOp<memref::ViewOp>([&](Operation *op) -> bool {
      auto viewOp = dyn_cast<memref::ViewOp>(op);
      auto memrefTy = viewOp.getType();
      auto byteShift = viewOp.getByteShift();
      auto sizes = viewOp.getSizes();
      if (sizes.size() > 0 || !isConstantIntValue(byteShift, 0))
        return true;
      auto memSpace =
          dyn_cast_if_present<IntegerAttr>(memrefTy.getMemorySpace());
      if (!memSpace || memSpace.getValue() != 3)
        return true;
      return memrefTy.getRank() != 2;
    });

    addDynamicallyLegalOp<memref::ReinterpretCastOp>(
        [&](Operation *op) -> bool {
          auto castOp = dyn_cast<memref::ReinterpretCastOp>(op);
          auto memrefTy = castOp.getSource().getType();
          auto memSpace =
              dyn_cast_if_present<IntegerAttr>(memrefTy.getMemorySpace());
          if (!memSpace || memSpace.getValue() != 3)
            return true;
          return memrefTy.getRank() != 2;
        });

    addIllegalOp<memref::TransposeOp>();
    addIllegalDialect<imex::xetile::XeTileDialect>();
    addLegalDialect<xegpu::XeGPUDialect>();
    addLegalOp<vector::ShapeCastOp>();
    markUnknownOpDynamicallyLegal([&](Operation *op) {
      for (auto ty : op->getResultTypes()) {
        if (isa<xetile::TileType>(ty))
          return false;
      }
      for (auto ty : op->getOperandTypes()) {
        if (isa<xetile::TileType>(ty))
          return false;
      }
      return true;
    });
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

// Full Pass
struct ConvertXeTileToXeGPUPass // convert XeTile to XeGPU
    : public imex::impl::ConvertXeTileToXeGPUBase<ConvertXeTileToXeGPUPass> {
  ConvertXeTileToXeGPUPass() = default;

  ConvertXeTileToXeGPUPass(const std::string &deviceName) {
    if (this->device.getNumOccurrences() == 0) {
      this->device = deviceName;

      if (deviceName == "pvc") {
        uArchInterface = std::make_shared<XePVCuArch>();
      }
    }
  }

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const llvm::Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler)))
      return failure();
    if (device == "pvc")
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    else
      return errorHandler(llvm::Twine("Invalid device: ") + device);
    return success();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    MLIRContext &context = getContext();

    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    if (!uArchInterface) {
      mod.emitOpError("Can not get GPU Arch Definition for given Arch param");
      return signalPassFailure();
    }

    XeTileConversionTarget target(context, uArchInterface);
    RewritePatternSet patterns(&context);

    TypeConverter typeConverter;

    typeConverter.addConversion([&](Type type) -> Type { return type; });

    typeConverter.addConversion([&](MemRefType type) -> MemRefType {
      auto elemTy = type.getElementType();
      auto memSpace = dyn_cast_if_present<IntegerAttr>(type.getMemorySpace());
      if (!memSpace || memSpace.getValue() != 3 || !elemTy.isIntOrFloat() ||
          type.getRank() != 2)
        return type;

      int vnni = getVnniFactor(elemTy);
      if (vnni > 1) {
        elemTy = isa<IntegerType>(elemTy)
                     ? (Type)IntegerType::get(type.getContext(), 32)
                     : (Type)Float32Type::get(type.getContext());
      }
      auto numElems = type.getNumElements() / vnni;
      return MemRefType::get(numElems, elemTy, MemRefLayoutAttrInterface(),
                             memSpace);
    });

    typeConverter.addConversion(
        [&](xetile::TileType type) -> xegpu::TensorDescType {
          auto context = type.getContext();
          auto scatterAttr = type.getScatterAttr();
          bool isScattered = scatterAttr ? scatterAttr.getValue() : false;

          // by default the targetTy is the element type, except for SLM cases,
          // where the data will be treated as 32-bit type implicitly.
          Type targetTy = type.getElementType();

          xegpu::LayoutAttr sgMap = nullptr;
          if (auto attr = type.getSgMap()) {
            auto layout = attr.getWiLayout().asArrayRef();
            auto data = attr.getWiData().asArrayRef();
            sgMap = xegpu::LayoutAttr::get(context, layout, data);
          }

          auto memSpaceAttr = convertMemorySpace(type.getMemorySpace());
          auto memSpace = memSpaceAttr ? memSpaceAttr.getValue()
                                       : xegpu::MemorySpace::Global;

          Attribute encoding;
          llvm::SmallVector<int64_t> shape;
          if (isScattered) {
            // Scattered tile is lowered to scattered tensor_desc with chunk
            // size 1. It supports both global memory and shared memory. while
            // scattered tile can support 2D shape, scattered tensor_desc only
            // support 1D shape.
            encoding = xegpu::ScatterTensorDescAttr::get(context, memSpace, 1);
            shape.push_back(type.getNumElements());
          } else if (memSpace == xegpu::MemorySpace::Global) {
            // Blocked tile on global memory is lowered to blocked tensor_desc
            // with the same shape.
            auto arrayLenAttr = type.getArrayLength();
            auto arrayLen = arrayLenAttr ? arrayLenAttr.getInt() : 1;
            encoding = xegpu::BlockTensorDescAttr::get(context, memSpace,
                                                       arrayLen, true);
            shape = llvm::to_vector(type.getShape());
          } else {
            // for TileType created for SLM access, it will be converted into:
            // 1. a 1D block tensor_desc if it is for row-major access
            // 2. a scattered tensor_desc if it is for col-major access.
            auto elemBits = type.getElementType().getIntOrFloatBitWidth();
            auto vnniFactor = std::max<int>(32 / elemBits, 1);

            // SLM access only supports 32-bit or 64-bit data type, so convert
            // the type if original element type is less than 32-bit.
            if (elemBits < 32)
              targetTy = type.getElementType().isInteger()
                             ? (Type)IntegerType::get(context, 32)
                             : (Type)Float32Type::get(context);

            if (isColMajorOrder(type.getOrder())) {
              // For access with col-major order
              auto chunkSize = type.getShape()[0] / vnniFactor;
              encoding = xegpu::ScatterTensorDescAttr::get(context, memSpace,
                                                           chunkSize);
              shape = {type.getShape()[1], chunkSize};
            } else {
              // For access with row-major order
              auto vecSize = type.getNumElements() / vnniFactor;
              encoding = xegpu::BlockTensorDescAttr::get(
                  context, memSpace, 1 /*array_len*/, true /*boundary_check*/);
              shape.push_back(vecSize);
            }
          }
          return xegpu::TensorDescType::get(context, shape, targetTy, encoding,
                                            sgMap);
        });

    auto materializeWithCast = [&](OpBuilder &builder, Type type,
                                   ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1 && "Expecting single input");
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };

    typeConverter.addTargetMaterialization(materializeWithCast);
    typeConverter.addSourceMaterialization(materializeWithCast);

    populateXeTileToXeGPUConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

/// Populate the given list with patterns that convert XeTile to XeGPU
void populateXeTileToXeGPUConversionPatterns(TypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.add<InitOpPattern, UpdateOpPattern, PrefetchOpPattern, LoadOpPattern,
               StoreOpPattern, GatherOpPattern, ScatterOpPattern, MMAOpPattern,
               BroadcastOpPattern, ReduceOpPattern, TransposeOpPattern,
               SCFForOpPattern, SCFYieldOpPattern, MemRefViewOpPattern,
               MemRefTransposeOpPattern, MemRefReinterpretCastOpPattern,
               AtomicRMWOpPattern>(converter, patterns.getContext());
}

/// Create a pass that convert XeTile to XeGPU
std::unique_ptr<::OperationPass<::gpu::GPUModuleOp>>
createConvertXeTileToXeGPUPass(const std::string &deviceName) {
  return std::make_unique<ConvertXeTileToXeGPUPass>(deviceName);
}

} // namespace imex
