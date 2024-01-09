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

#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h>

#include "ArithOpConversion.h"
#include "SCFOpConversion.h"
#include "XeTileOpConversion.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace imex {

// Sg-level XeTile::init_tile -> XeGPU::init_tile
class SgInitTileOpPattern
    : public SgXeTileToXeGPUConversion<xetile::InitTileOp> {
  using SgXeTileToXeGPUConversion<
      xetile::InitTileOp>::SgXeTileToXeGPUConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  XeGPUOneToNPatterRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto source = op.getSource();
    auto resultTile = op.getResult();
    auto resTileType = resultTile.getType();
    auto resTileShape = resTileType.getShape();
    auto indexType = rewriter.getIndexType();

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

    auto offsetsX = offsets[0];
    auto offsetsY = offsets[1];

    if (resTileType.getRank() != 4)
      return mlir::failure();

    auto createIndexConstant = [&](mlir::Type type, int64_t value) {
      auto attr = rewriter.getIndexAttr(value);
      return rewriter.create<mlir::arith::ConstantOp>(loc, type, attr);
    };

    auto tDescTy = xegpu::TensorDescType::get(
        {resTileShape[2], resTileShape[3]}, resTileType.getElementType());

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value> xegpuOps;
    for (int i = 0; i < resTileShape[0]; i++) {
      for (int j = 0; j < resTileShape[1]; j++) {
        auto subOffX = createIndexConstant(indexType, (resTileShape[2] * i));
        auto subOffY = createIndexConstant(indexType, (resTileShape[3] * j));
        auto tDescOffsetX =
            rewriter.createOrFold<mlir::arith::AddIOp>(loc, subOffX, offsetsX);
        auto tDescOffsetY =
            rewriter.createOrFold<mlir::arith::AddIOp>(loc, subOffY, offsetsY);
        mlir::SmallVector<mlir::OpFoldResult> tDescOffsets{tDescOffsetX,
                                                           tDescOffsetY};

        // TODO: this needs improvement, it assumes the source is static
        // memeref.
        auto createNdOp = rewriter.create<xegpu::CreateNdDescOp>(
            op.getLoc(), tDescTy /*resultTy*/, source /*source*/,
            tDescOffsets /*offsets*/, imex::xegpu::Mode::VC /*mode*/);

        xegpuOps.push_back(createNdOp);
      }
    }

    rewriter.replaceOp(op, xegpuOps);
    return mlir::success();
  }
};

// Sg-level XeTile::prefetch_tile -> XeGPU::prefetch_2d
struct SgPrefetchTileOpPattern
    : public SgXeTileToXeGPUConversion<xetile::PrefetchTileOp> {
  using SgXeTileToXeGPUConversion<
      xetile::PrefetchTileOp>::SgXeTileToXeGPUConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OpAdaptor adaptor,
                  XeGPUOneToNPatterRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    auto tiles = adaptor.getTile();
    if (tileTy.getRank() != 4)
      return mlir::failure();
    auto shape = tileTy.getShape();

    if (shape[0] * shape[1] != (int64_t)tiles.size()) {
      op.emitOpError("Failed to lower LoadTileOp because shape[0] * shape[1] "
                     "!= sources.size().");
      return mlir::failure();
    }

    auto L1 = xegpu::CacheReadHintAttr::get(op.getContext(),
                                            xegpu::CacheReadHint::CACHED);
    auto L2 = xegpu::CacheReadHintAttr::get(op.getContext(),
                                            xegpu::CacheReadHint::CACHED);
    auto L3 = xegpu::CacheReadHintAttr::get(op.getContext(),
                                            xegpu::CacheReadHint::CACHED);

    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        auto tile = tiles[i * shape[1] + j];
        rewriter.create<xegpu::PrefetchNDOp>(op.getLoc(), tile, L1, L2, L3,
                                             imex::xegpu::Mode::VC);
      }
    }

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

// Sg-level XeTile::load_tile -> XeGPU::load_2d
struct SgLoadTileOpPattern
    : public SgXeTileToXeGPUConversion<xetile::LoadTileOp> {
  using SgXeTileToXeGPUConversion<
      xetile::LoadTileOp>::SgXeTileToXeGPUConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  XeGPUOneToNPatterRewriter &rewriter) const override {
    auto resultTy = op.getValue().getType();
    auto tileTy = op.getSource().getType();

    if (resultTy.getRank() != 4 || tileTy.getRank() != 4)
      return mlir::failure();

    auto shape = resultTy.getShape();
    auto sources = adaptor.getSource();

    if (shape[0] * shape[1] != (int64_t)sources.size()) {
      op.emitOpError("Failed to lower LoadTileOp because shape[0] * shape[1] "
                     "!= sources.size().");
      return mlir::failure();
    }

    auto elementTy = resultTy.getElementType();

    // TODO: move these two into architecture abstracture in future.
    const int SIMD_WIDTH_IN_BITS = 32;
    int vnniFactor = SIMD_WIDTH_IN_BITS / elementTy.getIntOrFloatBitWidth();

    int vnniAxis = 1;
    mlir::IntegerAttr vnniAxisAttr;
    // FIXME : remove the usage of tranpose attribute and rely on order
    // attribute.
    mlir::DenseI64ArrayAttr transposeAttr;
    auto L1 = xegpu::CacheReadHintAttr::get(op.getContext(),
                                            xegpu::CacheReadHint::CACHED);
    auto L2 = xegpu::CacheReadHintAttr::get(op.getContext(),
                                            xegpu::CacheReadHint::CACHED);
    auto L3 = xegpu::CacheReadHintAttr::get(op.getContext(),
                                            xegpu::CacheReadHint::CACHED);

    llvm::SmallVector<int64_t> newShape = {shape[2], shape[3]};
    // needs vnni transform;
    if (vnniFactor > 1 && (isA(op) || isB(op))) {
      if (isB(op))
        vnniAxis = 0;
      newShape[vnniAxis] /= vnniFactor;
      newShape.push_back(vnniFactor);
      vnniAxisAttr =
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), vnniAxis);
    }

    auto subVectorTy =
        ::mlir::VectorType::get(newShape, resultTy.getElementType());

    rewriter.setInsertionPoint(op);

    llvm::SmallVector<::mlir::Value> xegpuOps;
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        auto tile = sources[i * shape[1] + j];
        // FIXME: (chencha3) it assumes arr_len is 1,
        // pending improvement in the new lowering passes.
        auto ldOp = rewriter.create<xegpu::LoadNDOp>(
            op.getLoc(), subVectorTy, tile, vnniAxisAttr, transposeAttr, L1, L2,
            L3, imex::xegpu::Mode::VC);
        xegpuOps.push_back(ldOp);
      }
    }

    rewriter.replaceOp(op, xegpuOps);
    return mlir::success();
  }
};

// Sg-level XeTile::store_tile -> XeGPU::store_2d
struct SgStoreTileOpPattern
    : public SgXeTileToXeGPUConversion<xetile::StoreTileOp> {
  using SgXeTileToXeGPUConversion<
      xetile::StoreTileOp>::SgXeTileToXeGPUConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  XeGPUOneToNPatterRewriter &rewriter) const override {
    auto tiles = adaptor.getTile();
    auto values = adaptor.getValue();

    if (tiles.size() != values.size()) {
      op.emitOpError() << "Failed to lower the StoreOp, because tile and block "
                          "size doesn't match."
                       << "tiles: " << tiles.size() << ", "
                       << "values: " << values.size() << "\n";
      return mlir::failure();
    }

    auto context = op.getContext();
    auto L1 = xegpu::CacheWriteHintAttr::get(context,
                                             xegpu::CacheWriteHint::WRITE_BACK);
    auto L2 = xegpu::CacheWriteHintAttr::get(context,
                                             xegpu::CacheWriteHint::WRITE_BACK);
    auto L3 = xegpu::CacheWriteHintAttr::get(context,
                                             xegpu::CacheWriteHint::WRITE_BACK);
    for (size_t i = 0; i < tiles.size(); i++)
      rewriter.create<xegpu::StoreNDOp>(op.getLoc(), tiles[i], values[i], L1,
                                        L2, L3, imex::xegpu::Mode::VC);

    rewriter.eraseOp(op);
    return ::mlir::success();
  }
};

// Sg-level XeTile::tile_mma-> XeGPU::dpas
struct SgTileMMAOpPattern
    : public SgXeTileToXeGPUConversion<xetile::TileMMAOp> {
  using SgXeTileToXeGPUConversion<xetile::TileMMAOp>::SgXeTileToXeGPUConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  XeGPUOneToNPatterRewriter &rewriter) const override {

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
          tmpC = rewriter.create<xegpu::DpasOp>(
              loc, subCTy /*result*/, aVec /*lhs*/, bVec /*rhs*/, tmpC /*acc*/,
              imex::xegpu::Mode::VC);
        }
        xegpuOps.push_back(tmpC);
      }
    }
    rewriter.replaceOp(op, xegpuOps);
    return mlir::success();
  }
};

struct SgUpdateTileOffsetOpPattern
    : public SgXeTileToXeGPUConversion<xetile::UpdateTileOffsetOp> {
  using SgXeTileToXeGPUConversion<
      xetile::UpdateTileOffsetOp>::SgXeTileToXeGPUConversion;

  mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  XeGPUOneToNPatterRewriter &rewriter) const override {
    auto offsetX = op.getOffsetX();
    auto offsetY = op.getOffsetY();
    auto tiles = adaptor.getTile();

    llvm::SmallVector<mlir::Value> xegpuOps;
    for (const auto &tile : tiles) {
      auto xegpuTile = rewriter.create<xegpu::UpdateNDOffsetOp>(
          op.getLoc(), tile.getType(), tile, mlir::ValueRange{offsetX, offsetY},
          imex::xegpu::Mode::VC);
      xegpuOps.push_back(xegpuTile);
    }
    rewriter.replaceOp(op, xegpuOps);
    return mlir::success();
  }
};

void populateXeTileOpConversionPatterns(imex::XeGPUTypeConverter &converter,
                                        mlir::RewritePatternSet &patterns) {
  patterns.insert<SgInitTileOpPattern, SgPrefetchTileOpPattern,
                  SgLoadTileOpPattern, SgStoreTileOpPattern, SgTileMMAOpPattern,
                  SgUpdateTileOffsetOpPattern>(patterns.getContext(),
                                               converter);
}

} // namespace imex
