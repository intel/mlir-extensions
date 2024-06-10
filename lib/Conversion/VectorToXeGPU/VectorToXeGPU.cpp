//===- VectorToXeGPU.cpp - VectorToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the VectorToXeGPU conversion, converting the Vector
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/VectorToXeGPU/VectorToXeGPU.h>
#include <imex/Utils/PassWrapper.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>

#include <mlir/IR/BuiltinOps.h>

#include "../PassDetail.h"
#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace imex {

namespace {

class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

struct MyTarget : public ConversionTarget {
  MyTarget(MLIRContext &ctx) : ConversionTarget(ctx) {

    /// Mark `cf.br` and `cf.cond_br` as illegal.
    addIllegalOp<vector::TransferReadOp>(); //, vector::TransferWriteOp
  }
};

// *******************************
// ***** Individual patterns *****
// *******************************

// Goal: vector.transfer_read -> xegpu.create_nd_tdesc + xegpu.load_nd
// E.g. translate 
//   %3 = vector.transfer_read %arg1[%0, %2], %arg2 : memref<512x640xf32>,
//   vector<1x32xf32> to %desc = xegpu.create_nd_tdesc %arg1[%0, %2] {mode = vc}
//   : memref<512x640xf32> -> !xegpu.tensor_desc<32xf32>
// to
//   %4 = xegpu.load_nd %3 {mode = vc}: !xegpu.tensor_desc<32xf32> ->
//     vector<32xf32>
//   %5 = vector.shape_cast %4 : vector<1x32xf32> to vector<32xf32>

struct TransferReadOpConverter
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    auto ctx = read->getContext();
    auto resultTile = read.getResult();      // %3 = ...
    auto resTileType = resultTile.getType(); // vector<32xf32>
    auto resTileShape = resTileType.getShape();
    auto rank = resTileType.getRank();
    auto source = read.getSource(); // memref<512x640xf32>
    // auto intermediateShape = resTileShape; // vector<1x32xf32>

    if (rank == 1) {
      // decltype(resTileShape) intermediateShape = {1, resTileShape[0]};
      auto intermediateType =
          VectorType::get({1, resTileShape[0]}, resTileType.getElementType());
      auto tDescTy = xegpu::TensorDescType::get({1, resTileShape[0]},
                                                resTileType.getElementType());
      mlir::SmallVector<mlir::OpFoldResult> tDescOffsets{read->getOperand(1),
                                                         read->getOperand(2)};

      rewriter.setInsertionPoint(read);
      mlir::Value desc;
      if (auto MemRefTypedSource =
              mlir::cast<mlir::TypedValue<mlir::MemRefType>>(source)) {
        desc = rewriter.create<mlir::xegpu::CreateNdDescOp>(
            read.getLoc(), tDescTy /*resultTy*/, MemRefTypedSource /*source*/,
            tDescOffsets /*offsets*/);
      } else {
        return mlir::failure();
      }
      mlir::IntegerAttr vnniAxisAttr;
      mlir::DenseI64ArrayAttr transposeAttr;
      mlir::IntegerAttr transposeBitWidthAttr;
      auto CACHED = mlir::xegpu::CachePolicy::CACHED;
      auto L1 = mlir::xegpu::CachePolicyAttr::get(ctx, CACHED);
      auto L2 = mlir::xegpu::CachePolicyAttr::get(ctx, CACHED);
      auto L3 = mlir::xegpu::CachePolicyAttr::get(ctx, CACHED);
      auto load = rewriter.create<xegpu::LoadNdOp>(
          read.getLoc(), intermediateType, desc, vnniAxisAttr, transposeAttr,
          transposeBitWidthAttr, L1, L2, L3);

      auto cast = rewriter.create<vector::ShapeCastOp>(
          read.getLoc(), resTileType, load->getResults());
      Operation *payload = cast;
      if (auto map = read.getPermutationMap(); map.isSingleConstant()) {
        SmallVector<int64_t> mask(resTileShape[0],
                                  map.getSingleConstantResult());
        payload =
            rewriter.create<vector::ShuffleOp>(read.getLoc(), cast, cast, mask);
      } else {
        AffineExpr d0, d1;
        bindDims(read.getContext(), d0, d1);
        auto mp = AffineMap::get(map.getNumDims(), 0, {d1}, read.getContext());
        // (d0, d1) -> (d1)
        if (map != mp) {
          // Unsupported permutation map
          return ::mlir::failure();
        }
      }
      rewriter.replaceOp(read, payload->getResults());
    } else {
      auto intermediateType =
          VectorType::get(resTileShape, resTileType.getElementType());
      auto tDescTy = xegpu::TensorDescType::get(resTileShape,
                                                resTileType.getElementType());
      mlir::SmallVector<mlir::OpFoldResult> tDescOffsets{read->getOperand(1),
                                                         read->getOperand(2)};

      rewriter.setInsertionPoint(read);
      mlir::Value desc;
      if (auto MemRefTypedSource =
              mlir::cast<mlir::TypedValue<mlir::MemRefType>>(source)) {
        desc = rewriter.create<mlir::xegpu::CreateNdDescOp>(
            read.getLoc(), tDescTy /*resultTy*/, MemRefTypedSource /*source*/,
            tDescOffsets /*offsets*/);
      } else {
        return mlir::failure();
      }
      mlir::IntegerAttr vnniAxisAttr;
      mlir::DenseI64ArrayAttr transposeAttr;
      mlir::IntegerAttr transposeBitWidthAttr;
      auto L1 = mlir::xegpu::CachePolicyAttr::get(
          ctx, mlir::xegpu::CachePolicy::CACHED);
      auto L2 = mlir::xegpu::CachePolicyAttr::get(
          ctx, mlir::xegpu::CachePolicy::CACHED);
      auto L3 = mlir::xegpu::CachePolicyAttr::get(
          ctx, mlir::xegpu::CachePolicy::CACHED);
      auto load = rewriter.create<xegpu::LoadNdOp>(
          read.getLoc(), intermediateType, desc, vnniAxisAttr, transposeAttr,
          transposeBitWidthAttr, L1, L2, L3);
      rewriter.replaceOp(read, load->getResults());
    }

    return ::mlir::success();
  }
};

// vector.transfer_write %5, %arg4[%0, %2] : vector<1x32xf32>,
// memref<512x640xf32> to %5 = vector.shape_cast %4 : vector<32xf32> to
// vector<1x32xf32> %desc2 = xegpu.create_nd_tdesc %arg4[%0, %2] {mode = vc} :
// memref<512x640xf32> -> !xegpu.tensor_desc<1x32xf32> xegpu.store_nd %5, %desc2
// {mode = vc} : vector<1x32xf32>, !xegpu.tensor_desc<1x32xf32>

struct TransferWriteOpConverter
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    auto ctx = write->getContext();
    auto resultTile = write->getOperand(0); //%5
    auto source = write.getSource();        // memref<512x640xi32>
    auto resTileType = dyn_cast<VectorType>(resultTile.getType());
    auto resTileShape = resTileType.getShape();
    auto rank = resTileType.getRank();
    auto intermediateType =
        VectorType::get({1, resTileShape[0]}, resTileType.getElementType());
    if (rank == 1) {
      auto tDescTy = xegpu::TensorDescType::get({1, resTileShape[0]},
                                                resTileType.getElementType());
      mlir::SmallVector<mlir::OpFoldResult> tDescOffsets{write->getOperand(2),
                                                         write->getOperand(3)};

      rewriter.setInsertionPoint(write);
      auto cast = rewriter.create<vector::ShapeCastOp>(
          write.getLoc(), intermediateType, write->getOperand(0));
      mlir::Value desc;
      if (auto MemRefTypedSource =
              mlir::cast<mlir::TypedValue<mlir::MemRefType>>(source)) {
        desc = rewriter.create<mlir::xegpu::CreateNdDescOp>(
            write.getLoc(), tDescTy /*resultTy*/, MemRefTypedSource /*source*/,
            tDescOffsets /*offsets*/);
      } else {
        return mlir::failure();
      }

      auto WRITE_BACK = mlir::xegpu::CachePolicy::WRITE_BACK;
      auto L1 = mlir::xegpu::CachePolicyAttr::get(ctx, WRITE_BACK);
      auto L2 = mlir::xegpu::CachePolicyAttr::get(ctx, WRITE_BACK);
      auto L3 = mlir::xegpu::CachePolicyAttr::get(ctx, WRITE_BACK);
      rewriter.create<xegpu::StoreNdOp>(write.getLoc(), cast, desc, L1, L2, L3);
      rewriter.eraseOp(write);
    } else {
      auto tDescTy = xegpu::TensorDescType::get(resTileShape,
                                                resTileType.getElementType());
      mlir::SmallVector<mlir::OpFoldResult> tDescOffsets{write->getOperand(2),
                                                         write->getOperand(3)};

      rewriter.setInsertionPoint(write);
      mlir::Value desc;
      if (auto MemRefTypedSource =
              mlir::cast<mlir::TypedValue<mlir::MemRefType>>(source)) {
        desc = rewriter.create<mlir::xegpu::CreateNdDescOp>(
            write.getLoc(), tDescTy /*resultTy*/, MemRefTypedSource /*source*/,
            tDescOffsets /*offsets*/);
      } else {
        return mlir::failure();
      }

      auto WRITE_BACK = mlir::xegpu::CachePolicy::WRITE_BACK;
      auto L1 = mlir::xegpu::CachePolicyAttr::get(ctx, WRITE_BACK);
      auto L2 = mlir::xegpu::CachePolicyAttr::get(ctx, WRITE_BACK);
      auto L3 = mlir::xegpu::CachePolicyAttr::get(ctx, WRITE_BACK);
      rewriter.create<xegpu::StoreNdOp>(write.getLoc(), write->getOperand(0),
                                        desc, L1, L2, L3);
      rewriter.eraseOp(write);
    }

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct ConvertVectorToXeGPUPass // convert Vector to XeGPU
    : public ::imex::ConvertVectorToXeGPUBase<ConvertVectorToXeGPUPass> {
  ConvertVectorToXeGPUPass() = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<TransferReadOpConverter, TransferWriteOpConverter>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

} // namespace

/// Populate the given list with patterns that convert Vector to XeGPU

/// Create a pass that convert Vector to XeGPU
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertVectorToXeGPUPass() {
  return std::make_unique<ConvertVectorToXeGPUPass>();
}

} // namespace imex
