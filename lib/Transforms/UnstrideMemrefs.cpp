//===- UnstrideMemrefs.cpp - UnstrideMemrefs Pass  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file converts the memref.allocs for device side to gpu.allocs to
/// distinguish between host & device side memory allocations.
/// The pass traverses all the memref (load/store) operations inside the gpu
/// launch op in the IR and checks for its aliases and its defining op. If the
/// defining op is a memref.alloc op it replaces that op in the IR with
/// gpu.alloc op, because all the operations under the gpu.launch op are device
/// side computations and will execute on the device.
///
//===----------------------------------------------------------------------===//

//#include <imex/Transforms/Passes.h>

#include "PassDetail.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallBitVector.h>

using namespace mlir;
using namespace imex;

namespace imex {

// Find producer op for "val" recursively up the parent block chain
// and set insertion point to right after the producer.
static void setInsertionPointToStart(mlir::OpBuilder &builder,
                                     mlir::Value val) {
  if (auto parentOp = val.getDefiningOp()) {
    builder.setInsertionPointAfter(parentOp);
  } else {
    builder.setInsertionPointToStart(val.getParentBlock());
  }
}


static mlir::Value getFlatIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value memref, mlir::ValueRange indices) {
  auto memrefType = memref.getType().cast<mlir::MemRefType>();
  auto rank = static_cast<unsigned>(memrefType.getRank());
  assert(indices.size() == rank);
  auto shape = memrefType.getShape();
  auto expr =
    mlir::makeCanonicalStridedLayoutExpr(shape, builder.getContext());
  llvm::SmallVector<mlir::Value> applyOperands;
  if (rank != 0) {
      applyOperands.reserve(rank * 2);
      applyOperands.assign(indices.begin(), indices.end());
      mlir::OpBuilder::InsertionGuard g(builder);
      setInsertionPointToStart(builder, memref);
      mlir::Value size;
      for (auto i : llvm::seq(0u, rank - 1)) {
        auto dimInd = rank - i - 1;
        auto dim =
            builder.createOrFold<mlir::memref::DimOp>(loc, memref, dimInd);
        if (i != 0) {
          size = builder.createOrFold<mlir::arith::MulIOp>(loc, size, dim);
        } else {
          size = dim;
        }

        applyOperands.emplace_back(size);
      }
    }
    auto affineMap = mlir::AffineMap::get(
        rank, static_cast<unsigned>(applyOperands.size()) - rank, expr);
    assert(affineMap.getNumDims() == indices.size());
    return builder.createOrFold<mlir::AffineApplyOp>(loc, affineMap,
                                                     applyOperands);
}

/*
static mlir::Value getFlatIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value memref,
                                llvm::ArrayRef<mlir::OpFoldResult> indices) {
  llvm::SmallVector<mlir::Value> vals(indices.size());
  for (auto it : llvm::enumerate(indices)) {
    auto i = it.index();
    auto val = it.value();
    if (auto attr = val.dyn_cast<mlir::Attribute>()) {
      auto ind = attr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
      vals[i] = builder.create<mlir::arith::ConstantIndexOp>(loc, ind);
    } else {
      vals[i] = val.get<mlir::Value>();
    }
  }
  return getFlatIndex(builder, loc, memref, vals);
}
*/

static mlir::Value getFlatMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value memref) {
  auto memrefType = memref.getType().cast<mlir::MemRefType>();
  auto resultType = mlir::MemRefType::get(mlir::ShapedType::kDynamicSize,
                                          memrefType.getElementType());
  mlir::OpBuilder::InsertionGuard g(builder);
  setInsertionPointToStart(builder, memref);
  mlir::OpFoldResult offset = builder.getIndexAttr(0);
  //mlir::OpFoldResult size =
  //    builder.createOrFold<imex::util::UndefOp>(loc, builder.getIndexType());

  // Fake place holder
  mlir::OpFoldResult size = builder.getIndexAttr(0);
  mlir::OpFoldResult stride = builder.getIndexAttr(1);
  return builder.createOrFold<mlir::memref::ReinterpretCastOp>(
      loc, resultType, memref, offset, size, stride);
}

// Helper function for determining if memref can be flattened
// Current criteria
// Rank greater than 1 and has identity map
static bool canFlatten(mlir::Value val) {
  auto type = val.getType().cast<mlir::MemRefType>();
  return (type.getRank() > 1) && type.getLayout().isIdentity();
}

struct FlattenLoad : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.memref();
    if (!canFlatten(memref))
      return mlir::failure();

    auto loc = op.getLoc();
    auto flatIndex = getFlatIndex(rewriter, loc, memref, op.indices());
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, flatMemref,
                                                      flatIndex);
    return mlir::success();
  }
};

struct FlattenStore : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.memref();
    if (!canFlatten(memref))
      return mlir::failure();

    auto loc = op.getLoc();
    auto flatIndex = getFlatIndex(rewriter, loc, memref, op.indices());
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, op.value(),
                                                       flatMemref, flatIndex);
    return mlir::success();
  }
};

// Turns memref dialect ops with strided access pattern into a flat 1-D access
// Ops that get converted are Load, Store and Subview
struct UnstrideMemrefsPass
    : public UnstrideMemrefsPassBase<UnstrideMemrefsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<FlattenLoad, FlattenStore>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createUnstrideMemrefsPass() {
  return std::make_unique<UnstrideMemrefsPass>();
}
} // namespace imex
