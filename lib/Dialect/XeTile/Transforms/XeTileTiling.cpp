//===- LowerXeTileToBlockLayout.cppp - LowerXeTileToBlockLayout Pass  -------*-
// C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains lowering transformation for  XeTile large tiles into
/// smaller tiles with blocked layout that maps to register region.
/// This blocked layout is represented by high dimension vectors, inner
/// dimension matches to DPAS size config.
///
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <optional>

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/DebugUtils.h"

#include "PassDetail.h"
#include "XeTileTiling.h"

using namespace mlir;
using namespace imex;
namespace imex {
#define GEN_PASS_DEF_XETILETILING
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {

// DPAS block size as per HW config - TODO, define platform specific sizes
#define M_SIZE 8
#define K_SIZE 16
#define N_SIZE 16

struct ArithConstantOpPattern
    : public XeTileConversion<mlir::arith::ConstantOp> {
  using XeTileConversion<mlir::arith::ConstantOp>::XeTileConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = op.getResult();
    auto resultTy = result.getType();

    if (!resultTy.isa<mlir::VectorType>())
      return mlir::failure();

    auto vectorTy = resultTy.cast<mlir::VectorType>();

    // We only interesting 2D vectors, and the one used as C
    if (vectorTy.getRank() != 2)
      return mlir::failure();

    auto valueAttr = op.getValue();
    if (!valueAttr.isa<mlir::DenseElementsAttr>())
      return mlir::failure();

    auto denseElementsAttr = valueAttr.cast<mlir::DenseElementsAttr>();
    if (!denseElementsAttr.isSplat())
      return mlir::failure();

    auto splatVal = denseElementsAttr.getSplatValue<mlir::FloatAttr>();

    auto shape = vectorTy.getShape();

    // TODO: a place holder for getting inner_blocks
    llvm::SmallVector<int> inner_blocks = {M_SIZE, N_SIZE};

    // set blockSizes default to inner_block size
    llvm::SmallVector<int> blockSizes(inner_blocks);
    if (isA(op)) {
      blockSizes = {M_SIZE, K_SIZE};
    } else if (isB(op)) {
      blockSizes = {K_SIZE, N_SIZE};
    } else if (isRC(op)) {
      blockSizes = {M_SIZE, N_SIZE};
    }

    auto vecTy = ::mlir::VectorType::get({shape[0] / blockSizes[0],
                                          shape[1] / blockSizes[1],
                                          blockSizes[0], blockSizes[1]},
                                         vectorTy.getElementType());

    auto newOp = rewriter.create<mlir::arith::ConstantOp>(
        loc, vecTy, mlir::DenseElementsAttr::get(vecTy, splatVal));

    // rewriter.replaceOp(op, newOp);
    rewriter.replaceOpWithIf(op, newOp->getResults(), [&](mlir::OpOperand &op) {
      auto *owner = op.getOwner();

      // the direct user is an xetile operator
      if (llvm::isa<imex::xetile::XeTileDialect>(owner->getDialect()))
        return true;

      // the direct user is an scf::ForOp, but the corresponding argument
      // is used by an xetile operator
      if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(owner)) {
        auto arg = forOp.getTiedLoopRegionIterArg(&op);

        auto haveXeTileUsers = std::any_of(
            arg.user_begin(), arg.user_end(), [&](mlir::Operation *op) {
              return llvm::isa<imex::xetile::XeTileDialect>(op->getDialect());
            });

        if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(
                forOp.getRegion().front().getTerminator())) {
          auto idx = forOp.getTiedLoopResult(&op).getResultNumber();
          auto definingOp = yieldOp.getResults()[idx].getDefiningOp();
          haveXeTileUsers |=
              llvm::isa<imex::xetile::XeTileDialect>(definingOp->getDialect());
        }

        return haveXeTileUsers;
      }

      return false;
    });

    return mlir::success();
  }
};

struct SCFForOpPattern : public XeTileConversion<mlir::scf::ForOp> {
  using XeTileConversion<mlir::scf::ForOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());
    mlir::Block *block = op.getBody();
    mlir::Block *newBlock = newOp.getBody();
    newBlock->clear();
    rewriter.mergeBlocks(block, newBlock, newBlock->getArguments());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct InitTileOpPattern : public XeTileConversion<xetile::InitTileOp> {
  using XeTileConversion<xetile::InitTileOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    if (tileTy.getRank() != 2) {
      op.emitWarning(
          "Skipped InitTileOp because the result tile is not rank 2.\n");
      return mlir::failure();
    }

    auto shape = tileTy.getShape();

    // TODO: a place holder for getting inner_blocks
    llvm::SmallVector<int> inner_blocks = {M_SIZE, N_SIZE};

    // set blockSizes default to inner_block size
    llvm::SmallVector<int> blockSizes(inner_blocks);
    if (isA(op)) {
      blockSizes = {M_SIZE, K_SIZE};
    } else if (isB(op)) {
      blockSizes = {K_SIZE, N_SIZE};
    } else if (isRC(op)) {
      blockSizes = {M_SIZE, N_SIZE};
    }

    auto newTileTy = imex::xetile::TileType::get({shape[0] / blockSizes[0],
                                                  shape[1] / blockSizes[1],
                                                  blockSizes[0], blockSizes[1]},
                                                 tileTy.getElementType());

    auto newOp = rewriter.create<::imex::xetile::InitTileOp>(
        op.getLoc(), newTileTy, op.getSource(), op.getOffsets(),
        op.getStaticOffsetsAttr(), op.getDynamicShape(),
        op.getDynamicStrides());

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct LoadTileOpPattern : public XeTileConversion<xetile::LoadTileOp> {
  using XeTileConversion<xetile::LoadTileOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultTy = op.getResult().getType();

    if (resultTy.getRank() != 2) {
      op.emitWarning("skipped because the result is not 2D.");
      return mlir::failure();
    }

    auto shape = resultTy.getShape();

    // TODO: a place holder for getting inner_blocks
    llvm::SmallVector<int> inner_blocks = {M_SIZE, N_SIZE};

    // set blockSizes default to inner_block size
    llvm::SmallVector<int> blockSizes(inner_blocks);

    if (isA(op)) {
      blockSizes = {M_SIZE, K_SIZE};
    } else if (isB(op)) {
      blockSizes = {K_SIZE, N_SIZE};
    } else if (isRC(op)) {
      blockSizes = {M_SIZE, N_SIZE};
    }

    auto vecTy = ::mlir::VectorType::get({shape[0] / blockSizes[0],
                                          shape[1] / blockSizes[1],
                                          blockSizes[0], blockSizes[1]},
                                         resultTy.getElementType());

    auto newOp = rewriter.create<::imex::xetile::LoadTileOp>(
        loc, vecTy, adaptor.getSource(), op.getTransposeAttr(),
        op.getPaddingAttr());

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct StoreTileOpPattern : public XeTileConversion<xetile::StoreTileOp> {
  using XeTileConversion<xetile::StoreTileOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {

    auto newOp = rewriter.create<::imex::xetile::StoreTileOp>(
        op.getLoc(), adaptor.getValue(), adaptor.getTile());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct TileMMAOpPattern : public XeTileConversion<xetile::TileMMAOp> {
  using XeTileConversion<xetile::TileMMAOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultTy = op.getOutput().getType();

    if (resultTy.getRank() != 2) {
      op.emitWarning("skipped because the result is not 2D.");
      return mlir::failure();
    }

    auto shape = resultTy.getShape();
    auto vecTy = ::mlir::VectorType::get(
        {shape[0] / M_SIZE, shape[1] / N_SIZE, M_SIZE, N_SIZE},
        resultTy.getElementType());

    auto newOp = rewriter.create<imex::xetile::TileMMAOp>(
        loc, vecTy, adaptor.getA(), adaptor.getB(), adaptor.getC());

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct UpdateTileOffsetOpPattern
    : public XeTileConversion<xetile::UpdateTileOffsetOp> {
  using XeTileConversion<xetile::UpdateTileOffsetOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {

    auto newOp = rewriter.create<::imex::xetile::UpdateTileOffsetOp>(
        op.getLoc(), adaptor.getTile().getType(), adaptor.getTile(),
        adaptor.getOffsetX(), adaptor.getOffsetY());

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

void populateXeTileTilingPatterns(imex::XeTypeConverter &converter,
                                  mlir::RewritePatternSet &patterns) {

  patterns.insert<ArithConstantOpPattern, SCFForOpPattern, InitTileOpPattern,
                  LoadTileOpPattern, StoreTileOpPattern, TileMMAOpPattern,
                  UpdateTileOffsetOpPattern>(patterns.getContext(), converter);
}

// Lowers XeTile to blocked layout with high-dim vector
struct XeTileTilingPass
    : public imex::impl::XeTileTilingBase<imex::XeTileTilingPass> {

  XeTileTilingPass() = default;

public:
  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    auto mod = this->getOperation();

    // skip functions with XeTile.TileType inputs and outputs
    bool hasTileTyInFuncTy = false;
    mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp op) {
      auto funcTy = op.getFunctionType();
      hasTileTyInFuncTy |= std::any_of(
          funcTy.getInputs().begin(), funcTy.getInputs().end(),
          [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
      hasTileTyInFuncTy |= std::any_of(
          funcTy.getResults().begin(), funcTy.getInputs().end(),
          [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
    });

    if (hasTileTyInFuncTy) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    imex::ValueAttributeMap map;
    mod.walk<mlir::WalkOrder::PreOrder>([&](imex::xetile::TileMMAOp op) {
      markDefChainValues(op.getA(), OperandType::DPASA, map);
      markDefChainValues(op.getB(), OperandType::DPASB, map);
      if (bool(op.getC()))
        markDefChainValues(op.getC(), OperandType::DPASC, map);
      markUseChainValues(op.getOutput(), OperandType::DPASR, map);
    });

    mlir::RewritePatternSet patterns(&context);
    XeTypeConverter typeConverter(context, map);

    populateXeTileTilingPatterns(typeConverter, patterns);

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 2;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(
            applyPatternsAndFoldGreedily(mod, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

/// Create a pass
std::unique_ptr<::mlir::Pass> createXeTileTilingPass() {
  return std::make_unique<XeTileTilingPass>();
}
} // namespace imex
