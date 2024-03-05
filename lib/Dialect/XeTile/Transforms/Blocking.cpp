//===-------------- Blocking.cpp --------- Blocking Pass  -------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains lowering transformation for determing the problem size
/// that can be handled by an XeGPU operator (hardware instruction). XeTile
/// program can work one bigger problem size that cannot be handled by a
/// hardware instruction. But it needs to be decomposed into smaller pieces
/// such that each pieces can be handled by a hardware instruction.
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
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <optional>

#include "imex/Dialect/XeTile/Transforms/Blocking.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/DebugUtils.h"

#include "PassDetail.h"

using namespace mlir;
using namespace imex;
namespace imex {
#define GEN_PASS_DECL_XETILEBLOCKING
#define GEN_PASS_DEF_XETILEBLOCKING
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {

enum OpType { Prefetch, Load, Store };

// TODO: placeholder, replace it with uArch interface
template <OpType op>
llvm::SmallVector<int64_t> getInnerBlocks(mlir::Type elemTy, bool vnni = false,
                                          bool transpose = false) {
  assert(elemTy.isIntOrFloat());
  int bits = elemTy.getIntOrFloatBitWidth();
  if (op == OpType::Load && bits > 16 && vnni) {
    llvm::dbgs() << "load with VNNI for \"" << elemTy
                 << "\" is not supported.\n";
    return {};
  }

  if (op == OpType::Prefetch) {
    switch (bits) {
    case 8:
      // return {32, 256, 256};
      return {32, 256};
      break;
    case 16:
      // return {32, 128, 128};
      return {32, 128};
      break;
    case 32:
      // return {32, 64, 64};
      return {32, 64};
      break;
    case 64:
      // return {32, 32, 32};
      return {32, 32};
      break;
    default:
      llvm_unreachable("Unsupported data type.\n");
      break;
    }
  }

  if (op == OpType::Load) {
    switch (bits) {
    case 8:
      // return {32, 64, 64};
      return {32, 64};
      break;
    case 16:
      // return {32, 32, 32};
      return {32, 32};
      break;
    case 32:
      // return {32, 16, 16};
      return {32, 16};
      break;
    case 64:
      // return {32, 8, 8};
      return {32, 8};
      break;
    default:
      llvm_unreachable("Unsupported data type.\n");
      break;
    }
  }

  if (op == OpType::Store) {
    switch (bits) {
    case 8:
      // return {8, 64, 1};
      return {8, 64};
      break;
    case 16:
      // return {8, 32, 1};
      return {8, 32};
      break;
    case 32:
      // return {8, 16, 1};
      return {8, 16};
      break;
    case 64:
      // return {8, 8, 1};
      return {8, 8};
      break;
    default:
      llvm_unreachable("Unsupported data type.\n");
      break;
    }
  }
  llvm_unreachable("Unsupported.");
  return {};
}

// it blocks a constant dense value if it is used by XeTile operators,
// e.g, tile_mma and store_tile. It currently extends a 2D vector into
// 4D vector with the last 2 dim corresponding to block size.
// example: arith.constant dense<0.0>: vector<32x32xf16>
//      --> arith.constant dense<0.0>: vector<4x2x8x16xf16>
// [8, 16] is the block size.
struct ArithConstantOpPattern
    : public XeTileConversion<mlir::arith::ConstantOp> {
  using XeTileConversion<mlir::arith::ConstantOp>::XeTileConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!value || value.getType().getRank() != 2)
      return mlir::failure();

    auto shape = value.getType().getShape();
    auto blkSZ = getInnerBlocks<Load>(value.getElementType());
    auto newTy = mlir::VectorType::get(
        {shape[0] / blkSZ[0], shape[1] / blkSZ[1], blkSZ[0], blkSZ[1]},
        value.getElementType());

    value = value.reshape(newTy);
    auto newOp = rewriter.create<mlir::arith::ConstantOp>(loc, value);
    auto unpack = addUnpackOp(newOp, rewriter);

    // TODO: we may can replace it with standard replaceOp method when
    // we have full support for other non-xetile operators.
    rewriter.replaceOpWithIf(
        op, unpack->getResults(), [&](mlir::OpOperand &op) {
          auto *owner = op.getOwner();

          // the direct user is an xetile operator
          if (llvm::isa<xetile::XeTileDialect>(owner->getDialect()))
            return true;

          // the direct user is an scf::ForOp, but the corresponding argument
          // is used by an xetile operator
          if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(owner)) {
            auto arg = forOp.getTiedLoopRegionIterArg(&op);

            auto haveXeTileUsers = std::any_of(
                arg.user_begin(), arg.user_end(), [&](mlir::Operation *op) {
                  return llvm::isa<xetile::XeTileDialect>(op->getDialect());
                });

            if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(
                    forOp.getRegion().front().getTerminator())) {
              auto idx = forOp.getTiedLoopResult(&op).getResultNumber();
              auto definingOp = yieldOp.getResults()[idx].getDefiningOp();
              if (definingOp)
                haveXeTileUsers |=
                    llvm::isa<xetile::XeTileDialect>(definingOp->getDialect());
            }
            return haveXeTileUsers;
          }
          return false;
        });
    return mlir::success();
  }
};

// It rewrites the SCF forOp, it mainly updates the arguments of its
// regeion block. unpack ops are added for VectorType operands if needed.
struct SCFForOpPattern : public XeTileConversion<mlir::scf::ForOp> {
  using XeTileConversion<mlir::scf::ForOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    // preprocess the init args and add pack op if it is
    // defined by an unpack op.
    llvm::SmallVector<mlir::Value> newInitArgs;
    for (auto arg : adaptor.getInitArgs()) {
      if (auto defOp = arg.getDefiningOp<xetile::TileUnpackOp>()) {
        auto packOp = addPackOp(arg, defOp.getInnerBlocks(), rewriter);
        newInitArgs.push_back(packOp);
      } else {
        newInitArgs.push_back(arg);
      }
    }

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), newInitArgs);

    mlir::Block *block = op.getBody();
    mlir::Block *newBlock = newOp.getBody();
    rewriter.mergeBlocks(block, newBlock, newBlock->getArguments());

    llvm::SmallVector<mlir::Value> newValues;
    for (auto [i, result] : llvm::enumerate(newOp->getResults())) {
      // if corresponding init arg is updated with a pack op
      // an unpack op is needed for the result to make it
      // transparent to its users.
      if (newInitArgs[i].getDefiningOp<xetile::TilePackOp>()) {
        auto unpack = addUnpackOp(result, rewriter);
        newValues.push_back(unpack);
      } else {
        newValues.push_back(result);
      }
    }
    rewriter.replaceOp(op, newValues);
    return mlir::success();
  }
};

// It serves to insert pack ops for approriate vales if needed.
// for example, tile_mma result is vector<32x32xf16> (after unpack),
// but its corresponding argument in forOp is with type vector<1x2x32x16xf16>
// This op pattern will insert a pack op to make it consistent with the
// corresponding argument type.
struct SCFYieldOpPattern : public XeTileConversion<mlir::scf::YieldOp> {
  using XeTileConversion<mlir::scf::YieldOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto forOp = llvm::dyn_cast_if_present<mlir::scf::ForOp>(op->getParentOp());
    if (!forOp)
      return mlir::failure();

    llvm::SmallVector<mlir::Value> results;
    for (auto [i, value] : llvm::enumerate(adaptor.getResults())) {
      auto valTy = value.getType().dyn_cast<mlir::VectorType>();
      auto tgtTy =
          forOp.getRegionIterArg(i).getType().dyn_cast<mlir::VectorType>();
      if (valTy && tgtTy && valTy.getRank() == 2 && tgtTy.getRank() == 4) {
        auto innerBlock = tgtTy.getShape().take_back(2);
        auto pack = addPackOp(value, innerBlock, rewriter);
        results.push_back(pack);
      } else {
        results.push_back(value);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, results);

    return mlir::success();
  }
};

// It updates init_tile by attaching innerBlock attribute to the result
// tile. The block size is choosed based on how the tile is used, including
// prefetch, load, store. Since hardware support different sizes for them.
struct InitTileOpPattern : public XeTileConversion<xetile::InitTileOp> {
  using XeTileConversion<xetile::InitTileOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = op.getType();
    if (tileTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "Skipped InitTileOp because the result tile is not rank 2.\n");

    auto innerBlocks = tileTy.getInnerBlocks();

    // skip it if innerBlocks has been set by user or compiler.
    if (innerBlocks)
      return mlir::failure();

    auto elemTy = tileTy.getElementType();
    if (isForPrefetch(op)) {
      innerBlocks = mlir::DenseI64ArrayAttr::get(
          getContext(), getInnerBlocks<Prefetch>(elemTy));
    } else if (isForLoad(op)) {
      innerBlocks = mlir::DenseI64ArrayAttr::get(getContext(),
                                                 getInnerBlocks<Load>(elemTy));
    } else if (isForStore(op)) {
      innerBlocks = mlir::DenseI64ArrayAttr::get(getContext(),
                                                 getInnerBlocks<Store>(elemTy));
    } else {
      return rewriter.notifyMatchFailure(
          op, "The tile is used for multiple purpose. The init-duplicate pass "
              "should be run first to resolve this issue.");
    }

    auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), tileTy.getSgMap(), tileTy.getWgMap(),
        tileTy.getOrder(), innerBlocks, tileTy.getWgData());

    auto newTileTy =
        imex::xetile::TileType::get(tileTy.getShape(), elemTy, attr);

    rewriter.replaceOpWithNewOp<xetile::InitTileOp>(
        op, newTileTy, op.getSource(), op.getOffsets(),
        op.getStaticOffsetsAttr(), op.getDynamicShape(),
        op.getDynamicStrides());

    return mlir::success();
  }
};

// It updates load_tile to reveal effects of innerblock attribute by
// representing value as 4D vector. An unpack op is added at the end
// to make this change to be transparent to its users.
struct LoadTileOpPattern : public XeTileConversion<xetile::LoadTileOp> {
  using XeTileConversion<xetile::LoadTileOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tileTy = op.getSource().getType();
    auto shape = tileTy.getShape();
    auto innerBlocks = tileTy.getInnerBlocks();

    if (!innerBlocks)
      return rewriter.notifyMatchFailure(
          op, "The InnerBlock attr is required by missing from the tile.\n");

    auto vecTy = ::mlir::VectorType::get({shape[0] / innerBlocks[0],
                                          shape[1] / innerBlocks[1],
                                          innerBlocks[0], innerBlocks[1]},
                                         tileTy.getElementType());

    auto newOp = rewriter.create<imex::xetile::LoadTileOp>(
        loc, vecTy, adaptor.getSource(), op.getPaddingAttr());

    auto unpack = addUnpackOp(newOp, rewriter);

    rewriter.replaceOp(op, unpack);

    return mlir::success();
  }
};

// It updates store_tile to reveal effects of innerblock attribute.
// It uses pack op to align the shape of its vector value to the tile shape.
struct StoreTileOpPattern : public XeTileConversion<xetile::StoreTileOp> {
  using XeTileConversion<xetile::StoreTileOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = llvm::dyn_cast<xetile::TileType>(adaptor.getTile().getType());
    auto innerBlocks = tileTy.getInnerBlocks().asArrayRef();
    auto value = addPackOp(adaptor.getValue(), innerBlocks, rewriter);
    rewriter.replaceOpWithNewOp<xetile::StoreTileOp>(op, value,
                                                     adaptor.getTile());
    return mlir::success();
  }
};

// TODO: M, K, N. replace it with uArch
llvm::SmallVector<int> getMMASize(mlir::Type elemTy) {
  assert(elemTy.isIntOrFloat());
  auto bits = elemTy.getIntOrFloatBitWidth();
  llvm::SmallVector<int> result;
  switch (bits) {
  case 16:
    result = llvm::SmallVector<int>({8, 16, 16});
    break;
  default:
    result = llvm::SmallVector<int>({8, 8, 8});
    break;
  }
  return result;
}

// It updates tile_mma to reveal effects of innerblock attribute.
// Values will be reprented as 4D vectors. An unpack op is applied
// to its result to make the change transparent to its users.
struct TileMMAOpPattern : public XeTileConversion<xetile::TileMMAOp> {
  using XeTileConversion<xetile::TileMMAOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto a = adaptor.getA();
    auto b = adaptor.getB();
    auto c = adaptor.getC();

    assert(a && b && "a operand or b operand is (are) missing.\n");

    auto mmaSize = getMMASize(op.getElementType());

    a = addPackOp(a, {mmaSize[0], mmaSize[1]}, rewriter);
    b = addPackOp(b, {mmaSize[1], mmaSize[2]}, rewriter);

    if (c) {
      auto ty = c.getType().dyn_cast<mlir::VectorType>();
      if (ty.getRank() == 2)
        c = addPackOp(c, {mmaSize[0], mmaSize[2]}, rewriter);
      if (ty.getRank() == 4)
        c = addUnpackAndPackOps(c, {mmaSize[0], mmaSize[2]}, rewriter);
    }

    auto resultTy = op.getResult().getType();
    auto shape = resultTy.getShape();
    auto vecTy = ::mlir::VectorType::get(
        {shape[0] / mmaSize[0], shape[1] / mmaSize[2], mmaSize[0], mmaSize[2]},
        resultTy.getElementType());

    mlir::Value newOp =
        rewriter.create<imex::xetile::TileMMAOp>(loc, vecTy, a, b, c);
    newOp = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// It updates update_tile_offset to reveal effects of innerblock attribute
// by updating the type of it result.
struct UpdateTileOffsetOpPattern
    : public XeTileConversion<xetile::UpdateTileOffsetOp> {
  using XeTileConversion<xetile::UpdateTileOffsetOp>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xetile::UpdateTileOffsetOp>(
        op, adaptor.getTile().getType(), adaptor.getTile(),
        adaptor.getOffsetX(), adaptor.getOffsetY());
    return mlir::success();
  }
};

void populateXeTileBlockingPatterns(imex::XeTypeConverter &converter,
                                    mlir::RewritePatternSet &patterns) {

  patterns.insert<ArithConstantOpPattern, SCFForOpPattern, SCFYieldOpPattern,
                  InitTileOpPattern, LoadTileOpPattern, StoreTileOpPattern,
                  TileMMAOpPattern, UpdateTileOffsetOpPattern>(
      patterns.getContext(), converter);
}

// Lowers XeTile to blocked layout with high-dim vector
class XeTileBlockingPass
    : public imex::impl::XeTileBlockingBase<imex::XeTileBlockingPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    auto mod = this->getOperation();

    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    auto &usageAnalysis = getAnalysis<TileUsageAnalysis>();

    mlir::RewritePatternSet patterns(&context);
    XeTypeConverter typeConverter(context, &usageAnalysis);

    populateXeTileBlockingPatterns(typeConverter, patterns);

    // Use TopDown traversal order, and only look at existing ops
    // to simpliy the code logic and speedup the pass
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
std::unique_ptr<::mlir::Pass> createXeTileBlockingPass() {
  return std::make_unique<XeTileBlockingPass>();
}
} // namespace imex
