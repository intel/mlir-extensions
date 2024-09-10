//===------- BlockAligning.cpp ------ mma-opt Pass -------*- C++ ----*----===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains lowering transformation for optmizing mma performance
/// by align the tile size of load based on the mma size for the dpas. The
/// blocking pass could choose [32, 32] as block size for, e.g,
/// %1 = xetile.init_tile ..., and [8, 16] as block size for A operand of
/// e.g., %2 = xetile.tile_mma ... . Therefore, when %1 is loaded as A operand
/// of %2, it will cause a lot of extract ops to extract 8x16 blocks from the
/// 32x32 block, which is not efficiently supported by the hardware. But the
/// hardware can efficiently extract 8x16 blocks from a 32x16 block.
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <optional>

#include "imex/Dialect/XeTile/Transforms/Blocking.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/DebugUtils.h"

using namespace mlir;
using namespace imex;
namespace imex {
#define GEN_PASS_DEF_XETILEBLOCKALIGNING
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {
namespace BlockAligning {

// reblocking the dense constant op if necessary to expected tiled shape
struct ArithConstantOpPattern
    : public XeTileConversion<mlir::arith::ConstantOp, PropagateAnalysis> {
  using XeTileConversion<mlir::arith::ConstantOp,
                         PropagateAnalysis>::XeTileConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto blockSize = getValue(op);
    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());

    // it is not the target operator.
    if (!blockSize || !value || value.getType().getRank() != 4)
      return failure();

    auto shape = value.getType().getShape();
    auto oldBlockSize = shape.take_back(2);

    // the blk size has been in the expected shape
    if (oldBlockSize == blockSize.asArrayRef())
      return failure();

    auto newTy = mlir::VectorType::get({shape[0] * shape[2] / blockSize[0],
                                        shape[1] * shape[3] / blockSize[1],
                                        blockSize[0], blockSize[1]},
                                       value.getElementType());

    // TODO: can reshape reveal the pack logic?
    value = value.reshape(newTy);
    auto newOp = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), value);
    auto packOp = addUnpackAndPackOps(newOp, oldBlockSize, rewriter);
    rewriter.replaceOp(op, packOp);
    return mlir::success();
  }
};

// Updating the the body of SCF::ForOp when its init args are changed.
struct SCFForOpPattern
    : public XeTileConversion<mlir::scf::ForOp, PropagateAnalysis> {
  using XeTileConversion<mlir::scf::ForOp, PropagateAnalysis>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    // we need to update the SCF ForOp if the types of its init arg values
    // do not match the types of the region iter args, or the init arg value
    // is defined by a TilePackOp. Otherwise we can skip the op.
    bool changed = false;
    llvm::SmallVector<mlir::Value> newInitArgs;
    llvm::SmallVector<mlir::DenseI64ArrayAttr> oldBlockSizes;
    llvm::SmallVector<mlir::DenseI64ArrayAttr> newBlockSizes;
    for (auto [i, arg] : llvm::enumerate(adaptor.getInitArgs())) {
      auto blockArg = op.getRegionIterArg(i);
      auto defOp = arg.getDefiningOp<xetile::TilePackOp>();
      auto oldSize = defOp ? defOp.getInnerBlocksAttr() : DenseI64ArrayAttr();
      auto newSize = defOp ? getValue(blockArg) : DenseI64ArrayAttr();
      auto newArg =
          defOp && newSize ? addUnpackAndPackOps(arg, newSize, rewriter) : arg;
      oldBlockSizes.push_back(oldSize);
      newBlockSizes.push_back(newSize);
      newInitArgs.push_back(newArg);
      changed |= (newArg.getType() != blockArg.getType());
    }

    if (!changed)
      return mlir::failure();

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), newInitArgs);

    mlir::Block *block = op.getBody();
    mlir::Block *newBlock = newOp.getBody();
    llvm::SmallVector<mlir::Value> newArguments;
    auto numCtrlOprs = newOp.getNumInductionVars();
    // remove the terminator of the new block
    if (newBlock->mightHaveTerminator())
      rewriter.eraseOp(newBlock->getTerminator());

    // add UnpackOp and PackOp pairs to the block arguments
    // if the corresponding init arg is repacked, such that
    // the old unpack op using it in the body will be folded
    for (auto [i, arg] : llvm::enumerate(newBlock->getArguments())) {
      if (i < numCtrlOprs || !oldBlockSizes[i - numCtrlOprs]) {
        newArguments.push_back(arg);
      } else {
        auto repackOp =
            addUnpackAndPackOps(arg, oldBlockSizes[i - numCtrlOprs], rewriter);
        newArguments.push_back(repackOp);
      }
    }
    rewriter.mergeBlocks(block, newBlock, newArguments);

    // update the yieldOp by adding UnpackOp and PackOp pairs
    // if corresponding init arg is repacked.
    auto yieldOp =
        llvm::dyn_cast<scf::YieldOp>(newOp.getBody()->getTerminator());
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.startOpModification(yieldOp);
    for (auto [i, v] : llvm::enumerate(yieldOp.getResults())) {
      if (newBlockSizes[i]) {
        rewriter.setInsertionPointAfter(v.getDefiningOp());
        auto repack = addUnpackAndPackOps(v, newBlockSizes[i], rewriter);
        yieldOp->setOperand(i, repack);
      }
    }
    rewriter.finalizeOpModification(yieldOp);

    // Update the results
    rewriter.setInsertionPointAfter(op);
    llvm::SmallVector<mlir::Value> newValues;
    for (auto [i, result] : llvm::enumerate(newOp->getResults())) {
      if (oldBlockSizes[i]) {
        auto unpack = addUnpackAndPackOps(result, oldBlockSizes[i], rewriter);
        newValues.push_back(unpack);
      } else {
        newValues.push_back(result);
      }
    }
    rewriter.replaceOp(op, newValues);

    return mlir::success();
  }
};

// Update the innerblock size of the result tile. We only update the second
// dimention of the innerblock size to use large load size but small compute
// size
struct InitTileOpPattern
    : public XeTileConversion<xetile::InitTileOp, PropagateAnalysis> {
  using XeTileConversion<xetile::InitTileOp,
                         PropagateAnalysis>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = op.getType();
    auto oldBlockSize = tileTy.getInnerBlocks();
    auto blockSize = getValue(op);

    // it is not the target to be updated
    if (!blockSize || oldBlockSize[1] == blockSize[1])
      return mlir::failure();

    auto newBlockSize = mlir::DenseI64ArrayAttr::get(
        getContext(), {oldBlockSize[0], blockSize[1]});

    auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), tileTy.getSgMap(), tileTy.getWgMap(),
        tileTy.getOrder(), newBlockSize, tileTy.getMemoryScope());

    auto newTileTy = imex::xetile::TileType::get(tileTy.getShape(),
                                                 tileTy.getElementType(), attr);

    rewriter.startOpModification(op);
    op.getTile().setType(newTileTy);
    rewriter.finalizeOpModification(op);
    return mlir::success();
  }
};

// Update the result shape of the load if its source is updated
struct LoadTileOpPattern
    : public XeTileConversion<xetile::LoadTileOp, PropagateAnalysis> {
  using XeTileConversion<xetile::LoadTileOp,
                         PropagateAnalysis>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = op.getSource().getType();
    auto resTy = op.getValue().getType();
    auto blockSize = tileTy.getInnerBlocks();

    // it is not the target since the source is not updated
    if (resTy.getShape().take_back(2) == blockSize.asArrayRef())
      return mlir::failure();

    auto shape = tileTy.getShape();
    auto vecTy = ::mlir::VectorType::get({shape[0] / blockSize[0],
                                          shape[1] / blockSize[1], blockSize[0],
                                          blockSize[1]},
                                         tileTy.getElementType());

    auto newOp = rewriter.create<imex::xetile::LoadTileOp>(
        op.getLoc(), vecTy, adaptor.getSource(), op.getPaddingAttr());

    auto repackOp =
        addUnpackAndPackOps(newOp, resTy.getShape().take_back(2), rewriter);
    rewriter.replaceOp(op, repackOp);
    return mlir::success();
  }
};

struct UpdateTileOffsetOpPattern
    : public XeTileConversion<xetile::UpdateTileOffsetOp, PropagateAnalysis> {
  using XeTileConversion<xetile::UpdateTileOffsetOp,
                         PropagateAnalysis>::XeTileConversion;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    if (adaptor.getTile().getType() == op.getResult().getType())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<xetile::UpdateTileOffsetOp>(
        op, adaptor.getTile().getType(), adaptor.getTile(),
        adaptor.getOffsetX(), adaptor.getOffsetY());
    return mlir::success();
  }
};

} // namespace BlockAligning

void populateXeTileBlockAligningPatterns(imex::XeTypeConverter &converter,
                                         mlir::RewritePatternSet &patterns,
                                         PropagateAnalysis &analysis) {
  patterns.insert<
      BlockAligning::ArithConstantOpPattern, BlockAligning::SCFForOpPattern,
      BlockAligning::InitTileOpPattern, BlockAligning::LoadTileOpPattern,
      BlockAligning::UpdateTileOffsetOpPattern>(patterns.getContext(),
                                                converter, analysis);
}

// TODO: [block-aligning] remove the following code when upstreaming the pass.
// The pass is not supposed to be exposed to users. Temporary keep in case we
// need debug it for down stream development.
class XeTileBlockAligningPass
    : public impl::XeTileBlockAligningBase<XeTileBlockAligningPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    auto mod = this->getOperation();

    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    auto &analysis = getAnalysis<PropagateAnalysis>();
    mlir::RewritePatternSet patterns(&context);
    XeTypeConverter typeConverter(context);
    populateXeTileBlockAligningPatterns(typeConverter, patterns, analysis);

    // Use TopDown traversal order, and only look at existing ops
    // to simpliy the code logic and speedup the pass
    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    config.useTopDownTraversal = true;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(
            applyPatternsAndFoldGreedily(mod, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

/// Create a pass
std::unique_ptr<::mlir::Pass> createXeTileBlockAligningPass() {
  return std::make_unique<XeTileBlockAligningPass>();
}
} // namespace imex
