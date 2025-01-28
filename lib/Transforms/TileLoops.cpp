//===- TileLoops.cpp ------------------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the TileLoops transform which tiles loops for GPU
/// mapping.
///
//===----------------------------------------------------------------------===//

#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/Interfaces/TilingInterface.h>
#include <mlir/Pass/Pass.h>

#include "llvm/Support/Threading.h"
#include <imex/Dialect/Region/RegionUtils.h>
#include <imex/Transforms/Passes.h>

namespace imex {
#define GEN_PASS_DEF_TILELOOPS
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

#define DEBUG_TYPE "tile-loops"

#ifndef NDEBUG
#define DEBUG_MSG(PREFIX, MSG)                                                 \
  LLVM_DEBUG(llvm::dbgs() << PREFIX << ": " << MSG << "\n");
#define DEBUG_OP(PREFIX, MSG, OP)                                              \
  LLVM_DEBUG(llvm::dbgs() << PREFIX << ": " << MSG << " '" << OP->getName()    \
                          << "' " << OP->getLoc() << "\n");
#define DEBUG_OP_VEC(PREFIX, MSG, OPVEC)                                       \
  LLVM_DEBUG(llvm::dbgs() << PREFIX << ": " << MSG << " (" << OPVEC.size()     \
                          << ")\n");                                           \
  for (auto op : OPVEC) {                                                      \
    DEBUG_OP(PREFIX, "  ", op)                                                 \
  }
#endif

using namespace imex;

namespace {

static ::mlir::FailureOr<::mlir::SmallVector<int64_t>>
getDefaultTileSizes(::mlir::linalg::LinalgOp linalgOp,
                    ::mlir::ArrayRef<int64_t> userProvidedTiles) {
  // The user-provided tiles are considered from the outer
  // most loop. If not enough tiles are provided we pad with
  // zeros.
  if (!userProvidedTiles.empty()) {
    size_t numParallelLoops = linalgOp.getNumParallelLoops();
    size_t nonZeros = 0;
    for (auto tile : userProvidedTiles)
      if (tile != 0)
        nonZeros++;
    if (nonZeros > numParallelLoops ||
        userProvidedTiles.size() > linalgOp.getNumLoops()) {
      return ::mlir::failure();
    }

    ::mlir::SmallVector<int64_t> userTiles(linalgOp.getNumLoops(), 0);
    for (auto tile : ::llvm::enumerate(userProvidedTiles))
      userTiles[tile.index()] = tile.value();
    return userTiles;
  }
  return ::mlir::failure();
}

struct TileLoops final : public imex::impl::TileLoopsBase<TileLoops> {

  using TileLoopsBase::TileLoopsBase;

  void runOnOperation() override {

    ::mlir::func::FuncOp func = getOperation();
    ::mlir::IRRewriter rewriter(&getContext());
    transform(rewriter, func, this->tileSizes, this->minTileFactor);

    return;
  }

private:
  void transform(::mlir::RewriterBase &rewriter, ::mlir::func::FuncOp func,
                 ::mlir::ArrayRef<int64_t> tileSizes, int64_t minTileFactor) {
    DEBUG_MSG("tile-loops", "Entering transform");
    ::mlir::SmallVector<::mlir::Operation *> allLinalgOps;
    func->walk([&](::mlir::linalg::LinalgOp linalgOp) {
      if (!inRegions || ::imex::region::isInGpuRegion(linalgOp)) {
        allLinalgOps.push_back(linalgOp);
      }
    });
    DEBUG_OP_VEC("tile-loops", "  Found linalg ops", allLinalgOps);

    for (auto op : allLinalgOps) {
      DEBUG_OP("tile-loops", "  Tiling op:", op);
      auto tiles = getDefaultTileSizes(
          ::llvm::cast<::mlir::linalg::LinalgOp>(op), tileSizes);
      if (failed(tiles)) {
        DEBUG_MSG("tile-loops",
                  "  Failed to compute default tile sizes. Aborting.");
        return;
      }
      DEBUG_MSG("tile-loops", "  tile sizes:");
      LLVM_DEBUG(llvm::dbgs() << "tile-loops:    (");
      LLVM_DEBUG(llvm::interleaveComma(*tiles, llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << ")\n");

      auto tilesRes =
          ::mlir::getAsOpFoldResult(rewriter.getI64ArrayAttr(*tiles));
      ::mlir::scf::SCFTilingOptions options;
      options.setTileSizes(tilesRes);
      options.setLoopType(::mlir::scf::SCFTilingOptions::LoopType::ForallOp);
      auto tileOp = ::mlir::cast<::mlir::TilingInterface>(op);
      ::mlir::FailureOr<::mlir::scf::SCFTilingResult> tilingResult =
          mlir::scf::tileUsingSCF(rewriter, tileOp, options);
      if (failed(tilingResult)) {
        DEBUG_MSG("tile-loops", "  Failed to tile op. Aborting.");
        return;
      }
      DEBUG_MSG("tile-loops", "  Tiling applied successfully.");
      rewriter.replaceOp(op, tilingResult->mergeResult.replacements);
    }
  }
};

} // end anonymous namespace

namespace imex {
std::unique_ptr<mlir::Pass> createTileLoopsPass() {
  return std::make_unique<TileLoops>();
}
} // namespace imex
