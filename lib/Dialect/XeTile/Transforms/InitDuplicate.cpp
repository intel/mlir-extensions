//===- InitDuplicate.cpp ------ xetile-init-duplicate Pass ------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains lowering transformation for duplicating init_tiles
/// if they are used in conflicting scenarios from lowering perspective,
/// including tiles created for both load and store, or load and prefetch.
/// Since when lowering to XeGPU, they may have different sizes at XeGPU
/// level for performance.
///
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

#include "imex/Dialect/XeTile/Transforms/Passes.h"

using namespace mlir;
using namespace imex;

namespace imex {
#define GEN_PASS_DEF_XETILEINITDUPLICATE
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {

class XeTileInitDuplicatePass
    : public impl::XeTileInitDuplicateBase<XeTileInitDuplicatePass> {
public:
  void runOnOperation() override {
    auto &usageAnalysis = getAnalysis<TileUsageAnalysis>();
    mlir::Operation *op = getOperation();
    op->walk([&](imex::xetile::InitTileOp op) {
      mlir::OpBuilder rewriter(op);
      if (usageAnalysis.isForLoadAndStore(op) ||
          usageAnalysis.isForLoadAndPrefetch(op)) {
        mlir::Operation *cloneOp = rewriter.clone(*op);
        for (auto user : op->getUsers()) {
          if (llvm::isa<xetile::StoreTileOp>(user) ||
              llvm::dyn_cast<xetile::PrefetchTileOp>(user)) {
            auto *targetOp = llvm::dyn_cast_if_present<mlir::Operation *>(user);
            targetOp->replaceUsesOfWith(op->getResults()[0],
                                        cloneOp->getResults()[0]);
          }
        }
      }
    });
  }
};

/// Create a pass
std::unique_ptr<::mlir::Pass> createXeTileInitDuplicatePass() {
  return std::make_unique<XeTileInitDuplicatePass>();
}
} // namespace imex
