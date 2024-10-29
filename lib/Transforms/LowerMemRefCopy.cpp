//===- LowerMemRefCopy.cpp - lower memref.copy pass --------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass lowers memref copyOp to linalg generic operations and enables
/// simple memref copyOp canonicalization
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

namespace imex {
#define GEN_PASS_DEF_LOWERMEMREFCOPY
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {
struct LowerMemRefCopy
    : public imex::impl::LowerMemRefCopyBase<LowerMemRefCopy> {
  void runOnOperation() override {
    auto &domInfo = getAnalysis<DominanceInfo>();
    auto func = getOperation();
    // walk through memref.copy ops in the funcion body
    (void)func.walk<WalkOrder::PreOrder>([&](memref::CopyOp op) -> WalkResult {
      if (op->getParentOp() != func)
        return WalkResult::skip();
      auto src = op.getSource();
      auto dst = op.getTarget();
      // supposed to work on same memref type
      auto srcType = mlir::cast<MemRefType>(src.getType());
      auto dstType = mlir::cast<MemRefType>(dst.getType());
      if (srcType != dstType)
        return WalkResult::skip();
      // supposed to work on memref.alloc
      auto srcOp = src.getDefiningOp<memref::AllocOp>();
      auto dstOp = dst.getDefiningOp<memref::AllocOp>();
      if (!srcOp || !dstOp)
        return WalkResult::skip();
      // check use of src after this copyOp, being conservative
      // FIXME: handle dealloc of src and dst
      bool hasSubsequentUse = false;
      for (auto user : src.getUsers()) {
        if (isa<memref::DeallocOp>(user)) {
          continue;
        }
        if (domInfo.properlyDominates(op, user)) {
          hasSubsequentUse = true;
          break;
        }
      }

      // replace copy with linalg.generic
      if (hasSubsequentUse) {
        OpBuilder builder(op);
        linalg::makeMemRefCopyOp(builder, op.getLoc(), src, dst);
      } else {
        // coalesce buffer
        dst.replaceAllUsesWith(src);
      }
      op.erase();
      return WalkResult::advance();
    });
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createLowerMemRefCopyPass() {
  return std::make_unique<LowerMemRefCopy>();
}
} // namespace imex
