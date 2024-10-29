//===- AddOuterParallelLoop.cpp - add outer parallel loop pass--*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// When the original func does not have an outer parallel loop, this pass adds
/// one so that the immediately followed pass gpu-map-parallel-loops can work.
///
//===----------------------------------------------------------------------===//

#include "imex/Dialect/Region/IR/RegionOps.h"
#include "imex/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SetVector.h"

namespace imex {
#define GEN_PASS_DEF_ADDOUTERPARALLELLOOP
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {
struct AddOuterParallelLoopPass
    : public imex::impl::AddOuterParallelLoopBase<AddOuterParallelLoopPass> {
private:
  void runOnBlock(::mlir::Block &block, ::mlir::Operation *parent,
                  mlir::OpBuilder &builder) {
    llvm::SmallVector<llvm::SmallVector<Operation *, 4>, 4> groupedOps;
    // populate the top level for-loop
    for (auto topIt = block.begin(); topIt != block.end();) {
      auto regOp = dyn_cast<::imex::region::EnvironmentRegionOp>(*topIt);
      if (regOp) {
        runOnBlock(regOp.getRegion().front(), regOp, builder);
        ++topIt;
        continue;
      }

      scf::ForOp forOp = dyn_cast<scf::ForOp>(*topIt++);
      if (!forOp) {
        continue;
      }
      // populate forOp w/o iter_args
      if (forOp.getInitArgs().size() == 0) {
        groupedOps.push_back({forOp});
        continue;
      }
      // populate forOp with iter_args
      llvm::SetVector<Value> topUsers{forOp.getResults().begin(),
                                      forOp.getResults().end()};
      Operation *endOp = forOp;
      bool hasReturnOp = false;
      for (auto it = topUsers.begin(); it != topUsers.end();) {
        if (hasReturnOp) {
          break;
        }
        for (auto *user : it->getUsers()) {
          if (isa<func::ReturnOp>(user)) {
            hasReturnOp = true;
            break;
          }
          while (user->getParentOp() != parent) {
            user = user->getParentOp();
          }
          topUsers.insert(user->getResults().begin(), user->getResults().end());
          if (endOp->isBeforeInBlock(user)) {
            endOp = user;
          }
        }
        it = std::next(it);
      }
      if (!hasReturnOp) {
        Block::iterator endIt = std::next(endOp->getIterator());
        topIt = endIt;
        llvm::SmallVector<Operation *, 4> ops;
        for (auto it = forOp->getIterator(); it != endIt; it = std::next(it)) {
          ops.push_back(&*it);
        }
        if (!ops.empty())
          groupedOps.push_back(ops);
      }
    }
    // move the for-loop and its users into the newly created parallel-loop
    for (const auto &ops : groupedOps) {
      auto op = ops.front();
      builder.setInsertionPoint(op);
      auto loc = op->getLoc();
      mlir::Value cst0 =
          builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
      mlir::Value cst1 =
          builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));
      scf::ParallelOp outer =
          builder.create<scf::ParallelOp>(loc, cst0, cst1, cst1);
      auto yieldOp = outer.getBody()->getTerminator();
      for (auto op : ops) {
        op->moveBefore(yieldOp);
      }
    }
  }

public:
  void runOnOperation() override {
    auto func = getOperation();
    if (func.getBody().empty())
      return;
    llvm::SmallVector<llvm::SmallVector<Operation *, 4>, 4> groupedOps;
    mlir::OpBuilder builder(func.getContext());
    // populate the top level for-loop
    runOnBlock(func.getBody().front(), func, builder);
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createAddOuterParallelLoopPass() {
  return std::make_unique<AddOuterParallelLoopPass>();
}
} // namespace imex
