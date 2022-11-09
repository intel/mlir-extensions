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

#include "PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace imex;

namespace {
struct AddOuterParallelLoopPass
    : public AddOuterParallelLoopBase<AddOuterParallelLoopPass> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    llvm::SmallVector<scf::ForOp, 4> ops;
    // populate the top level for-loop
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp op) {
      if (op->getParentOp() == func) {
        ops.push_back(op);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    mlir::OpBuilder builder(func.getContext());
    // move the for-loop inside the newly created parallel-loop
    for (scf::ForOp op : ops) {
      builder.setInsertionPoint(op);
      mlir::Value cst0 = builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getIndexAttr(0));
      mlir::Value cst1 = builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getIndexAttr(1));
      scf::ParallelOp outer = builder.create<scf::ParallelOp>(
          op.getLoc(), SmallVector<Value, 2>{cst0}, SmallVector<Value, 2>{cst1},
          SmallVector<Value, 2>{cst1});
      op->moveBefore(&outer.getBody()->front());
    }
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createAddOuterParallelLoopPass() {
  return std::make_unique<AddOuterParallelLoopPass>();
}
} // namespace imex
