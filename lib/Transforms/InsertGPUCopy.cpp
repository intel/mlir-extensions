//===- InsertGPUCopy.cpp - InsertGPUCopy Pass  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file replaces the memref.copy ops with gpu.memcpy ops if the
/// memref.copy resides in an environment region. This environment region must
/// be created in a prior pass where the device/host memory semantics are
/// present.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/Threading.h"
#include <imex/Transforms/Passes.h>

#include <imex/Dialect/Region/RegionUtils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <optional>

namespace imex {
#define GEN_PASS_DEF_INSERTGPUCOPY
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {

inline bool isInEnvRegion(::mlir::Operation *op) {
  if (!op)
    return false;
  if (!op->getParentOfType<::imex::region::EnvironmentRegionOp>())
    return false;
  return true;
}

class InsertGPUCopyPass final
    : public imex::impl::InsertGPUCopyBase<InsertGPUCopyPass> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    auto &funcBody = func.getBody();
    if (funcBody.empty()) {
      return;
    } else if (!llvm::hasSingleElement(funcBody)) {
      func.emitError("Function must have exactly one block");
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(func);
    // collect copy ops in GPU regions
    ::mlir::SmallVector<::mlir::memref::CopyOp> copyOpsInGpuRegion;

    // traverse ops and identify memref.copy ops which are in GPU region
    (void)func.walk([&](::mlir::memref::CopyOp op) {
      if (isInEnvRegion(op)) {
        copyOpsInGpuRegion.emplace_back(op);
      }
    });

    // Replace copy ops with gpu.memcpy
    for (auto copyOp : copyOpsInGpuRegion) {
      builder.setInsertionPoint(copyOp);
      // /*asyncToken*/ std::nullopt,
      ::mlir::gpu::MemcpyOp::create(builder,
          copyOp.getLoc(), /*resultTypes*/ ::mlir::TypeRange{},
          /*asyncDependencies*/ ::mlir::ValueRange{}, copyOp.getTarget(),
          copyOp.getSource());
      copyOp.erase();
    }
  }
};

} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createInsertGPUCopyPass() {
  return std::make_unique<InsertGPUCopyPass>();
}
} // namespace imex
