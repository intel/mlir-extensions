//===- RegionParallelLoopToGpu.cpp -  --------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file extends upstream ParallelLoopToGpuPass by applying the transform
/// only if the parallel loop is within a GPU region
/// (`region.env_region #region.gpu_env<...>`).
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/RegionParallelLoopToGpu/RegionParallelLoopToGpu.h>
#include <imex/Dialect/Region/RegionUtils.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPU.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace imex {
#define GEN_PASS_DEF_CONVERTREGIONPARALLELLOOPTOGPU
#include "imex/Conversion/Passes.h.inc"

namespace {
struct ConvertRegionParallelLoopToGpuPass
    : public ::imex::impl::ConvertRegionParallelLoopToGpuBase<
          ConvertRegionParallelLoopToGpuPass> {
  ConvertRegionParallelLoopToGpuPass() = default;

  void runOnOperation() override {
    ::mlir::RewritePatternSet patterns(&getContext());
    ::mlir::populateParallelLoopToGPUPatterns(patterns);
    ::mlir::ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(
        [](::mlir::Operation *) { return true; });
    ::mlir::configureParallelLoopToGPULegality(target);

    // collect all gpu regions
    ::mlir::SmallVector<::mlir::Operation *> ops;
    getOperation()->walk([&](::imex::region::EnvironmentRegionOp op,
                             const ::mlir::WalkStage &stage) {
      if (::imex::region::isGpuRegion(op)) {
        ops.push_back(op);
        return ::mlir::WalkResult::skip();
      }
      return ::mlir::WalkResult::advance();
    });

    // apply par-loop to gpu conversion to collected gpu regions
    if (::mlir::failed(
            ::mlir::applyPartialConversion(ops, target, std::move(patterns)))) {
      signalPassFailure();
    }
    ::mlir::finalizeParallelLoopToGPUConversion(getOperation());
  }
};
} // namespace

/// Create a pass to convert the Region dialect to the GPU dialect.
std::unique_ptr<::mlir::Pass> createConvertRegionParallelLoopToGpuPass() {
  return std::make_unique<ConvertRegionParallelLoopToGpuPass>();
}

} // namespace imex
