//===- AddGPURegions.cpp - NDArrayToDist Transform  ------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements adding GPU regions around NDArray operations
///       where appropriate.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Transforms/Passes.h>
#include <imex/Dialect/Region/IR/RegionOps.h>
#include <imex/Dialect/Region/RegionUtils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

namespace imex {
#define GEN_PASS_DEF_ADDGPUREGIONS
#include <imex/Dialect/NDArray/Transforms/Passes.h.inc>

namespace {

/// If given operation operates on or returns a tensor with an
/// GPUEnvironment, create GPU region operation which yields the operation.
static ::mlir::LogicalResult matchAndRewritePTOP(::mlir::Operation *op,
                                                 ::mlir::IRRewriter &rewriter,
                                                 bool checkOprnds = true) {
  auto parent = op->getParentOp();
  if (!parent) {
    return ::mlir::failure();
  } else {
    auto pregion =
        ::mlir::dyn_cast<::imex::region::EnvironmentRegionOp>(parent);
    if (pregion && isGpuRegion(pregion)) {
      return ::mlir::failure();
    }
  }

  // check if any of the operands/returns NDArrays with GPUEnvironments
  auto getenv = [](::mlir::TypeRange types) -> ::imex::region::GPUEnvAttr {
    for (auto rt : types) {
      if (auto env = ::imex::ndarray::getGPUEnv(rt)) {
        return env;
      }
    }
    return {};
  };

  auto env = getenv(op->getResultTypes());
  if (checkOprnds && !env)
    env = getenv(op->getOperandTypes());

  // if the op is not related to GPU -> nothing to do
  if (!env) {
    return ::mlir::failure();
  }

  // create a region with given env and clone creator op within and yield it
  rewriter.replaceOpWithNewOp<::imex::region::EnvironmentRegionOp>(
      op, env, llvm::ArrayRef<mlir::Value>(), op->getResultTypes(),
      [op](::mlir::OpBuilder &builder, ::mlir::Location loc) {
        auto cOp = builder.clone(*op);
        (void)::imex::region::EnvironmentRegionYieldOp::create(
            builder, loc, cOp->getResults());
      });

  return ::mlir::success();
}

struct AddGPURegionsPass
    : public ::imex::impl::AddGPURegionsBase<AddGPURegionsPass> {

  AddGPURegionsPass() = default;

  void runOnOperation() override {
    ::mlir::IRRewriter rewriter(&getContext());
    getOperation()->walk([&](::mlir::Operation *op) {
      auto doCopy = !::mlir::isa<::imex::ndarray::CopyOp>(op);
      rewriter.setInsertionPointAfter(op);
      (void)matchAndRewritePTOP(op, rewriter, doCopy);
    });
  }
};
} // namespace

std::unique_ptr<::mlir::Pass> createAddGPURegionsPass() {
  return std::make_unique<::imex::AddGPURegionsPass>();
}
} // namespace imex
