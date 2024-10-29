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

#include <imex/Dialect/Dist/IR/DistOps.h>
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
} // namespace imex

namespace imex {
namespace {

// Base-class for RewriterPatterns which handle recursion
// Rewriters will not rewrite (stop recursion)
// if input NDArray operands have no device or are within a EnvRegion.
template <typename T>
struct RecOpRewritePattern : public ::mlir::OpRewritePattern<T> {
  using ::mlir::OpRewritePattern<T>::OpRewritePattern;
  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    RecOpRewritePattern<T>::setHasBoundedRewriteRecursion();
  }
};

/// If given NDArray operation operates on or returns a NDArray with an
/// GPUEnvironment, create GPU region operation which yields the operation.
static ::mlir::LogicalResult
matchAndRewritePTOP(::mlir::Operation *op, ::mlir::PatternRewriter &rewriter,
                    bool checkOprnds) {
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
  auto rOp = rewriter.create<::imex::region::EnvironmentRegionOp>(
      op->getLoc(), env, std::nullopt, op->getResultTypes(),
      [op](::mlir::OpBuilder &builder, ::mlir::Location loc) {
        auto cOp = builder.clone(*op);
        (void)builder.create<::imex::region::EnvironmentRegionYieldOp>(
            loc, cOp->getResults());
      });
  rewriter.replaceOp(op, rOp);

  return ::mlir::success();
}

// Shallow wrapper template to handle all NDArrayOps
// The matchAndWrite method simply calls matchAndRewritePTOP
template <typename PTOP, bool CHECK_OPERANDS = true>
struct NDArrayOpRWP : public RecOpRewritePattern<PTOP> {
  using RecOpRewritePattern<PTOP>::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(PTOP op, ::mlir::PatternRewriter &rewriter) const override {
    return matchAndRewritePTOP(op, rewriter, CHECK_OPERANDS);
  }
};

struct AddGPURegionsPass
    : public ::imex::impl::AddGPURegionsBase<AddGPURegionsPass> {

  AddGPURegionsPass() = default;

  void runOnOperation() override {
    ::mlir::FrozenRewritePatternSet patterns;
    // It would be nicer to have a single rewrite-pattern which covers all
    // NDArrayOps
    insertPatterns<NDArrayOpRWP<::imex::ndarray::ToTensorOp>,
                   NDArrayOpRWP<::imex::ndarray::FromMemRefOp>,
                   NDArrayOpRWP<::imex::ndarray::DeleteOp>,
                   NDArrayOpRWP<::imex::ndarray::DimOp>,
                   NDArrayOpRWP<::imex::ndarray::SubviewOp>,
                   NDArrayOpRWP<::imex::ndarray::ExtractSliceOp>,
                   NDArrayOpRWP<::imex::ndarray::InsertSliceOp>,
                   NDArrayOpRWP<::imex::ndarray::ImmutableInsertSliceOp>,
                   NDArrayOpRWP<::imex::ndarray::LoadOp>,
                   NDArrayOpRWP<::imex::ndarray::CopyOp, false>,
                   NDArrayOpRWP<::imex::ndarray::CastOp>,
                   NDArrayOpRWP<::imex::ndarray::CastElemTypeOp>,
                   NDArrayOpRWP<::imex::ndarray::LinSpaceOp>,
                   NDArrayOpRWP<::imex::ndarray::CreateOp>,
                   NDArrayOpRWP<::imex::ndarray::ReshapeOp>,
                   NDArrayOpRWP<::imex::ndarray::EWBinOp>,
                   NDArrayOpRWP<::imex::ndarray::EWUnyOp>,
                   NDArrayOpRWP<::imex::ndarray::ReductionOp>,
                   NDArrayOpRWP<::imex::ndarray::PermuteDimsOp>,
                   NDArrayOpRWP<::imex::dist::InitDistArrayOp>,
                   NDArrayOpRWP<::imex::dist::LocalOffsetsOfOp>,
                   NDArrayOpRWP<::imex::dist::PartsOfOp>,
                   NDArrayOpRWP<::imex::dist::DefaultPartitionOp>,
                   NDArrayOpRWP<::imex::dist::LocalTargetOfSliceOp>,
                   NDArrayOpRWP<::imex::dist::LocalBoundingBoxOp>,
                   NDArrayOpRWP<::imex::dist::LocalCoreOp>,
                   NDArrayOpRWP<::imex::dist::RePartitionOp>,
                   NDArrayOpRWP<::imex::dist::SubviewOp>,
                   NDArrayOpRWP<::imex::dist::EWBinOp>,
                   NDArrayOpRWP<::imex::dist::EWUnyOp>>(getContext(), patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createAddGPURegionsPass() {
  return std::make_unique<::imex::AddGPURegionsPass>();
}

} // namespace imex
