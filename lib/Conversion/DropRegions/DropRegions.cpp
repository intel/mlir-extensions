//===- DropRegions.cpp - DropRegions conversion  -------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DropRegions conversion, removing all Region ops.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/DropRegions/DropRegions.h>
#include <imex/Dialect/Region/IR/RegionOps.h>
#include <imex/Dialect/Region/RegionUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace imex {
#define GEN_PASS_DEF_DROPREGIONS
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

namespace imex {

namespace {
// *******************************
// ***** Individual patterns *****
// *******************************

struct RemoveRegion
    : public ::mlir::OpRewritePattern<::imex::region::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::region::EnvironmentRegionOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    ::imex::region::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct DropRegionsPass : public ::imex::impl::DropRegionsBase<DropRegionsPass> {
  DropRegionsPass() = default;

  void runOnOperation() override {
    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<RemoveRegion>(getContext(), patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);
  }
};

} // namespace

/// Populate the given list with patterns that drops regions
void populateDropRegionsConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass that drops regions
std::unique_ptr<::mlir::Pass> createDropRegionsPass() {
  return std::make_unique<DropRegionsPass>();
}

} // namespace imex
