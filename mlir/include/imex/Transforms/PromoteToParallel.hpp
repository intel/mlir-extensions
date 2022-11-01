// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir {
namespace scf {
class ForOp;
class ParallelOp;
} // namespace scf
} // namespace mlir

namespace imex {
struct PromoteToParallel : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

struct MergeNestedForIntoParallel
    : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using mlir::OpRewritePattern<mlir::scf::ParallelOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace imex
