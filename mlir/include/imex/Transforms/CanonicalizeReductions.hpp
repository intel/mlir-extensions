// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <mlir/IR/PatternMatch.h>

namespace mlir {
class Pass;
namespace scf {
class ForOp;
}
} // namespace mlir

namespace imex {
struct CanonicalizeReduction : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Tries to promote loads/stores in scf.for to loop-carried variables.
std::unique_ptr<mlir::Pass> createCanonicalizeReductionsPass();
} // namespace imex
