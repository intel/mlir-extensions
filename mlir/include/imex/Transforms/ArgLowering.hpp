// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/PatternMatch.h>

namespace plier {
class ArgOp;
}

namespace imex {
struct ArgOpLowering : public mlir::OpRewritePattern<plier::ArgOp> {
  ArgOpLowering(mlir::MLIRContext *context);

  mlir::LogicalResult
  matchAndRewrite(plier::ArgOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace imex
