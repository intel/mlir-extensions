// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace imex {
mlir::LogicalResult applyCSE(mlir::Region &region,
                             mlir::PatternRewriter &rewriter, bool recursive);
mlir::LogicalResult applyCSE(mlir::Region &region, bool recursive);

template <typename Op, bool Recursive>
struct CSERewrite : public mlir::OpRewritePattern<Op> {
  CSERewrite(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<Op>(context, /*benefit*/ 1) {} // TODO: benefit=0

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    return ::imex::applyCSE(op.getRegion(), rewriter, Recursive);
  }
};
} // namespace imex
