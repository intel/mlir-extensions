// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "imex/Dialect/plier/Dialect.hpp"

#include <mlir/IR/PatternMatch.h>

namespace imex {
struct ExpandCallVarargs : public mlir::OpRewritePattern<plier::PyCallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

struct CallOpLowering : public mlir::OpRewritePattern<plier::PyCallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override;

protected:
  using KWargs = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location loc,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const = 0;
};
} // namespace imex
