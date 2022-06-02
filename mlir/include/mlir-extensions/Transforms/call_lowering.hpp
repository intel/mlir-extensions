// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "mlir-extensions/Dialect/plier/dialect.hpp"

#include <mlir/IR/PatternMatch.h>

namespace plier {
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
} // namespace plier
