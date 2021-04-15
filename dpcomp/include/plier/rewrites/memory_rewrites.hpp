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

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
namespace memref
{
class AllocOp;
class AllocaOp;
}
class FuncOp;
struct LogicalResult;
}

namespace plier
{

struct RemoveTrivialAlloc : public mlir::OpRewritePattern<mlir::memref::AllocOp>
{
    using mlir::OpRewritePattern<mlir::memref::AllocOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::memref::AllocOp op, mlir::PatternRewriter &rewriter) const override;
};

struct RemoveTrivialAlloca : public mlir::OpRewritePattern<mlir::memref::AllocaOp>
{
    using mlir::OpRewritePattern<mlir::memref::AllocaOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::memref::AllocaOp op, mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult optimizeMemoryOps(mlir::FuncOp func);
}
