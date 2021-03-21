#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class AllocOp;
class AllocaOp;
class FuncOp;
struct LogicalResult;
}

namespace plier
{

struct RemoveTrivialAlloc : public mlir::OpRewritePattern<mlir::AllocOp>
{
    using mlir::OpRewritePattern<mlir::AllocOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::AllocOp op, mlir::PatternRewriter &rewriter) const override;
};

struct RemoveTrivialAlloca : public mlir::OpRewritePattern<mlir::AllocaOp>
{
    using mlir::OpRewritePattern<mlir::AllocaOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::AllocaOp op, mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult optimizeMemoryOps(mlir::FuncOp func);
}
