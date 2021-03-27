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
