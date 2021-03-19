#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class FuncOp;
class StoreOp;
}

namespace plier
{
struct PromoteLoads : public mlir::OpRewritePattern<mlir::FuncOp>
{
    using mlir::OpRewritePattern<mlir::FuncOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::FuncOp op, mlir::PatternRewriter &rewriter) const override;
};

struct SingeWriteMemref : public mlir::OpRewritePattern<mlir::StoreOp>
{
    using mlir::OpRewritePattern<mlir::StoreOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::StoreOp op, mlir::PatternRewriter &rewriter) const override;
};
}
