#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class CallOp;
}

namespace plier
{
struct ForceInline : public mlir::OpRewritePattern<mlir::CallOp>
{
    using mlir::OpRewritePattern<mlir::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::CallOp op, mlir::PatternRewriter &rewriter) const override;
};
}
