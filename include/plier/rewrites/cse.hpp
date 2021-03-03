#pragma once

#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/PatternMatch.h>

namespace plier
{
namespace detail
{
mlir::LogicalResult applyCSE(mlir::Region& region, mlir::PatternRewriter& rewriter);
}

template<typename Op>
struct CSERewrite : public mlir::OpRewritePattern<Op>
{
    CSERewrite(mlir::MLIRContext *context):
        mlir::OpRewritePattern<Op>(context, /*benefit*/1) {} // TODO: benefit=0

    mlir::LogicalResult matchAndRewrite(
        Op op, mlir::PatternRewriter &rewriter) const override
    {
        return ::plier::detail::applyCSE(op.getRegion(), rewriter);
    }
};
}
