#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class SelectOp;
namespace scf
{
class IfOp;
}
}

namespace plier
{
struct IfOpConstCond : public mlir::OpRewritePattern<mlir::scf::IfOp>
{
    IfOpConstCond(mlir::MLIRContext *context):
        mlir::OpRewritePattern<mlir::scf::IfOp>(context, /*benefit*/1) {}

    mlir::LogicalResult matchAndRewrite(
        mlir::scf::IfOp op, mlir::PatternRewriter &rewriter) const override;
};
}
