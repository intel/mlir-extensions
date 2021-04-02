#pragma once

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class TypeConverter;
}

namespace plier
{
class ArgOp;

struct ArgOpLowering : public mlir::OpRewritePattern<plier::ArgOp>
{
    ArgOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context);

    mlir::LogicalResult matchAndRewrite(
        plier::ArgOp op, mlir::PatternRewriter &rewriter) const override;
private:
    mlir::TypeConverter& converter;
};
}
