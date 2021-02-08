#include "plier/rewrites/cast_lowering.hpp"

#include <mlir/Transforms/DialectConversion.h>

plier::CastOpLowering::CastOpLowering(
    mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
    CastOpLowering::cast_t cast_func):
    OpRewritePattern(context), converter(typeConverter),
    cast_func(std::move(cast_func)) {}

mlir::LogicalResult plier::CastOpLowering::matchAndRewrite(
    plier::CastOp op, mlir::PatternRewriter& rewriter) const
{
    auto src = op.getOperand();
    auto src_type = src.getType();
    auto dst_type = converter.convertType(op.getType());
    if (dst_type)
    {
        if (src_type == dst_type)
        {
            rewriter.replaceOp(op, src);
            return mlir::success();
        }
        if (nullptr != cast_func)
        {
            if (auto new_op = cast_func(dst_type, src, rewriter))
            {
                rewriter.replaceOp(op, new_op);
                return mlir::success();
            }
        }
    }
    return mlir::failure();
}
