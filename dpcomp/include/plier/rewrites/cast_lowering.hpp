#pragma once

#include <functional>

#include "plier/dialect.hpp"

#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class TypeConverter;
}

namespace plier
{
struct CastOpLowering : public mlir::OpRewritePattern<plier::CastOp>
{
    using cast_t = std::function<mlir::Value(mlir::Type, mlir::Value, mlir::PatternRewriter&)>;

    CastOpLowering(mlir::TypeConverter &typeConverter,
                   mlir::MLIRContext *context,
                   cast_t cast_func = nullptr);

    mlir::LogicalResult matchAndRewrite(
        plier::CastOp op, mlir::PatternRewriter &rewriter) const override;

private:
    mlir::TypeConverter& converter;
    cast_t cast_func;
};
}
