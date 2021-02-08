#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class TypeConverter;
}

namespace plier
{
struct FuncOpSignatureConversion : public mlir::OpRewritePattern<mlir::FuncOp>
{
    FuncOpSignatureConversion(mlir::TypeConverter& conv,
                              mlir::MLIRContext* ctx);

    /// Hook for derived classes to implement combined matching and rewriting.
    mlir::LogicalResult
    matchAndRewrite(mlir::FuncOp funcOp, mlir::PatternRewriter &rewriter) const override;

private:
    mlir::TypeConverter& converter;
};
}
