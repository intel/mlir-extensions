#include "plier/rewrites/arg_lowering.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Transforms/DialectConversion.h>

#include "plier/dialect.hpp"

plier::ArgOpLowering::ArgOpLowering(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context):
    OpRewritePattern(context), converter(typeConverter) {}

mlir::LogicalResult plier::ArgOpLowering::matchAndRewrite(plier::ArgOp op, mlir::PatternRewriter& rewriter) const
{
    auto func = op->getParentOfType<mlir::FuncOp>();
    if (!func)
    {
        return mlir::failure();
    }

    auto index= op.index();
    if (index >= func.getNumArguments())
    {
        return mlir::failure();
    }

    auto arg = func.getArgument(index);
    if(converter.convertType(op.getType()) != arg.getType())
    {
        return mlir::failure();
    }
    rewriter.replaceOp(op, arg);
    return mlir::success();
}

