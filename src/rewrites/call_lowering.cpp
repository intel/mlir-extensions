#include "plier/rewrites/call_lowering.hpp"

plier::CallOpLowering::CallOpLowering(
    mlir::TypeConverter&, mlir::MLIRContext* context,
    CallOpLowering::resolver_t resolver):
    OpRewritePattern(context), resolver(resolver) {}

mlir::LogicalResult plier::CallOpLowering::matchAndRewrite(plier::PyCallOp op, mlir::PatternRewriter& rewriter) const
{
    auto operands = op.getOperands();
    if (operands.empty())
    {
        return mlir::failure();
    }
    auto func_type = operands[0].getType();
    if (!func_type.isa<plier::PyType>())
    {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Type, 8> arg_types;
    llvm::SmallVector<mlir::Value, 8> args;
    auto getattr = mlir::dyn_cast_or_null<plier::GetattrOp>(operands[0].getDefiningOp());
    if (!getattr)
    {
        llvm::copy(llvm::drop_begin(op.getOperandTypes(), 1), std::back_inserter(arg_types));
        llvm::copy(llvm::drop_begin(op.getOperands(), 1), std::back_inserter(args));
        // TODO kwargs
    }
    else
    {
        arg_types.push_back(getattr.getOperand().getType());
        args.push_back(getattr.getOperand());
        llvm::copy(llvm::drop_begin(op.getOperandTypes(), 1), std::back_inserter(arg_types));
        llvm::copy(llvm::drop_begin(op.getOperands(), 1), std::back_inserter(args));
        // TODO kwargs
    }

    return resolver(op, op.func_name(), args, rewriter);
}
