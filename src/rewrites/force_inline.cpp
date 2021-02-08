#include "plier/rewrites/force_inline.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/InliningUtils.h>

#include "plier/dialect.hpp"

mlir::LogicalResult plier::ForceInline::matchAndRewrite(mlir::CallOp op, mlir::PatternRewriter& rewriter) const
{
    auto attr_name = plier::attributes::getForceInlineName();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);
    auto func = mod.lookupSymbol<mlir::FuncOp>(op.callee());
    if (!func)
    {
        return mlir::failure();
    }
    if (!op->hasAttr(attr_name) &&
        !func->hasAttr(attr_name))
    {
        return mlir::failure();
    }

    if (!llvm::hasNItems(func.getRegion(), 1))
    {
        return mlir::failure();
    }
    mlir::InlinerInterface inliner_interface(op->getContext());
    auto parent = op->getParentOp();
    rewriter.startRootUpdate(parent);
    auto res = mlir::inlineCall(inliner_interface, op, func, &func.getRegion());
    if (mlir::succeeded(res))
    {
        assert(op->getUsers().empty());
        rewriter.eraseOp(op);
        rewriter.finalizeRootUpdate(parent);
    }
    else
    {
        rewriter.cancelRootUpdate(parent);
    }
    return res;
}
