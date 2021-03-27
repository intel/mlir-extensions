#include "plier/rewrites/if_rewrites.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/SCF/SCF.h>

mlir::LogicalResult plier::IfOpConstCond::matchAndRewrite(mlir::scf::IfOp op, mlir::PatternRewriter& rewriter) const
{
    auto cond = mlir::dyn_cast_or_null<mlir::CmpIOp>(op.condition().getDefiningOp());
    if (!cond)
    {
        return mlir::failure();
    }
    auto is_const = [](mlir::Value val)
    {
        if (auto parent = val.getDefiningOp())
        {
            return parent->hasTrait<mlir::OpTrait::ConstantLike>();
        }
        return false;
    };

    auto replace = [&](mlir::Block& block, mlir::Value to_replace, mlir::Value new_val)
    {
        for (auto& use : llvm::make_early_inc_range(to_replace.getUses()))
        {
            auto owner = use.getOwner();
            if (block.findAncestorOpInBlock(*owner))
            {
                rewriter.updateRootInPlace(owner, [&]()
                {
                    use.set(new_val);
                });
            }
        }
    };

    mlir::Value const_val;
    mlir::Value to_replace;
    if (is_const(cond.lhs()))
    {
        const_val = cond.lhs();
        to_replace = cond.rhs();
    }
    else if (is_const(cond.rhs()))
    {
        const_val = cond.rhs();
        to_replace = cond.lhs();
    }
    else
    {
        return mlir::failure();
    }

    if (cond.predicate() == mlir::CmpIPredicate::eq)
    {
        replace(op.thenRegion().front(), to_replace, const_val);
    }
    else if (cond.predicate() == mlir::CmpIPredicate::ne)
    {
        replace(op.elseRegion().front(), to_replace, const_val);
    }
    else
    {
        return mlir::failure();
    }

    return mlir::success();
}
