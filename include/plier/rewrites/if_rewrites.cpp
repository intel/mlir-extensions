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

mlir::LogicalResult plier::SimplifyEmptyIf::matchAndRewrite(mlir::scf::IfOp op, mlir::PatternRewriter& rewriter) const
{
    if (op.getNumResults() == 0 || op.elseRegion().empty())
    {
        return mlir::failure();
    }
    if (!llvm::hasNItems(op.thenRegion().front(), 1) ||
        !llvm::hasNItems(op.elseRegion().front(), 1))
    {
        return mlir::failure();
    }
    auto then_yield_args = mlir::cast<mlir::scf::YieldOp>(op.thenRegion().front().getTerminator()).getOperands();
    auto else_yield_args = mlir::cast<mlir::scf::YieldOp>(op.elseRegion().front().getTerminator()).getOperands();
    for (auto it : llvm::zip(then_yield_args, else_yield_args))
    {
        if (std::get<0>(it) != std::get<1>(it))
        {
            return mlir::failure();
        }
    }
    llvm::SmallVector<mlir::Value> args(then_yield_args.begin(), then_yield_args.end());
    assert(args.size() == op.getNumResults());
    rewriter.replaceOp(op, args);
    return mlir::success();
}

mlir::LogicalResult plier::SimplifySelect::matchAndRewrite(mlir::SelectOp op, mlir::PatternRewriter& rewriter) const
{
    auto true_val = op.getTrueValue();
    auto false_val = op.getFalseValue();
    if (true_val == false_val)
    {
        rewriter.replaceOp(op, true_val);
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult plier::SimplifySelectEq::matchAndRewrite(mlir::SelectOp op, mlir::PatternRewriter& rewriter) const
{
    auto cond = mlir::dyn_cast_or_null<mlir::CmpIOp>(op.condition().getDefiningOp());
    if (!cond)
    {
        return mlir::failure();
    }
    if (cond.predicate() != mlir::CmpIPredicate::eq &&
        cond.predicate() != mlir::CmpIPredicate::ne)
    {
        return mlir::failure();
    }

    auto cond_lhs = cond.lhs();
    auto cond_rhs = cond.rhs();

    auto true_val = op.getTrueValue();
    auto false_val = op.getFalseValue();

    if (cond.predicate() == mlir::CmpIPredicate::ne)
    {
        std::swap(true_val, false_val);
    }

    if ((cond_lhs == true_val && cond_rhs == false_val) ||
        (cond_rhs == true_val && cond_lhs == false_val))
    {
        rewriter.replaceOp(op, false_val);
        return mlir::success();
    }

    return mlir::failure();
}
