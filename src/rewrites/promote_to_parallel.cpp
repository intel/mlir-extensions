#include "plier/rewrites/promote_to_parallel.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>

#include "plier/dialect.hpp"

namespace
{
bool hasSideEffects(mlir::Operation *op)
{
    return op->walk([&](mlir::Operation *op)
    {
        if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
        {
            if(effects.hasEffect<mlir::MemoryEffects::Write>())
            {
                return mlir::WalkResult::interrupt();
            }
        }
//        if (op->hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
//        {
//            return mlir::WalkResult::interrupt();
//        }
        if (mlir::isa<mlir::CallOpInterface>(op))
        {
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    }).wasInterrupted();
}
}

mlir::LogicalResult plier::PromoteToParallel::matchAndRewrite(mlir::scf::ForOp op, mlir::PatternRewriter& rewriter) const
{
    auto has_parallel_attr = op->hasAttr(plier::attributes::getParallelName());
    if (!has_parallel_attr && hasSideEffects(op))
    {
        return mlir::failure();
    }

    auto& old_body = op.getLoopBody().front();
    auto old_yield = mlir::cast<mlir::scf::YieldOp>(old_body.getTerminator());
    auto reduce_args = old_body.getArguments().drop_front();
    llvm::SmallVector<llvm::SmallVector<mlir::Operation*, 1>, 8> reduce_bodies(reduce_args.size());
    llvm::DenseSet<mlir::Operation*> reduce_ops;
    for (auto it : llvm::enumerate(reduce_args))
    {
        auto reduce_arg = it.value();
        auto reduce_index = it.index();
        if (!reduce_arg.hasOneUse())
        {
            return mlir::failure();
        }
        auto reduce_op = *reduce_arg.user_begin();
        if (reduce_op->getNumOperands() != 2)
        {
            return mlir::failure();
        }
        auto& reduce_body = reduce_bodies[reduce_index];
        while (true)
        {
            if (!reduce_op->hasOneUse())
            {
                return mlir::failure();
            }
            reduce_body.push_back(reduce_op);
            reduce_ops.insert(reduce_op);
            auto next_op = *reduce_op->user_begin();
            if (next_op == old_yield)
            {
                auto yield_operand = old_yield.getOperand(static_cast<unsigned>(reduce_index));
                if (yield_operand != reduce_op->getResult(0))
                {
                    return mlir::failure();
                }
                break;
            }
            for (auto operand : next_op->getOperands())
            {
                if (operand.getDefiningOp() != reduce_op &&
                    operand.getParentBlock() == &old_body)
                {
                    return mlir::failure();
                }
            }
            reduce_op = next_op;
        }
    }

    auto body_builder = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange iter_vals, mlir::ValueRange temp)
    {
        assert(1 == iter_vals.size());
        assert(temp.empty());
        mlir::BlockAndValueMapping mapping;
        mapping.map(old_body.getArguments().front(), iter_vals.front());
        for (auto& old_op : old_body.without_terminator())
        {
            if (0 == reduce_ops.count(&old_op))
            {
                builder.clone(old_op, mapping);
            }
        }
        mlir::BlockAndValueMapping reduce_mapping;
        for (auto it : llvm::enumerate(reduce_bodies))
        {
            auto& reduce_body = it.value();
            assert(!reduce_body.empty());
            reduce_mapping = mapping;
            auto first_op = reduce_body.front();
            assert(first_op->getNumOperands() == 2);
            auto reduce_body_builder = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value val0, mlir::Value val1)
            {
                reduce_mapping.map(first_op->getOperand(0), val0);
                reduce_mapping.map(first_op->getOperand(1), val1);
                mlir::Operation* last_op = nullptr;
                for (auto reduce_op : reduce_body)
                {
                    last_op = builder.clone(*reduce_op, reduce_mapping);
                    assert(1 == last_op->getNumResults());
                }
                builder.create<mlir::scf::ReduceReturnOp>(loc, last_op->getResult(0));
            };
            auto reduce_arg = reduce_args[it.index()];
            auto first_op_operands = first_op->getOperands();
            auto reduce_operand = (first_op_operands[0] == reduce_arg ? first_op_operands[1] : first_op_operands[0]);
            assert(reduce_operand != reduce_arg);
            reduce_operand = mapping.lookupOrDefault(reduce_operand);
            assert(reduce_operand);
            builder.create<mlir::scf::ReduceOp>(loc, reduce_operand, reduce_body_builder);
        }
    };

    auto parallel_op = rewriter.create<mlir::scf::ParallelOp>(op.getLoc(), op.lowerBound(), op.upperBound(), op.step(), op.initArgs(), body_builder);
    if (has_parallel_attr)
    {
        parallel_op->setAttr(plier::attributes::getParallelName(), rewriter.getUnitAttr());
    }
    rewriter.replaceOp(op, parallel_op.getResults());

    return mlir::success();
}
