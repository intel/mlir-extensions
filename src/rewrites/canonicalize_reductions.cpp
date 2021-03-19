#include "plier/rewrites/canonicalize_reductions.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>

#include "plier/transforms/block_utils.hpp"

namespace
{
bool checkMemrefType(mlir::Value value)
{
    if (auto type = value.getType().dyn_cast<mlir::MemRefType>())
    {
//        auto shape = type.getShape();
//        return shape.empty() || (1 == shape.size() && 1 == shape[0]);
        return true;
    }
    return false;
}

bool isOutsideBlock(mlir::ValueRange values, mlir::Block& block)
{
    auto blockArgs = block.getArguments();
    for (auto val : values)
    {
        if (llvm::find(blockArgs, val) != blockArgs.end())
        {
            return false;
        }
        auto op = val.getDefiningOp();
        if (op && block.findAncestorOpInBlock(*op))
        {
            return false;
        }
    }
    return true;
}

bool checkForPotentialAliases(mlir::Value value, mlir::Operation* parent)
{
    assert(parent->getRegions().size() == 1);
    assert(llvm::hasNItems(parent->getRegions().front(), 1));
    if (auto effects = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(value.getDefiningOp()))
    {
        if (!effects.onlyHasEffect<mlir::MemoryEffects::Allocate>())
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    mlir::LoadOp load;
    mlir::StoreOp store;
    auto& parentBlock = parent->getRegions().front().front();
    for (auto user : value.getUsers())
    {
        if (mlir::isa<mlir::ViewLikeOpInterface>(user))
        {
            // TODO: very conservative
            return false;
        }
        auto relation = plier::relativeTo(user, parent);
        if (plier::OpRelation::Unknown == relation)
        {
            return false;
        }
        if (plier::OpRelation::In == relation)
        {
            if (auto effects = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(user))
            {
                if (user->getBlock() != &parentBlock)
                {
                    return false;
                }
                if (effects.hasEffect<mlir::MemoryEffects::Read>())
                {
                    if (load || !mlir::isa<mlir::LoadOp>(user))
                    {
                        return false;
                    }
                    load = mlir::cast<mlir::LoadOp>(user);
                }
                if (effects.hasEffect<mlir::MemoryEffects::Write>())
                {
                    if (store || !mlir::isa<mlir::StoreOp>(user))
                    {
                        return false;
                    }
                    store = mlir::cast<mlir::StoreOp>(user);
                }
            }
        }
    }
    if (!load || !store || !load->isBeforeInBlock(store) ||
        load.indices() != store.indices() || !isOutsideBlock(load.indices(), parentBlock))
    {
        return false;
    }
    return true;
}

bool checkSupportedOps(mlir::Value value, mlir::Operation* parent)
{
    for (auto user : value.getUsers())
    {
        if (user->getParentOp() == parent && !mlir::isa<mlir::LoadOp, mlir::StoreOp>(user))
        {
            return false;
        }
    }
    return true;
}

bool checkMemref(mlir::Value value, mlir::Operation* parent)
{
    return checkMemrefType(value) &&
           checkForPotentialAliases(value, parent) &&
           checkSupportedOps(value, parent);
}

mlir::Value createScalarLoad(mlir::PatternRewriter &builder, mlir::Location loc, mlir::Value memref, mlir::ValueRange indices)
{
    return builder.create<mlir::LoadOp>(loc, memref, indices);
}

void createScalarStore(
    mlir::PatternRewriter &builder, mlir::Location loc, mlir::Value val,
    mlir::Value memref, mlir::ValueRange indices)
{
    builder.create<mlir::StoreOp>(loc, val, memref, indices);
}
}

mlir::LogicalResult plier::CanonicalizeReduction::matchAndRewrite(mlir::scf::ForOp op, mlir::PatternRewriter& rewriter) const
{
    llvm::SmallVector<std::pair<mlir::Value, mlir::ValueRange>> to_process;
    for (auto& current : op.getLoopBody().front())
    {
        if (auto load = mlir::dyn_cast<mlir::LoadOp>(current))
        {
            auto memref = load.memref();
            if (checkMemref(memref, op))
            {
                to_process.push_back({memref, load.indices()});
            }
        }
    }

    if (!to_process.empty())
    {
        auto loc = op.getLoc();
        auto init_args = llvm::to_vector<8>(op.initArgs());
        for (auto it : to_process)
        {
            init_args.emplace_back(createScalarLoad(rewriter, loc, it.first, it.second));
        }
        auto prev_args_offset = op.initArgs().size();
        auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter, mlir::ValueRange iter_vals)
        {
            auto& old_body = op.getLoopBody().front();
            mlir::BlockAndValueMapping mapping;
            mapping.map(old_body.getArguments().front(), iter);
            mapping.map(old_body.getArguments().drop_front(), iter_vals);
            auto yield_args = llvm::to_vector<8>(iter_vals);
            for (auto& body_op : old_body.without_terminator())
            {
                auto invalid_index = static_cast<unsigned>(-1);
                auto get_iter_index = [&](auto op)->unsigned
                {
                    auto arg = op.memref();
                    for (auto it : llvm::enumerate(llvm::make_first_range(to_process)))
                    {
                        if (arg == it.value())
                        {
                            return static_cast<unsigned>(it.index() + prev_args_offset);
                        }
                    }
                    return invalid_index;
                };
                if (auto load = mlir::dyn_cast<mlir::LoadOp>(body_op))
                {
                    auto index = get_iter_index(load);
                    if (index != invalid_index)
                    {
                        mapping.map(body_op.getResults().front(), yield_args[index]);
                    }
                    else
                    {
                        builder.clone(body_op, mapping);
                    }
                }
                else if (auto store = mlir::dyn_cast<mlir::StoreOp>(body_op))
                {
                    auto index = get_iter_index(store);
                    if (index != invalid_index)
                    {
                        yield_args[index] = mapping.lookup(store.value());
                    }
                    else
                    {
                        builder.clone(body_op, mapping);
                    }
                }
                else
                {
                    builder.clone(body_op, mapping);
                }
            }
            auto yield = mlir::cast<mlir::scf::YieldOp>(old_body.getTerminator());
            llvm::copy(yield.results(), yield_args.begin());
            builder.create<mlir::scf::YieldOp>(loc, yield_args);
        };
        auto results = rewriter.create<mlir::scf::ForOp>(loc, op.lowerBound(), op.upperBound(), op.step(), init_args, body).results();
        for (auto it : llvm::enumerate(to_process))
        {
            auto index = prev_args_offset + it.index();
            auto result = results[static_cast<unsigned>(index)];
            createScalarStore(rewriter, loc, result, it.value().first, it.value().second);
        }
        rewriter.replaceOp(op, results.take_front(prev_args_offset));
        return mlir::success();
    }

    return mlir::failure();
}
