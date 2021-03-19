#include "plier/rewrites/memory_rewrites.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <mlir/Dialect/StandardOps/IR/Ops.h>

namespace
{
bool isWrite(mlir::Operation& op)
{
    if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
    {
        return effects.hasEffect<mlir::MemoryEffects::Write>();
    }
    return false;
}

bool isRead(mlir::Operation& op)
{
    if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
    {
        return effects.hasEffect<mlir::MemoryEffects::Read>();
    }
    return false;
}

struct Result
{
    bool changed;
    bool hasWrites;
    bool hasReads;
};

Result promoteLoads(llvm::MutableArrayRef<mlir::Region> regions, mlir::PatternRewriter& rewriter)
{
    bool changed = false;
    bool hasWrites = false;
    bool hasReads = false;
    bool storeDead = false;
    for (auto& region : regions)
    {
        for (auto& block : region.getBlocks())
        {
            mlir::StoreOp currentStore;
            for (auto& op : llvm::make_early_inc_range(block))
            {
                if (!op.getRegions().empty())
                {
                    auto res = promoteLoads(op.getRegions(), rewriter);
                    if (res.changed)
                    {
                        changed = true;
                    }
                    if (res.hasWrites)
                    {
                        currentStore = {};
                    }
                    if (res.hasReads)
                    {
                        storeDead = false;
                    }
                    continue;
                }

                if (auto load = mlir::dyn_cast<mlir::LoadOp>(op))
                {
                    hasReads = true;
                    if (currentStore)
                    {
                        if (load.memref() == currentStore.memref() &&
                            load.indices() == currentStore.indices())
                        {
                            rewriter.replaceOp(&op, currentStore.value());
                            changed = true;
                        }
                        else
                        {
                            storeDead = false;
                        }
                    }
                }
                else if (auto store = mlir::dyn_cast<mlir::StoreOp>(op))
                {
                    if (currentStore && storeDead &&
                        currentStore.memref() == store.memref() &&
                        currentStore.indices() == store.indices())
                    {
                        rewriter.eraseOp(currentStore);
                    }
                    hasWrites = true;
                    currentStore = store;
                    storeDead = true;
                }
                else if (isWrite(op))
                {
                    hasWrites = true;
                    currentStore = {};
                }
                else if (isRead(op))
                {
                    hasReads = true;
                    storeDead = false;
                }
                else if(op.hasTrait<mlir::OpTrait::HasRecursiveSideEffects>())
                {
                    currentStore = {};
                    hasWrites = true;
                    hasReads = true;
                    storeDead = false;
                }
            }
        }
    }
    return Result{changed, hasWrites, hasReads};
}

bool checkIsSingleElementsMemref(mlir::ShapedType type)
{
    if (!type.hasRank())
    {
        return false;
    }
    return llvm::all_of(type.getShape(), [](auto val) { return val == 1; });
}
}

mlir::LogicalResult plier::PromoteLoads::matchAndRewrite(mlir::FuncOp op, mlir::PatternRewriter& rewriter) const
{
    auto res = promoteLoads(op->getRegions(), rewriter);
    return mlir::success(res.changed);
}

mlir::LogicalResult plier::SingeWriteMemref::matchAndRewrite(mlir::StoreOp op, mlir::PatternRewriter& rewriter) const
{
    auto memref = op.memref();
    if (!checkIsSingleElementsMemref(memref.getType().cast<mlir::ShapedType>()))
    {
        return mlir::failure();
    }
    auto parent = memref.getDefiningOp();
    if (!mlir::isa_and_nonnull<mlir::AllocOp, mlir::AllocaOp>(parent))
    {
        return mlir::failure();
    }

    mlir::StoreOp valueStore;
    llvm::SmallVector<mlir::Operation*> loads;
    for (auto user : memref.getUsers())
    {
        if (auto store = mlir::dyn_cast<mlir::StoreOp>(user))
        {
            if (valueStore)
            {
                // More than one store
                return mlir::failure();
            }
            valueStore = store;
        }
        else if (auto load = mlir::dyn_cast<mlir::LoadOp>(user))
        {
            loads.emplace_back(load);
        }
        else if (mlir::isa<mlir::DeallocOp>(user))
        {
            // nothing
        }
        else
        {
            // Unsupported op
            return mlir::failure();
        }
    }

    auto parentBlock = parent->getBlock();
    if (!valueStore || valueStore->getBlock() != parentBlock)
    {
        return mlir::failure();
    }

    auto val = valueStore.value();
    for (auto load : loads)
    {
        rewriter.replaceOp(load, val);
    }
    for (auto user : llvm::make_early_inc_range(parent->getUsers()))
    {
        rewriter.eraseOp(user);
    }
    rewriter.eraseOp(parent);
    return mlir::success();
}
