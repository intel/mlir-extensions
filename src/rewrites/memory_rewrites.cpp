#include "plier/rewrites/memory_rewrites.hpp"

#include <mlir/Analysis/BufferAliasAnalysis.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>

#include <plier/analysis/memory_ssa.hpp>

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

Result promoteLoadsImpl(mlir::Region& region, mlir::PatternRewriter& rewriter)
{
    bool changed = false;
    bool hasWrites = false;
    bool hasReads = false;
    bool storeDead = false;
    for (auto& block : region.getBlocks())
    {
        mlir::StoreOp currentStore;
        for (auto& op : llvm::make_early_inc_range(block))
        {
            if (!op.getRegions().empty())
            {
                for (auto& nestedRegion : op.getRegions())
                {
                    auto res = promoteLoadsImpl(nestedRegion, rewriter);
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

mlir::LogicalResult plier::promoteLoads(mlir::Region& region, mlir::PatternRewriter& rewriter)
{
    return mlir::success(promoteLoadsImpl(region, rewriter).changed);
}

mlir::LogicalResult plier::promoteLoads(mlir::Region& region)
{
    class MyPatternRewriter : public mlir::PatternRewriter
    {
    public:
        MyPatternRewriter(mlir::MLIRContext *ctx) : PatternRewriter(ctx) {}
    };

    MyPatternRewriter dummyRewriter(region.getContext());
    return mlir::success(promoteLoadsImpl(region, dummyRewriter).changed);
}

mlir::LogicalResult plier::PromoteLoads::matchAndRewrite(mlir::FuncOp op, mlir::PatternRewriter& rewriter) const
{
    return promoteLoads(op.getRegion(), rewriter);
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

namespace
{
struct Meminfo
{
    mlir::Value memref;
    mlir::ValueRange indices;
};

llvm::Optional<Meminfo> getMeminfo(mlir::Operation* op)
{
    assert(nullptr != op);
    if (auto load = mlir::dyn_cast<mlir::LoadOp>(op))
    {
        return Meminfo{load.memref(), load.indices()};
    }
    if (auto store = mlir::dyn_cast<mlir::StoreOp>(op))
    {
        return Meminfo{store.memref(), store.indices()};
    }
    return {};
}

mlir::LogicalResult optimizeUses(mlir::BufferAliasAnalysis& aliases, plier::MemorySSA& memSSA)
{
    auto mayAlias = [&](mlir::Operation* op1, mlir::Operation* op2)
    {
        auto meminfo1 = getMeminfo(op1);
        if (!meminfo1)
        {
            return true;
        }
        auto meminfo2 = getMeminfo(op2);
        if (!meminfo2)
        {
            return true;
        }
        return aliases.resolve(meminfo1->memref).count(meminfo2->memref) != 0;
    };
    llvm::errs() << "optimizeMemoryOps2\n";
    memSSA.print(llvm::errs());
    (void)memSSA.optimizeUses(mayAlias);
    llvm::errs() << "optimizeMemoryOps3\n";
    memSSA.print(llvm::errs());
    llvm::errs() << "optimizeMemoryOps4\n";
    return mlir::failure();
}

mlir::LogicalResult foldLoads(mlir::BufferAliasAnalysis& /*aliases*/, plier::MemorySSA& memSSA)
{
    auto mustAlias = [&](mlir::Operation* op1, mlir::Operation* op2)
    {
        auto meminfo1 = getMeminfo(op1);
        if (!meminfo1)
        {
            return false;
        }
        auto meminfo2 = getMeminfo(op2);
        if (!meminfo2)
        {
            return false;
        }
        return meminfo1->memref == meminfo2->memref &&
               meminfo1->indices == meminfo2->indices;
    };
    bool changed = false;
    for (auto& node : llvm::make_early_inc_range(memSSA.getNodes()))
    {
        if (plier::MemorySSA::NodeType::Use == memSSA.getNodeType(&node))
        {
            auto def = memSSA.getNodeDef(&node);
            assert(nullptr != def);
            if (plier::MemorySSA::NodeType::Def != memSSA.getNodeType(def))
            {
                continue;
            }
            auto op1 = memSSA.getNodeOperation(&node);
            auto op2 = memSSA.getNodeOperation(def);
            assert(nullptr != op1);
            assert(nullptr != op2);
            if (mustAlias(op1, op2))
            {
                mlir::ValueRange val = mlir::cast<mlir::StoreOp>(op2).value();
                op1->replaceAllUsesWith(val);
                memSSA.eraseNode(&node);
                changed = true;
            }
        }
    }
    return mlir::success(changed);
}

}

mlir::LogicalResult plier::optimizeMemoryOps(mlir::FuncOp func)
{
    mlir::BufferAliasAnalysis aliases(func);

    llvm::errs() << "optimizeMemoryOps1\n";
    auto memSSA = buildMemorySSA(func.getRegion());
    if (!memSSA)
    {
        return mlir::failure();
    }

    using fptr_t = mlir::LogicalResult (*)(mlir::BufferAliasAnalysis&, MemorySSA&);
    const fptr_t funcs[] = {
        &optimizeUses,
        &foldLoads,
    };

    bool changed = false;
    bool repeat = false;

    do
    {
        repeat = false;
        for (auto func : funcs)
        {
            if (mlir::succeeded(func(aliases, *memSSA)))
            {
                changed = true;
                repeat = true;
            }
        }
    }
    while (repeat);

    return mlir::success(changed);
}
