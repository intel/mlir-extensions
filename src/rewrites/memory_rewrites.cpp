#include "plier/rewrites/memory_rewrites.hpp"

#include <mlir/Analysis/BufferAliasAnalysis.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>

#include <plier/analysis/memory_ssa.hpp>

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

    (void)memSSA.optimizeUses(mayAlias);
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
