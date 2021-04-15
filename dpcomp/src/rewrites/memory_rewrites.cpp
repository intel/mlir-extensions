// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "plier/rewrites/memory_rewrites.hpp"

#include <mlir/Analysis/BufferAliasAnalysis.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>

#include <plier/analysis/memory_ssa.hpp>

namespace
{
mlir::LogicalResult simplifyAlloc(mlir::Operation* op, mlir::PatternRewriter& rewriter)
{
    for (auto user : op->getUsers())
    {
        if (!mlir::isa<mlir::memref::StoreOp, mlir::memref::DeallocOp>(user))
        {
            return mlir::failure();
        }
    }

    for (auto user : llvm::make_early_inc_range(op->getUsers()))
    {
        rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);
    return mlir::success();
}
}

mlir::LogicalResult plier::RemoveTrivialAlloc::matchAndRewrite(mlir::memref::AllocOp op, mlir::PatternRewriter& rewriter) const
{
    return simplifyAlloc(op, rewriter);
}

mlir::LogicalResult plier::RemoveTrivialAlloca::matchAndRewrite(mlir::memref::AllocaOp op, mlir::PatternRewriter& rewriter) const
{
    return simplifyAlloc(op, rewriter);
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
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    {
        return Meminfo{load.memref(), load.indices()};
    }
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    {
        return Meminfo{store.memref(), store.indices()};
    }
    return {};
}

struct MayAlias
{
    bool operator()(mlir::Operation* op1, mlir::Operation* op2) const
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
    }

    mlir::BufferAliasAnalysis& aliases;
};

struct MustAlias
{
    bool operator()(mlir::Operation* op1, mlir::Operation* op2) const
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
    }

    mlir::BufferAliasAnalysis& aliases;
};

mlir::LogicalResult optimizeUses(mlir::BufferAliasAnalysis& aliases, plier::MemorySSA& memSSA)
{
    (void)memSSA.optimizeUses(MayAlias{aliases});
    return mlir::failure();
}

mlir::LogicalResult foldLoads(mlir::BufferAliasAnalysis& aliases, plier::MemorySSA& memSSA)
{
    using NodeType = plier::MemorySSA::NodeType;
    bool changed = false;
    for (auto& node : llvm::make_early_inc_range(memSSA.getNodes()))
    {
        if (NodeType::Use == memSSA.getNodeType(&node))
        {
            auto def = memSSA.getNodeDef(&node);
            assert(nullptr != def);
            if (NodeType::Def != memSSA.getNodeType(def))
            {
                continue;
            }
            auto op1 = memSSA.getNodeOperation(&node);
            auto op2 = memSSA.getNodeOperation(def);
            assert(nullptr != op1);
            assert(nullptr != op2);
            if (MustAlias{aliases}(op1, op2))
            {
                auto val = mlir::cast<mlir::memref::StoreOp>(op2).value();
                op1->replaceAllUsesWith(mlir::ValueRange(val));
                op1->erase();
                memSSA.eraseNode(&node);
                changed = true;
            }
        }
    }
    return mlir::success(changed);
}

mlir::LogicalResult deadStoreElemination(mlir::BufferAliasAnalysis& aliases, plier::MemorySSA& memSSA)
{
    using NodeType = plier::MemorySSA::NodeType;
    auto getNextDef = [&](plier::MemorySSA::Node* node)->plier::MemorySSA::Node*
    {
        plier::MemorySSA::Node* def = nullptr;
        for (auto user : memSSA.getUsers(node))
        {
            auto type = memSSA.getNodeType(user);
            if (NodeType::Def == type)
            {
                if (def != nullptr)
                {
                    return nullptr;
                }
                def = user;
            }
            else
            {
                return nullptr;
            }
        }
        return def;
    };
    bool changed = false;
    for (auto& node : llvm::make_early_inc_range(memSSA.getNodes()))
    {
        if (NodeType::Def == memSSA.getNodeType(&node))
        {
            if (auto nextDef = getNextDef(&node))
            {
                auto op1 = memSSA.getNodeOperation(&node);
                auto op2 = memSSA.getNodeOperation(nextDef);
                assert(nullptr != op1);
                assert(nullptr != op2);
                if (MustAlias{aliases}(op1, op2))
                {
                    op1->erase();
                    memSSA.eraseNode(&node);
                    changed = true;
                }
            }
        }
    }
    return mlir::success(changed);
}

}

mlir::LogicalResult plier::optimizeMemoryOps(mlir::FuncOp func)
{
    auto memSSA = buildMemorySSA(func.getRegion());
    if (!memSSA)
    {
        return mlir::failure();
    }
    mlir::BufferAliasAnalysis aliases(func);

    using fptr_t = mlir::LogicalResult (*)(mlir::BufferAliasAnalysis&, MemorySSA&);
    const fptr_t funcs[] = {
        &optimizeUses, // must be first
        &foldLoads,
        &deadStoreElemination
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
