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

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>

#include "plier/analysis/memory_ssa_analysis.hpp"

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
};

mlir::LogicalResult optimizeUses(plier::MemorySSAAnalysis& memSSAAnalysis)
{
    return memSSAAnalysis.optimizeUses();
}

mlir::LogicalResult foldLoads(plier::MemorySSAAnalysis& memSSAAnalysis)
{
    assert(memSSAAnalysis.memssa);
    auto& memSSA = *memSSAAnalysis.memssa;
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
            if (MustAlias()(op1, op2))
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

mlir::LogicalResult deadStoreElemination(plier::MemorySSAAnalysis& memSSAAnalysis)
{
    assert(memSSAAnalysis.memssa);
    auto& memSSA = *memSSAAnalysis.memssa;
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
                if (MustAlias()(op1, op2))
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

mlir::LogicalResult plier::optimizeMemoryOps(mlir::AnalysisManager& am)
{
    auto& memSSAAnalysis = am.getAnalysis<MemorySSAAnalysis>();
    if (!memSSAAnalysis.memssa)
    {
        return mlir::failure();
    }

    using fptr_t = mlir::LogicalResult (*)(MemorySSAAnalysis&);
    const fptr_t funcs[] = {
        &optimizeUses,
        &foldLoads,
        &deadStoreElemination,
    };

    bool changed = false;
    bool repeat = false;

    do
    {
        repeat = false;
        for (auto func : funcs)
        {
            if (mlir::succeeded(func(memSSAAnalysis)))
            {
                changed = true;
                repeat = true;
            }
        }
    }
    while (repeat);

    return mlir::success(changed);
}
