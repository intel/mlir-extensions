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

#include "plier/analysis/memory_ssa_analysis.hpp"

#include <mlir/Analysis/BufferAliasAnalysis.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

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
}

plier::MemorySSAAnalysis::MemorySSAAnalysis(mlir::Operation* op, mlir::AnalysisManager& am)
{
    if (op->getNumRegions() != 1)
    {
        return;
    }
    memssa = buildMemorySSA(op->getRegion(0));
    if (memssa)
    {
        aliasAnalysis = &am.getAnalysis<mlir::BufferAliasAnalysis>();
        (void)optimizeUses();
    }
}

mlir::LogicalResult plier::MemorySSAAnalysis::optimizeUses()
{
    if (memssa)
    {
        assert(nullptr != aliasAnalysis);
        auto mayAlias = [&](mlir::Operation* op1, mlir::Operation* op2)
        {
            auto info1 = getMeminfo(op1);
            if (!info1)
            {
                return true;
            }
            auto info2 = getMeminfo(op2);
            if (!info2)
            {
                return true;
            }
            return aliasAnalysis->resolve(info1->memref).count(info2->memref) != 0;
        };
        return memssa->optimizeUses(mayAlias);
    }
    return mlir::failure();
}

bool plier::MemorySSAAnalysis::isInvalidated(const mlir::AnalysisManager::PreservedAnalyses& pa)
{
    return !pa.isPreserved<MemorySSAAnalysis>() ||
           !pa.isPreserved<mlir::BufferAliasAnalysis>();
}
