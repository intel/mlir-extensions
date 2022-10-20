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

#include "imex/Analysis/MemorySsaAnalysis.hpp"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace {
struct Meminfo {
  mlir::Value memref;
  mlir::ValueRange indices;
};

static llvm::Optional<Meminfo> getMeminfo(mlir::Operation *op) {
  assert(nullptr != op);
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return Meminfo{load.getMemref(), load.getIndices()};

  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return Meminfo{store.getMemref(), store.getIndices()};

  return {};
}
} // namespace

imex::MemorySSAAnalysis::MemorySSAAnalysis(mlir::Operation *op,
                                           mlir::AnalysisManager &am) {
  if (op->getNumRegions() != 1)
    return;

  memssa = buildMemorySSA(op->getRegion(0));
  if (memssa) {
    aliasAnalysis = &am.getAnalysis<mlir::AliasAnalysis>();
    (void)optimizeUses();
  }
}

mlir::LogicalResult imex::MemorySSAAnalysis::optimizeUses() {
  if (memssa) {
    assert(nullptr != aliasAnalysis);
    auto mayAlias = [&](mlir::Operation *op1, mlir::Operation *op2) {
      auto info1 = getMeminfo(op1);
      if (!info1)
        return true;

      auto info2 = getMeminfo(op2);
      if (!info2)
        return true;

      auto memref1 = info1->memref;
      auto memref2 = info2->memref;
      assert(memref1);
      assert(memref2);
      auto result = aliasAnalysis->alias(memref1, memref2);
      return !result.isNo();
    };
    return memssa->optimizeUses(mayAlias);
  }
  return mlir::failure();
}

bool imex::MemorySSAAnalysis::isInvalidated(
    const mlir::AnalysisManager::PreservedAnalyses &pa) {
  return !pa.isPreserved<MemorySSAAnalysis>() ||
         !pa.isPreserved<mlir::AliasAnalysis>();
}
