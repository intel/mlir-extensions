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

#pragma once

#include "plier/analysis/memory_ssa.hpp"

#include <mlir/Pass/AnalysisManager.h>

namespace mlir {
class Operation;
class AliasAnalysis;
} // namespace mlir

namespace plier {
class MemorySSAAnalysis {
public:
  MemorySSAAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);
  MemorySSAAnalysis(const MemorySSAAnalysis &) = delete;

  mlir::LogicalResult optimizeUses();

  static bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa);

  llvm::Optional<plier::MemorySSA> memssa;
  mlir::AliasAnalysis *aliasAnalysis = nullptr;
};
} // namespace plier
