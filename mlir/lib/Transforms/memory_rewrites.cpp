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

#include "mlir-extensions/Transforms/memory_rewrites.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>

#include "mlir-extensions/analysis/memory_ssa_analysis.hpp"

namespace {
struct Meminfo {
  mlir::Value memref;
  mlir::ValueRange indices;
};

llvm::Optional<Meminfo> getMeminfo(mlir::Operation *op) {
  assert(nullptr != op);
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
    return Meminfo{load.memref(), load.indices()};
  }
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    return Meminfo{store.memref(), store.indices()};
  }
  return {};
}

struct MustAlias {
  bool operator()(mlir::Operation *op1, mlir::Operation *op2) const {
    auto meminfo1 = getMeminfo(op1);
    if (!meminfo1) {
      return false;
    }
    auto meminfo2 = getMeminfo(op2);
    if (!meminfo2) {
      return false;
    }
    return meminfo1->memref == meminfo2->memref &&
           meminfo1->indices == meminfo2->indices;
  }
};

mlir::LogicalResult optimizeUses(plier::MemorySSAAnalysis &memSSAAnalysis) {
  return memSSAAnalysis.optimizeUses();
}

mlir::LogicalResult foldLoads(plier::MemorySSAAnalysis &memSSAAnalysis) {
  assert(memSSAAnalysis.memssa);
  auto &memSSA = *memSSAAnalysis.memssa;
  using NodeType = plier::MemorySSA::NodeType;
  bool changed = false;
  for (auto &node : llvm::make_early_inc_range(memSSA.getNodes())) {
    if (NodeType::Use == memSSA.getNodeType(&node)) {
      auto def = memSSA.getNodeDef(&node);
      assert(nullptr != def);
      if (NodeType::Def != memSSA.getNodeType(def)) {
        continue;
      }
      auto op1 = memSSA.getNodeOperation(&node);
      auto op2 = memSSA.getNodeOperation(def);
      assert(nullptr != op1);
      assert(nullptr != op2);
      if (MustAlias()(op1, op2)) {
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

mlir::LogicalResult
deadStoreElemination(plier::MemorySSAAnalysis &memSSAAnalysis) {
  assert(memSSAAnalysis.memssa);
  auto &memSSA = *memSSAAnalysis.memssa;
  using NodeType = plier::MemorySSA::NodeType;
  auto getNextDef =
      [&](plier::MemorySSA::Node *node) -> plier::MemorySSA::Node * {
    plier::MemorySSA::Node *def = nullptr;
    for (auto user : memSSA.getUsers(node)) {
      auto type = memSSA.getNodeType(user);
      if (NodeType::Def == type) {
        if (def != nullptr) {
          return nullptr;
        }
        def = user;
      } else {
        return nullptr;
      }
    }
    return def;
  };
  bool changed = false;
  for (auto &node : llvm::make_early_inc_range(memSSA.getNodes())) {
    if (NodeType::Def == memSSA.getNodeType(&node)) {
      if (auto nextDef = getNextDef(&node)) {
        auto op1 = memSSA.getNodeOperation(&node);
        auto op2 = memSSA.getNodeOperation(nextDef);
        assert(nullptr != op1);
        assert(nullptr != op2);
        if (MustAlias()(op1, op2)) {
          op1->erase();
          memSSA.eraseNode(&node);
          changed = true;
        }
      }
    }
  }
  return mlir::success(changed);
}

struct SimpleOperationInfo : public llvm::DenseMapInfo<mlir::Operation *> {
  static unsigned getHashValue(const mlir::Operation *opC) {
    return static_cast<unsigned>(mlir::OperationEquivalence::computeHash(
        const_cast<mlir::Operation *>(opC),
        mlir::OperationEquivalence::directHashValue,
        mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::IgnoreLocations));
  }
  static bool isEqual(const mlir::Operation *lhsC,
                      const mlir::Operation *rhsC) {
    auto *lhs = const_cast<mlir::Operation *>(lhsC);
    auto *rhs = const_cast<mlir::Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return mlir::OperationEquivalence::isEquivalentTo(
        const_cast<mlir::Operation *>(lhsC),
        const_cast<mlir::Operation *>(rhsC),
        mlir::OperationEquivalence::exactValueMatch,
        mlir::OperationEquivalence::ignoreValueEquivalence,
        mlir::OperationEquivalence::IgnoreLocations);
  }
};

mlir::LogicalResult loadCSE(plier::MemorySSAAnalysis &memSSAAnalysis) {
  mlir::DominanceInfo dom;
  assert(memSSAAnalysis.memssa);
  auto &memSSA = *memSSAAnalysis.memssa;
  using NodeType = plier::MemorySSA::NodeType;
  bool changed = false;
  llvm::SmallDenseMap<mlir::Operation *, mlir::Operation *, 4,
                      SimpleOperationInfo>
      opsMap;
  for (auto &node : memSSA.getNodes()) {
    auto nodeType = memSSA.getNodeType(&node);
    if (NodeType::Def != nodeType && NodeType::Phi != nodeType &&
        NodeType::Root != nodeType)
      continue;

    opsMap.clear();
    for (auto user : memSSA.getUsers(&node)) {
      if (memSSA.getNodeType(user) != NodeType::Use)
        continue;

      auto op = memSSA.getNodeOperation(user);
      if (!op->getRegions().empty())
        continue;

      auto it = opsMap.find(op);
      if (it == opsMap.end()) {
        opsMap.insert({op, op});
      } else {
        auto firstUser = it->second;
        if (!MustAlias()(op, firstUser))
          continue;

        if (dom.properlyDominates(op, firstUser)) {
          firstUser->replaceAllUsesWith(op);
          opsMap[firstUser] = op;
          auto firstUserNode = memSSA.getNode(firstUser);
          assert(firstUserNode);
          memSSA.eraseNode(firstUserNode);
          firstUser->erase();
          changed = true;
        } else if (dom.properlyDominates(firstUser, op)) {
          op->replaceAllUsesWith(firstUser);
          op->erase();
          memSSA.eraseNode(user);
          changed = true;
        }
      }
    }
  }
  return mlir::success(changed);
}

} // namespace

llvm::Optional<mlir::LogicalResult>
plier::optimizeMemoryOps(mlir::AnalysisManager &am) {
  auto &memSSAAnalysis = am.getAnalysis<MemorySSAAnalysis>();
  if (!memSSAAnalysis.memssa) {
    return {};
  }

  using fptr_t = mlir::LogicalResult (*)(MemorySSAAnalysis &);
  const fptr_t funcs[] = {
      &optimizeUses,
      &foldLoads,
      &deadStoreElemination,
      &loadCSE,
  };

  bool changed = false;
  bool repeat = false;

  do {
    repeat = false;
    for (auto func : funcs) {
      if (mlir::succeeded(func(memSSAAnalysis))) {
        changed = true;
        repeat = true;
      }
    }
  } while (repeat);

  return mlir::success(changed);
}
