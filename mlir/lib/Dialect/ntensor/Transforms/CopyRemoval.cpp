// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Dialect/ntensor/Transforms/CopyRemoval.hpp"

#include "imex/Analysis/AliasAnalysis.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/IR/Dominance.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace {
struct CopyRemovalPass
    : public mlir::PassWrapper<CopyRemovalPass,
                               mlir::InterfacePass<mlir::FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyRemovalPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
  }

  void runOnOperation() override {
    auto root = getOperation();

    llvm::SmallVector<mlir::Operation *> reads;
    llvm::SmallVector<mlir::Operation *> writes;
    llvm::SmallVector<imex::ntensor::CopyOp> copies;

    root->walk([&](mlir::Operation *op) {
      if (auto copy = mlir::dyn_cast<imex::ntensor::CopyOp>(op)) {
        copies.emplace_back(copy);
        reads.emplace_back(copy);
        writes.emplace_back(copy);
        return;
      }

      // Only process ops operation on ntensor arrays.
      if (!llvm::any_of(op->getOperands(), [](mlir::Value arg) {
            return arg.getType().isa<imex::ntensor::NTensorType>();
          }))
        return;

      auto memEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
      if (!memEffects) {
        // Conservatively assume them bot readesr and writers.
        reads.emplace_back(op);
        writes.emplace_back(op);
        return;
      }

      if (memEffects.hasEffect<mlir::MemoryEffects::Read>())
        reads.emplace_back(op);

      if (memEffects.hasEffect<mlir::MemoryEffects::Write>())
        writes.emplace_back(op);
    });

    if (copies.empty())
      return markAllAnalysesPreserved();

    auto &dom = getAnalysis<mlir::DominanceInfo>();
    auto &postDom = getAnalysis<mlir::PostDominanceInfo>();
    auto &aa = getAnalysis<imex::LocalAliasAnalysis>();

    auto inBetween = [&](mlir::Operation *op, mlir::Operation *begin,
                         mlir::Operation *end) -> bool {
      if (postDom.postDominates(begin, op))
        return false;

      if (dom.dominates(end, op))
        return false;

      return true;
    };

    auto hasAliasingWrites = [&](mlir::Value array, mlir::Operation *begin,
                                 mlir::Operation *end) -> bool {
      for (auto write : writes) {
        if (!inBetween(write, begin, end))
          continue;

        for (auto arg : write->getOperands()) {
          if (!arg.getType().isa<imex::ntensor::NTensorType>())
            continue;

          if (!aa.alias(array, arg).isNo())
            return true;
        }
      }
      return false;
    };

    mlir::OpBuilder builder(&getContext());

    // Propagate copy src.
    for (auto copy : copies) {
      auto src = copy.getSource();
      auto dst = copy.getTarget();
      for (auto &use : llvm::make_early_inc_range(dst.getUses())) {
        auto owner = use.getOwner();
        if (owner == copy || !dom.properlyDominates(copy, owner))
          continue;

        if (hasAliasingWrites(dst, copy, owner))
          continue;

        auto memInterface =
            mlir::dyn_cast<mlir::MemoryEffectOpInterface>(owner);
        if (!memInterface ||
            memInterface.getEffectOnValue<mlir::MemoryEffects::Write>(dst))
          continue;

        mlir::Value newArg = src;
        if (src.getType() != dst.getType()) {
          auto loc = owner->getLoc();
          builder.setInsertionPoint(owner);
          newArg =
              builder.create<imex::ntensor::CastOp>(loc, dst.getType(), newArg);
        }

        use.set(newArg);
      }
    }

    auto getNextCopy = [&](imex::ntensor::CopyOp src) -> imex::ntensor::CopyOp {
      for (auto copy : copies) {
        if (src == copy)
          continue;

        if (!dom.properlyDominates(src, copy))
          continue;

        if (src.getTarget() == copy.getTarget())
          return copy;
      }

      return {};
    };

    auto hasAliasingReads = [&](mlir::Value array, mlir::Operation *begin,
                                mlir::Operation *end) -> bool {
      for (auto read : reads) {
        if (!inBetween(read, begin, end))
          continue;

        for (auto arg : read->getOperands()) {
          if (!arg.getType().isa<imex::ntensor::NTensorType>())
            continue;

          if (!aa.alias(array, arg).isNo())
            return true;
        }
      }
      return false;
    };

    llvm::SmallVector<mlir::Operation *> toErase;

    // Remove redundant copies.
    for (auto copy : copies) {
      auto nextCopy = getNextCopy(copy);
      if (!nextCopy)
        continue;

      auto dst = copy.getTarget();
      if (hasAliasingReads(dst, copy, nextCopy))
        continue;

      toErase.emplace_back(copy);
    }

    for (auto op : toErase)
      op->erase();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::ntensor::createCopyRemovalPass() {
  return std::make_unique<CopyRemovalPass>();
}
