// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CanonicalizeReductions.hpp"

#include "imex/Analysis/AliasAnalysis.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
struct CanonicalizeReduction : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  CanonicalizeReduction(mlir::MLIRContext *context,
                        imex::LocalAliasAnalysis &analysis)
      : mlir::OpRewritePattern<mlir::scf::ForOp>(context),
        aliasAnalysis(analysis) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::memref::LoadOp> loads;

    struct StoreDesc {
      StoreDesc(mlir::memref::StoreOp s) : store(s) {}

      mlir::memref::StoreOp store;
      llvm::SmallVector<mlir::memref::LoadOp, 1> loads;
      bool aliasing = false;

      bool isValid() const { return !aliasing && !loads.empty(); }
      mlir::ValueRange getIndices() { return store.getIndices(); }
    };

    llvm::SmallVector<StoreDesc> stores;

    auto &loopBlock = op.getLoopBody().front();

    auto visitor = [&](mlir::Operation *bodyOp) -> mlir::WalkResult {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(bodyOp)) {
        loads.emplace_back(load);
        return mlir::WalkResult::advance();
      }

      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(bodyOp)) {
        if (store->getParentOp() != op)
          return mlir::WalkResult::interrupt();

        stores.emplace_back(store);
        return mlir::WalkResult::advance();
      }

      if (bodyOp->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>() ||
          bodyOp->hasTrait<mlir::OpTrait::IsTerminator>())
        return mlir::WalkResult::advance();

      auto memEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(bodyOp);
      if (!memEffects || !memEffects.hasNoEffect())
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::advance();
    };

    if (loopBlock.walk(visitor).wasInterrupted())
      return mlir::failure();

    if (stores.empty())
      return mlir::failure();

    bool changed = false;
    for (auto &[i, desc] : llvm::enumerate(stores)) {
      if (desc.aliasing)
        continue;

      if (llvm::any_of(desc.getIndices(),
                       [](auto v) { return !mlir::getConstantIntValue(v); }))
        continue;

      auto store = desc.store;
      auto memref = store.getMemRef();

      bool aliasing = false;
      for (auto &otherDesc :
           llvm::makeMutableArrayRef(stores).drop_front(i + 1)) {
        auto otherMemref = otherDesc.store.getMemRef();
        if (!aliasAnalysis.alias(memref, otherMemref).isNo()) {
          desc.aliasing = true;
          otherDesc.aliasing = true;
          aliasing = true;
          break;
        }
      }

      if (aliasing)
        continue;

      mlir::DominanceInfo dom;
      for (auto load : loads) {
        auto loadMemref = load.getMemRef();
        if (loadMemref == memref) {
          if (load.getIndices() != store.getIndices() ||
              !dom.properlyDominates(load.getOperation(), store)) {
            desc.aliasing = true;
            aliasing = true;
            break;
          }

          desc.loads.emplace_back(load);
        } else if (!aliasAnalysis.alias(memref, loadMemref).isNo()) {
          desc.aliasing = true;
          aliasing = true;
          break;
        }
      }

      if (aliasing)
        continue;

      if (!desc.loads.empty())
        changed = true;
    }

    if (!changed)
      return mlir::failure();

    auto origBlockArgsCount = loopBlock.getNumArguments();

    auto oldInits = op.getInitArgs();
    auto oldInitsCount = static_cast<unsigned>(oldInits.size());
    llvm::SmallVector<mlir::Value> inits;
    inits.assign(oldInits.begin(), oldInits.end());

    auto loc = op.getLoc();
    for (auto &desc : stores) {
      if (!desc.isValid())
        continue;

      auto store = desc.store;
      auto memref = store.getMemRef();
      mlir::Value init =
          rewriter.create<mlir::memref::LoadOp>(loc, memref, desc.getIndices());
      inits.emplace_back(init);
    }

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        loc, op.getLowerBound(), op.getUpperBound(), op.getStep(), inits);

    auto &newRegion = newOp.getLoopBody();
    assert(llvm::hasSingleElement(newRegion));
    auto &newBlock = newRegion.front();
    rewriter.mergeBlocks(
        &loopBlock, &newBlock,
        newBlock.getArguments().take_front(origBlockArgsCount));

    auto oldYield = mlir::cast<mlir::scf::YieldOp>(newBlock.getTerminator());

    auto oldYieldArgs = oldYield.getResults();
    llvm::SmallVector<mlir::Value> newYieldArgs;
    newYieldArgs.assign(oldYieldArgs.begin(), oldYieldArgs.end());

    unsigned idx = 0;
    for (auto &desc : stores) {
      if (!desc.isValid())
        continue;

      mlir::Value init = newBlock.getArgument(origBlockArgsCount + idx);

      assert(!desc.loads.empty());
      for (auto load : desc.loads)
        rewriter.replaceOp(load, init);

      auto store = desc.store;
      newYieldArgs.emplace_back(store.getValue());
      ++idx;
    }

    rewriter.setInsertionPoint(oldYield);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(oldYield, newYieldArgs);

    rewriter.setInsertionPointAfter(newOp);

    idx = oldInitsCount;
    for (auto &desc : stores) {
      if (!desc.isValid())
        continue;

      auto newVal = newOp.getResult(idx);
      auto store = desc.store;
      auto memref = store.getMemRef();
      rewriter.create<mlir::memref::StoreOp>(loc, newVal, memref,
                                             desc.getIndices());
      rewriter.eraseOp(store);

      ++idx;
    }

    rewriter.replaceOp(op, newOp.getResults().take_front(oldInitsCount));
    return mlir::success();
  }

private:
  imex::LocalAliasAnalysis &aliasAnalysis;
};

struct CanonicalizeReductionsPass
    : public mlir::PassWrapper<CanonicalizeReductionsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeReductionsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto context = &getContext();

    auto &aliasAnalysis = getAnalysis<imex::LocalAliasAnalysis>();

    mlir::RewritePatternSet patterns(context);
    patterns.insert<CanonicalizeReduction>(context, aliasAnalysis);
    auto op = getOperation();
    (void)mlir::applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createCanonicalizeReductionsPass() {
  return std::make_unique<CanonicalizeReductionsPass>();
}
