// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CanonicalizeReductions.hpp"

#include "imex/Analysis/AliasAnalysis.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static bool checkMemrefType(mlir::Value value) {
  if (auto type = value.getType().dyn_cast<mlir::MemRefType>()) {
    //        auto shape = type.getShape();
    //        return shape.empty() || (1 == shape.size() && 1 == shape[0]);
    return true;
  }
  return false;
}

static bool isOutsideBlock(mlir::ValueRange values, mlir::Block &block) {
  auto blockArgs = block.getArguments();
  for (auto val : values) {
    if (llvm::is_contained(blockArgs, val))
      return false;

    auto op = val.getDefiningOp();
    if (op && block.findAncestorOpInBlock(*op))
      return false;
  }
  return true;
}

static bool checkForPotentialAliases(mlir::Value value,
                                     mlir::Operation *parent) {
  assert(parent->getRegions().size() == 1);
  assert(llvm::hasNItems(parent->getRegions().front(), 1));
  if (auto effects = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(
          value.getDefiningOp())) {
    if (!effects.onlyHasEffect<mlir::MemoryEffects::Allocate>())
      return false;
  } else {
    return false;
  }

  mlir::memref::LoadOp load;
  mlir::memref::StoreOp store;
  auto &parentBlock = parent->getRegions().front().front();
  for (auto user : value.getUsers()) {
    if (mlir::isa<mlir::ViewLikeOpInterface>(user))
      return false; // TODO: very conservative

    if (!parent->isProperAncestor(user))
      continue;

    if (auto effects =
            mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(user)) {
      if (user->getBlock() != &parentBlock)
        return false;

      if (effects.hasEffect<mlir::MemoryEffects::Read>()) {
        if (load || !mlir::isa<mlir::memref::LoadOp>(user))
          return false;

        load = mlir::cast<mlir::memref::LoadOp>(user);
      }
      if (effects.hasEffect<mlir::MemoryEffects::Write>()) {
        if (store || !mlir::isa<mlir::memref::StoreOp>(user))
          return false;

        store = mlir::cast<mlir::memref::StoreOp>(user);
      }
    }
  }
  if (!load || !store || !load->isBeforeInBlock(store) ||
      load.getIndices() != store.getIndices() ||
      !isOutsideBlock(load.getIndices(), parentBlock)) {
    return false;
  }
  return true;
}

static bool checkSupportedOps(mlir::Value value, mlir::Operation *parent) {
  for (auto user : value.getUsers()) {
    if (user->getParentOp() == parent &&
        !mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user))
      return false;
  }
  return true;
}

static bool checkMemref(mlir::Value value, mlir::Operation *parent) {
  return checkMemrefType(value) && checkForPotentialAliases(value, parent) &&
         checkSupportedOps(value, parent);
}

static mlir::Value createScalarLoad(mlir::PatternRewriter &builder,
                                    mlir::Location loc, mlir::Value memref,
                                    mlir::ValueRange indices) {
  return builder.create<mlir::memref::LoadOp>(loc, memref, indices);
}

static void createScalarStore(mlir::PatternRewriter &builder,
                              mlir::Location loc, mlir::Value val,
                              mlir::Value memref, mlir::ValueRange indices) {
  builder.create<mlir::memref::StoreOp>(loc, val, memref, indices);
}

mlir::LogicalResult imex::CanonicalizeReduction::matchAndRewrite(
    mlir::scf::ForOp op, mlir::PatternRewriter &rewriter) const {
  llvm::SmallVector<std::pair<mlir::Value, mlir::ValueRange>> toProcess;
  for (auto &current : op.getLoopBody().front()) {
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(current)) {
      auto memref = load.getMemref();
      if (checkMemref(memref, op))
        toProcess.push_back({memref, load.getIndices()});
    }
  }

  if (!toProcess.empty()) {
    auto loc = op.getLoc();
    auto initArgs = llvm::to_vector<8>(op.getInitArgs());
    for (auto it : toProcess) {
      initArgs.emplace_back(
          createScalarLoad(rewriter, loc, it.first, it.second));
    }
    auto prevArgsOffset = op.getInitArgs().size();
    auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value iter, mlir::ValueRange iterVals) {
      auto &oldBody = op.getLoopBody().front();
      mlir::BlockAndValueMapping mapping;
      mapping.map(oldBody.getArguments().front(), iter);
      mapping.map(oldBody.getArguments().drop_front(), iterVals);
      auto yieldArgs = llvm::to_vector(iterVals);
      for (auto &bodyOp : oldBody.without_terminator()) {
        auto invalidIndex = static_cast<unsigned>(-1);
        auto getIterIndex = [&](auto op) -> unsigned {
          auto arg = op.getMemref();
          for (auto it : llvm::enumerate(llvm::make_first_range(toProcess))) {
            if (arg == it.value()) {
              return static_cast<unsigned>(it.index() + prevArgsOffset);
            }
          }
          return invalidIndex;
        };
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(bodyOp)) {
          auto index = getIterIndex(load);
          if (index != invalidIndex) {
            mapping.map(bodyOp.getResults().front(), yieldArgs[index]);
          } else {
            builder.clone(bodyOp, mapping);
          }
        } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(bodyOp)) {
          auto index = getIterIndex(store);
          if (index != invalidIndex) {
            yieldArgs[index] = mapping.lookup(store.getValue());
          } else {
            builder.clone(bodyOp, mapping);
          }
        } else {
          builder.clone(bodyOp, mapping);
        }
      }
      auto yield = mlir::cast<mlir::scf::YieldOp>(oldBody.getTerminator());
      llvm::copy(yield.getResults(), yieldArgs.begin());
      builder.create<mlir::scf::YieldOp>(loc, yieldArgs);
    };
    auto results = rewriter
                       .create<mlir::scf::ForOp>(loc, op.getLowerBound(),
                                                 op.getUpperBound(),
                                                 op.getStep(), initArgs, body)
                       .getResults();
    for (auto it : llvm::enumerate(toProcess)) {
      auto index = prevArgsOffset + it.index();
      auto result = results[static_cast<unsigned>(index)];
      createScalarStore(rewriter, loc, result, it.value().first,
                        it.value().second);
    }
    rewriter.replaceOp(op, results.take_front(prevArgsOffset));
    return mlir::success();
  }

  return mlir::failure();
}

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
    for (auto &bodyOp : loopBlock.without_terminator()) {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(bodyOp)) {
        loads.emplace_back(load);
        continue;
      }

      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(bodyOp)) {
        stores.emplace_back(store);
        continue;
      }

      if (bodyOp.getNumRegions() != 0)
        return mlir::failure();

      auto memEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(bodyOp);
      if (!memEffects || !memEffects.hasNoEffect())
        return mlir::failure();
    }

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

      for (auto load : loads) {
        auto loadMemref = load.getMemRef();
        if (loadMemref == memref) {
          if (load.getIndices() != store.getIndices() ||
              !load->isBeforeInBlock(store)) {
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
