// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/LoopUtils.hpp"

#include "imex/Analysis/AliasAnalysis.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/plier/Dialect.hpp"
#include "imex/Transforms/CastUtils.hpp"
#include "imex/Transforms/ConstUtils.hpp"

#include <llvm/ADT/SmallVector.h>

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace {
template <typename Op>
static Op getNextOpInner(llvm::iterator_range<mlir::Block::iterator> &iters) {
  if (iters.empty())
    return nullptr;

  auto res = mlir::dyn_cast<Op>(iters.begin());
  if (res) {
    auto next = std::next(iters.begin());
    iters = {next, iters.end()};
  }
  return res;
}

template <typename Op>
static Op getNextOp(llvm::iterator_range<mlir::Block::iterator> &iters) {
  if (iters.empty())
    return nullptr;

  while (getNextOpInner<mlir::UnrealizedConversionCastOp>(iters) ||
         getNextOpInner<plier::CastOp>(iters)) {
  } // skip casts

  auto res = mlir::dyn_cast<Op>(iters.begin());
  if (res) {
    auto next = std::next(iters.begin());
    iters = {next, iters.end()};
  }
  return res;
}

static mlir::Value getLastIterValue(mlir::PatternRewriter &builder,
                                    mlir::Location loc, mlir::Value lower_bound,
                                    mlir::Value upper_bound, mlir::Value step) {
  auto len =
      builder.createOrFold<mlir::arith::SubIOp>(loc, upper_bound, lower_bound);
  auto count = builder.createOrFold<mlir::arith::DivSIOp>(loc, len, step);
  auto inc = builder.createOrFold<mlir::arith::MulIOp>(loc, count, step);
  return builder.createOrFold<mlir::arith::AddIOp>(loc, lower_bound, inc);
}

static mlir::Value getCastArg(mlir::Value val) {
  if (auto cast = val.getDefiningOp<plier::CastOp>())
    return cast.getValue();

  if (auto cast = val.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    auto inputs = cast.getInputs();
    if (inputs.size() == 1)
      return inputs.front();
  }

  return {};
}

static mlir::Value skipCasts(mlir::Value val) {
  while (auto castArg = getCastArg(val))
    val = castArg;

  return val;
};
} // namespace

bool imex::canLowerWhileToFor(mlir::scf::WhileOp whileOp) {
  auto &beforeBlock = whileOp.getBefore().front();
  auto iters = llvm::iterator_range<mlir::Block::iterator>(beforeBlock);
  auto iternext = getNextOp<plier::IternextOp>(iters);
  /*auto pairfirst =*/getNextOp<plier::PairfirstOp>(iters);
  auto pairsecond = getNextOp<plier::PairsecondOp>(iters);
  auto beforeTerm = getNextOp<mlir::scf::ConditionOp>(iters);

  if (!iternext || !pairsecond || !beforeTerm ||
      skipCasts(beforeTerm.getCondition()) != pairsecond)
    return false;

  return true;
}

llvm::SmallVector<mlir::scf::ForOp, 2> imex::lowerWhileToFor(
    mlir::scf::WhileOp whileOp, mlir::PatternRewriter &builder,
    llvm::function_ref<std::tuple<mlir::Value, mlir::Value, mlir::Value>(
        mlir::OpBuilder &, mlir::Location)>
        getBounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                   mlir::Type, mlir::Value)>
        getIterVal) {
  if (!canLowerWhileToFor(whileOp))
    return {};

  llvm::SmallVector<mlir::scf::ForOp, 2> results;
  auto loc = whileOp.getLoc();
  mlir::Value zeroVal;
  auto getZeroIndex = [&]() {
    if (!zeroVal) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(whileOp);
      zeroVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    }
    return zeroVal;
  };

  auto getNeg = [&](mlir::Value value) {
    return builder.createOrFold<mlir::arith::SubIOp>(loc, getZeroIndex(),
                                                     value);
  };

  auto &beforeBlock = whileOp.getBefore().front();
  auto iters = llvm::iterator_range<mlir::Block::iterator>(beforeBlock);
  /*auto iternext =*/getNextOp<plier::IternextOp>(iters);
  auto pairfirst = getNextOp<plier::PairfirstOp>(iters);
  auto beforeTerm =
      mlir::cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());
  auto &afterBlock = whileOp.getAfter().front();

  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    return ::imex::indexCast(builder, loc, val);
  };

  auto bounds = getBounds(builder, loc);
  auto origLowerBound = indexCast(std::get<0>(bounds));
  auto origUpperBound = indexCast(std::get<1>(bounds));
  auto origStep = indexCast(std::get<2>(bounds));

  // scf::ForOp/ParallelOp doesn't support negative step, so generate
  // IfOp and 2 version for different step signs
  // branches for const steps will be pruned later
  auto genFor = [&](bool positive) {
    auto getLoopBodyBuilder = [&](bool positive) {
      return [&, positive](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value iv, mlir::ValueRange iterargs) {
        if (!positive)
          iv = getNeg(iv);

        mlir::BlockAndValueMapping mapper;
        assert(beforeBlock.getNumArguments() == iterargs.size());
        assert(afterBlock.getNumArguments() == beforeTerm.getArgs().size());
        mapper.map(beforeBlock.getArguments(), iterargs);

        for (auto it :
             llvm::zip(afterBlock.getArguments(), beforeTerm.getArgs())) {
          auto blockArg = std::get<0>(it);
          auto termArg = std::get<1>(it);
          if (pairfirst && skipCasts(termArg) == pairfirst) {
            // iter arg
            auto iterVal = getIterVal(builder, loc, pairfirst.getType(), iv);
            iterVal = builder.createOrFold<plier::CastOp>(
                loc, blockArg.getType(), iterVal);
            mapper.map(blockArg, iterVal);
          } else {
            mapper.map(blockArg, mapper.lookupOrDefault(termArg));
          }
        }

        for (auto &op : afterBlock) // with terminator
          builder.clone(op, mapper);
      };
    };

    auto lowerBound = origLowerBound;
    auto upperBound = origUpperBound;
    auto step = origStep;

    if (!positive) {
      lowerBound = getNeg(lowerBound);
      upperBound = getNeg(upperBound);
      step = getNeg(step);
    }

    return builder.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step,
                                            whileOp.getOperands(), // iterArgs
                                            getLoopBodyBuilder(positive));
  };

  auto getIfBodyBuilder = [&](bool positive) {
    return [&, positive](mlir::OpBuilder &builder, mlir::Location loc) {
      auto loopOp = genFor(positive);
      results.emplace_back(loopOp);
      builder.create<mlir::scf::YieldOp>(loc, loopOp.getResults());
    };
  };

  builder.setInsertionPoint(whileOp);
  auto stepSign = builder.createOrFold<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, origStep, getZeroIndex());

  auto loopResults = [&]() -> mlir::ValueRange {
    auto ind = getConstVal<mlir::IntegerAttr>(stepSign);
    auto resTypes = whileOp.getOperands().getTypes();
    if (!ind)
      return builder
          .create<mlir::scf::IfOp>(loc, resTypes, stepSign,
                                   getIfBodyBuilder(true),
                                   getIfBodyBuilder(false))
          .getResults();

    auto reg = builder.create<mlir::scf::ExecuteRegionOp>(loc, resTypes);
    auto &regBlock = reg.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&regBlock);
    getIfBodyBuilder(ind.getValue() != 0)(builder, loc);
    return reg.getResults();
  }();

  assert(whileOp.getNumResults() >= loopResults.size());
  builder.updateRootInPlace(whileOp, [&]() {
    assert(whileOp.getNumResults() == beforeTerm.getArgs().size());
    for (auto it : llvm::zip(whileOp.getResults(), beforeTerm.getArgs())) {
      auto oldRes = std::get<0>(it);
      auto operand = std::get<1>(it);
      for (auto it2 : llvm::enumerate(beforeBlock.getArguments())) {
        auto arg = it2.value();
        if (arg == operand) {
          assert(it2.index() < loopResults.size());
          auto newRes = loopResults[static_cast<unsigned>(it2.index())];
          newRes = builder.createOrFold<plier::CastOp>(loc, oldRes.getType(),
                                                       newRes);
          oldRes.replaceAllUsesWith(newRes);
          break;
        }
      }
      if (pairfirst && skipCasts(operand) == pairfirst &&
          !oldRes.getUsers().empty()) {
        auto val = getLastIterValue(builder, loc, origLowerBound,
                                    origUpperBound, origStep);
        auto newRes =
            builder.createOrFold<plier::CastOp>(loc, oldRes.getType(), val);
        oldRes.replaceAllUsesWith(newRes);
      }
      assert(oldRes.getUsers().empty());
    }
  });

  assert(whileOp.getOperation()->getUsers().empty());
  builder.eraseOp(whileOp);
  return results;
}

mlir::LogicalResult imex::lowerWhileToFor(
    plier::GetiterOp getiter, mlir::PatternRewriter &builder,
    llvm::function_ref<std::tuple<mlir::Value, mlir::Value, mlir::Value>(
        mlir::OpBuilder &, mlir::Location)>
        getBounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                   mlir::Type, mlir::Value)>
        getIterVal,
    llvm::function_ref<void(mlir::scf::ForOp)> results) {
  llvm::SmallVector<mlir::scf::WhileOp, 4> toProcess;
  for (auto user : getiter.getOperation()->getUsers()) {
    if (auto whileOp =
            mlir::dyn_cast<mlir::scf::WhileOp>(user->getParentOp())) {
      toProcess.emplace_back(whileOp);
    }
  }

  bool changed = false;
  for (auto whileOp : toProcess) {
    auto res = lowerWhileToFor(whileOp, builder, getBounds, getIterVal);
    if (!res.empty()) {
      changed = true;
      if (results) {
        for (auto r : res)
          results(r);
      }
    }
  }

  if (getiter.getOperation()->getUsers().empty()) {
    builder.eraseOp(getiter);
    changed = true;
  }
  return mlir::success(changed);
}

// TODO: Copypasted from mlir
namespace {
using namespace mlir;

/// Verify there are no nested ParallelOps.
static bool hasNestedParallelOp(scf::ParallelOp ploop) {
  auto walkResult = ploop.getBody()->walk(
      [](scf::ParallelOp) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Verify equal iteration spaces.
static bool equalIterationSpaces(scf::ParallelOp firstPloop,
                                 scf::ParallelOp secondPloop) {
  if (firstPloop.getNumLoops() != secondPloop.getNumLoops())
    return false;

  auto matchOperands = [&](const OperandRange &lhs,
                           const OperandRange &rhs) -> bool {
    // TODO: Extend this to support aliases and equal constants.
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
  };
  return matchOperands(firstPloop.getLowerBound(),
                       secondPloop.getLowerBound()) &&
         matchOperands(firstPloop.getUpperBound(),
                       secondPloop.getUpperBound()) &&
         matchOperands(firstPloop.getStep(), secondPloop.getStep());
}

/// Checks if the parallel loops have mixed access to the same buffers. Returns
/// `true` if the first parallel loop writes to the same indices that the second
/// loop reads.
static bool haveNoReadsAfterWriteExceptSameIndex(
    scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
    const BlockAndValueMapping &firstToSecondPloopIndices,
    llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  DenseMap<Value, SmallVector<ValueRange, 1>> bufferStores;
  SmallVector<Value> bufferStoresVec;
  firstPloop.getBody()->walk([&](memref::StoreOp store) {
    bufferStores[store.getMemRef()].push_back(store.getIndices());
    bufferStoresVec.emplace_back(store.getMemRef());
  });
  auto walkResult = secondPloop.getBody()->walk([&](memref::LoadOp load) {
    // Stop if the memref is defined in secondPloop body. Careful alias analysis
    // is needed.
    auto *memrefDef = load.getMemRef().getDefiningOp();
    if (memrefDef && memrefDef->getBlock() == load->getBlock())
      return WalkResult::interrupt();

    if (bufferStores.empty())
      return WalkResult::advance();

    auto write = bufferStores.find(load.getMemRef());
    if (write == bufferStores.end()) {
      auto &analysis = getAnalysis();
      for (auto store : bufferStoresVec)
        if (!analysis.alias(store, load.getMemRef()).isNo())
          return WalkResult::interrupt();

      return WalkResult::advance();
    }

    // Allow only single write access per buffer.
    if (write->second.size() != 1)
      return WalkResult::interrupt();

    // Check that the load indices of secondPloop coincide with store indices of
    // firstPloop for the same memrefs.
    auto storeIndices = write->second.front();
    auto loadIndices = load.getIndices();
    if (storeIndices.size() != loadIndices.size())
      return WalkResult::interrupt();
    for (size_t i = 0, e = storeIndices.size(); i < e; ++i) {
      if (firstToSecondPloopIndices.lookupOrDefault(storeIndices[i]) !=
          loadIndices[i])
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

/// Analyzes dependencies in the most primitive way by checking simple read and
/// write patterns.
static LogicalResult
verifyDependencies(scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
                   const BlockAndValueMapping &firstToSecondPloopIndices,
                   llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  for (auto res : firstPloop.getResults()) {
    for (auto user : res.getUsers()) {
      if (secondPloop->isAncestor(user))
        return mlir::failure();
    }
  }

  if (!haveNoReadsAfterWriteExceptSameIndex(
          firstPloop, secondPloop, firstToSecondPloopIndices, getAnalysis))
    return failure();

  BlockAndValueMapping secondToFirstPloopIndices;
  secondToFirstPloopIndices.map(secondPloop.getBody()->getArguments(),
                                firstPloop.getBody()->getArguments());
  return success(haveNoReadsAfterWriteExceptSameIndex(
      secondPloop, firstPloop, secondToFirstPloopIndices, getAnalysis));
}

static bool
isFusionLegal(scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
              const BlockAndValueMapping &firstToSecondPloopIndices,
              llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  return !hasNestedParallelOp(firstPloop) &&
         !hasNestedParallelOp(secondPloop) &&
         equalIterationSpaces(firstPloop, secondPloop) &&
         succeeded(verifyDependencies(firstPloop, secondPloop,
                                      firstToSecondPloopIndices, getAnalysis));
}

/// Prepends operations of firstPloop's body into secondPloop's body.
static bool
fuseIfLegal(scf::ParallelOp firstPloop, scf::ParallelOp &secondPloop,
            OpBuilder &b,
            llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  BlockAndValueMapping firstToSecondPloopIndices;
  firstToSecondPloopIndices.map(firstPloop.getBody()->getArguments(),
                                secondPloop.getBody()->getArguments());

  if (!isFusionLegal(firstPloop, secondPloop, firstToSecondPloopIndices,
                     getAnalysis))
    return false;

  mlir::DominanceInfo dom;
  for (auto user : firstPloop->getUsers())
    if (!dom.properlyDominates(secondPloop, user))
      return false;

  auto init1 = firstPloop.getInitVals();
  auto numResults1 = init1.size();
  auto init2 = secondPloop.getInitVals();
  auto numResults2 = init2.size();

  SmallVector<mlir::Value> newInitVars;
  newInitVars.reserve(numResults1 + numResults2);
  newInitVars.assign(init2.begin(), init2.end());
  newInitVars.append(init1.begin(), init1.end());

  b.setInsertionPoint(secondPloop);
  auto newSecondPloop = b.create<mlir::scf::ParallelOp>(
      secondPloop.getLoc(), secondPloop.getLowerBound(),
      secondPloop.getUpperBound(), secondPloop.getStep(), newInitVars);
  if (secondPloop->hasAttr(imex::util::attributes::getParallelName()))
    newSecondPloop->setAttr(imex::util::attributes::getParallelName(),
                            mlir::UnitAttr::get(b.getContext()));

  newSecondPloop.getRegion().getBlocks().splice(
      newSecondPloop.getRegion().begin(), secondPloop.getRegion().getBlocks());
  auto term =
      mlir::cast<mlir::scf::YieldOp>(newSecondPloop.getBody()->getTerminator());

  b.setInsertionPointToStart(newSecondPloop.getBody());
  for (auto &op : firstPloop.getBody()->without_terminator()) {
    if (isa<mlir::scf::ReduceOp>(op)) {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPoint(term);
      b.clone(op, firstToSecondPloopIndices);
    } else {
      b.clone(op, firstToSecondPloopIndices);
    }
  }
  firstPloop.replaceAllUsesWith(
      newSecondPloop.getResults().take_back(numResults1));
  firstPloop.erase();
  secondPloop.replaceAllUsesWith(
      newSecondPloop.getResults().take_front(numResults2));
  secondPloop.erase();
  secondPloop = newSecondPloop;
  return true;
}

bool hasNoEffect(mlir::Operation *op) {
  if (op->getNumRegions() != 0)
    return false;

  if (mlir::isa<mlir::CallOpInterface>(op))
    return false;

  if (auto interface = dyn_cast<MemoryEffectOpInterface>(op))
    return !interface.hasEffect<mlir::MemoryEffects::Read>() &&
           !interface.hasEffect<mlir::MemoryEffects::Write>();

  return !op->hasTrait<::mlir::OpTrait::HasRecursiveMemoryEffects>();
}

bool hasNoEffect(mlir::scf::ParallelOp currentPloop, mlir::Operation *op) {
  if (currentPloop && currentPloop->getNumResults() != 0) {
    for (auto arg : op->getOperands()) {
      if (llvm::is_contained(currentPloop.getResults(), arg))
        return false;
    }
  }

  return hasNoEffect(op);
}
} // namespace

mlir::LogicalResult imex::naivelyFuseParallelOps(Region &region) {
  std::unique_ptr<mlir::AliasAnalysis> analysis;
  auto getAnalysis = [&]() -> mlir::AliasAnalysis & {
    if (!analysis) {
      auto parent = region.getParentOfType<mlir::FunctionOpInterface>();
      analysis = std::make_unique<mlir::AliasAnalysis>(parent);
      analysis->addAnalysisImplementation(imex::LocalAliasAnalysis());
    }

    return *analysis;
  };
  OpBuilder b(region);
  // Consider every single block and attempt to fuse adjacent loops.
  bool changed = false;
  SmallVector<SmallVector<scf::ParallelOp, 8>, 1> ploopChains;
  for (auto &block : region) {
    for (auto &op : block)
      for (auto &innerReg : op.getRegions())
        if (succeeded(naivelyFuseParallelOps(innerReg)))
          changed = true;

    ploopChains.clear();
    ploopChains.push_back({});
    // Not using `walk()` to traverse only top-level parallel loops and also
    // make sure that there are no side-effecting ops between the parallel
    // loops.
    scf::ParallelOp currentPloop;
    bool noSideEffects = true;
    for (auto &op : block) {
      if (auto ploop = dyn_cast<scf::ParallelOp>(op)) {
        currentPloop = ploop;
        if (noSideEffects) {
          ploopChains.back().push_back(ploop);
        } else {
          ploopChains.push_back({ploop});
          noSideEffects = true;
        }
        continue;
      }
      // TODO: Handle region side effects properly.
      noSideEffects &= hasNoEffect(currentPloop, &op);
    }
    for (llvm::MutableArrayRef<scf::ParallelOp> ploops : ploopChains) {
      for (size_t i = 0, e = ploops.size(); i + 1 < e; ++i)
        if (fuseIfLegal(ploops[i], ploops[i + 1], b, getAnalysis))
          changed = true;
    }
  }
  return mlir::success(changed);
}

LogicalResult imex::prepareForFusion(Region &region) {
  DominanceInfo dom(region.getParentOp());
  bool changed = false;
  for (auto &block : region) {
    for (auto &parallelOp : llvm::make_early_inc_range(block)) {
      for (auto &innerReg : parallelOp.getRegions())
        if (succeeded(prepareForFusion(innerReg)))
          changed = true;

      if (!isa<scf::ParallelOp>(parallelOp))
        continue;

      auto it = Block::iterator(parallelOp);
      if (it == block.begin())
        continue;

      --it;

      auto terminate = false;
      while (!terminate) {
        auto &currentOp = *it;
        if (isa<scf::ParallelOp>(currentOp))
          break;

        if (it == block.begin()) {
          terminate = true;
        } else {
          --it;
        }

        bool canMove = [&]() {
          if (!hasNoEffect(&currentOp))
            return false;

          for (auto arg : currentOp.getOperands())
            if (!dom.properlyDominates(arg, &parallelOp))
              return false;

          for (auto user : currentOp.getUsers())
            if (parallelOp.isAncestor(user) ||
                !dom.properlyDominates(&parallelOp, user))
              return false;

          return true;
        }();

        if (canMove) {
          currentOp.moveAfter(&parallelOp);
          changed = true;
        }
      }
    }
  }
  return mlir::success(changed);
}
