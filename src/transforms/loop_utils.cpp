#include "plier/transforms/loop_utils.hpp"

#include <llvm/ADT/SmallVector.h>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "plier/dialect.hpp"

#include "plier/transforms/cast_utils.hpp"

namespace
{
template<typename Op>
Op get_next_op(llvm::iterator_range<mlir::Block::iterator>& iters)
{
    if (iters.empty())
    {
        return nullptr;
    }
    auto res = mlir::dyn_cast<Op>(iters.begin());
    if (res)
    {
        auto next = std::next(iters.begin());
        iters = {next, iters.end()};
    }
    return res;
}

mlir::Value get_last_iter_value(
    mlir::PatternRewriter& builder, mlir::Location loc,
    mlir::Value lower_bound, mlir::Value upper_bound, mlir::Value step)
{
    auto len = builder.create<mlir::SubIOp>(loc, upper_bound, lower_bound);
    auto count = builder.create<mlir::SignedDivIOp>(loc, len, step);
    auto inc = builder.create<mlir::MulIOp>(loc, count, step);
    return builder.create<mlir::AddIOp>(loc, lower_bound, inc);
}

}

mlir::LogicalResult plier::lower_while_to_for(
    plier::GetiterOp getiter, mlir::PatternRewriter& builder,
    llvm::function_ref<std::tuple<mlir::Value,mlir::Value,mlir::Value>(mlir::OpBuilder&, mlir::Location)> get_bounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Type, mlir::Value)> get_iter_val,
    llvm::function_ref<void(mlir::scf::ForOp)> results)
{
    llvm::SmallVector<mlir::scf::WhileOp, 4> to_process;
    for (auto user : getiter.getOperation()->getUsers())
    {
        if( auto while_op = mlir::dyn_cast<mlir::scf::WhileOp>(user->getParentOp()))
        {
            to_process.emplace_back(while_op);
        }
    }

    auto loc = getiter.getLoc();
    mlir::Value zero_val;
    auto get_zero_index = [&]()
    {
        if (!zero_val)
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(getiter);
            zero_val = builder.create<mlir::ConstantIndexOp>(loc, 0);
        }
        return zero_val;
    };

    auto get_neg = [&](mlir::Value value)
    {
        return builder.create<mlir::SubIOp>(loc, get_zero_index(), value);
    };

    bool changed = false;
    for (auto while_op : to_process)
    {
        auto& before_block = while_op.before().front();
        auto iters = llvm::iterator_range<mlir::Block::iterator>(before_block);
        auto iternext = get_next_op<plier::IternextOp>(iters);
        auto pairfirst = get_next_op<plier::PairfirstOp>(iters);
        auto pairsecond = get_next_op<plier::PairsecondOp>(iters);
        while (get_next_op<plier::CastOp>(iters)) {} // skip casts
        auto before_term = get_next_op<mlir::scf::ConditionOp>(iters);

        auto skip_casts = [](mlir::Value op)
        {
            while (auto cast = mlir::dyn_cast_or_null<plier::CastOp>(op.getDefiningOp()))
            {
                op = cast.getOperand();
            }
            return op;
        };
        if (!iternext || !pairsecond || !before_term ||
            skip_casts(before_term.condition()) != pairsecond)
        {
            continue;
        }

        auto& after_block = while_op.after().front();

        auto index_cast = [&](mlir::Value val)->mlir::Value
        {
            return ::plier::index_cast(builder, loc, val);
        };

        auto bounds = get_bounds(builder, loc);
        auto orig_lower_bound = index_cast(std::get<0>(bounds));
        auto orig_upper_bound = index_cast(std::get<1>(bounds));
        auto orig_step = index_cast(std::get<2>(bounds));

        // scf::ForOp/ParallelOp doesn't support negative step, so generate
        // IfOp and 2 version for different step signs
        // branches for const steps will be pruned later
        auto gen_for = [&](bool positive)
        {
            auto get_loop_body_builder = [&](bool positive)
            {
                return [&, positive](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterargs)
                {
                    if (!positive)
                    {
                        iv = get_neg(iv);
                    }
                    mlir::BlockAndValueMapping mapper;
                    assert(before_block.getNumArguments() == iterargs.size());
                    assert(after_block.getNumArguments() == before_term.args().size());
                    mapper.map(before_block.getArguments(), iterargs);
                    for (auto it : llvm::zip(after_block.getArguments(), before_term.args()))
                    {
                        auto block_arg = std::get<0>(it);
                        auto term_arg = std::get<1>(it);
                        if (pairfirst && term_arg == pairfirst) // iter arg
                        {
                            auto iter_val = get_iter_val(builder, loc, pairfirst.getType(), iv);
                            mapper.map(block_arg, iter_val);
                        }
                        else
                        {
                            mapper.map(block_arg, mapper.lookupOrDefault(term_arg));
                        }
                    }

                    for (auto& op : after_block) // with terminator
                    {
                        builder.clone(op, mapper);
                    }
                };
            };

            auto lower_bound = orig_lower_bound;
            auto upper_bound = orig_upper_bound;
            auto step = orig_step;

            if (!positive)
            {
                lower_bound = get_neg(lower_bound);
                upper_bound = get_neg(upper_bound);
                step = get_neg(step);
            }

            return builder.create<mlir::scf::ForOp>(
                loc,
                lower_bound,
                upper_bound,
                step,
                while_op.getOperands(), // iterArgs
                get_loop_body_builder(positive)
                );
        };


        auto get_if_body_builder = [&](bool positive)
        {
            return [&, positive](mlir::OpBuilder& builder, mlir::Location loc)
            {
                auto loop_op = gen_for(positive);
                if (results)
                {
                    results(loop_op);
                }
                builder.create<mlir::scf::YieldOp>(loc, loop_op.getResults());
            };
        };

        builder.setInsertionPoint(while_op);
        auto step_sign = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sge, orig_step, get_zero_index());
        auto loop_op = builder.create<mlir::scf::IfOp>(
            loc,
            while_op.getOperands().getTypes(),
            step_sign,
            get_if_body_builder(true),
            get_if_body_builder(false));

        assert(while_op.getNumResults() >= loop_op.getNumResults());
        builder.updateRootInPlace(while_op, [&]()
        {
            assert(while_op.getNumResults() == before_term.args().size());
            for (auto it : llvm::zip(while_op.getResults(), before_term.args()))
            {
                auto old_res = std::get<0>(it);
                auto operand = std::get<1>(it);
                for (auto it2 : llvm::enumerate(before_block.getArguments()))
                {
                    auto arg = it2.value();
                    if (arg == operand)
                    {
                        assert(it2.index() < loop_op.getNumResults());
                        auto new_res = loop_op.getResult(static_cast<unsigned>(it2.index()));
                        old_res.replaceAllUsesWith(new_res);
                        break;
                    }
                }
                if (pairfirst && operand == pairfirst && !old_res.getUsers().empty())
                {
                    auto val = get_last_iter_value(builder, loc, orig_lower_bound, orig_upper_bound, orig_step);
                    auto new_res = builder.create<plier::CastOp>(loc, old_res.getType(), val);
                    old_res.replaceAllUsesWith(new_res);
                }
                assert(old_res.getUsers().empty());
            }
        });

        assert(while_op.getOperation()->getUsers().empty());
        builder.eraseOp(while_op);
        changed = true;
    }

    if (getiter.getOperation()->getUsers().empty())
    {
        builder.eraseOp(getiter);
        changed = true;
    }
    return mlir::success(changed);
}

// TODO: Copypasted from mlir
namespace
{
using namespace mlir;

/// Verify there are no nested ParallelOps.
static bool hasNestedParallelOp(scf::ParallelOp ploop) {
  auto walkResult =
      ploop.getBody()->walk([](scf::ParallelOp) { return WalkResult::interrupt(); });
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
  return matchOperands(firstPloop.lowerBound(), secondPloop.lowerBound()) &&
         matchOperands(firstPloop.upperBound(), secondPloop.upperBound()) &&
         matchOperands(firstPloop.step(), secondPloop.step());
}

/// Checks if the parallel loops have mixed access to the same buffers. Returns
/// `true` if the first parallel loop writes to the same indices that the second
/// loop reads.
static bool haveNoReadsAfterWriteExceptSameIndex(
    scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
    const BlockAndValueMapping &firstToSecondPloopIndices) {
  DenseMap<Value, SmallVector<ValueRange, 1>> bufferStores;
  firstPloop.getBody()->walk([&](StoreOp store) {
    bufferStores[store.getMemRef()].push_back(store.indices());
  });
  auto walkResult = secondPloop.getBody()->walk([&](LoadOp load) {
    // Stop if the memref is defined in secondPloop body. Careful alias analysis
    // is needed.
    auto *memrefDef = load.getMemRef().getDefiningOp();
    if (memrefDef && memrefDef->getBlock() == load->getBlock())
      return WalkResult::interrupt();

    auto write = bufferStores.find(load.getMemRef());
    if (write == bufferStores.end())
      return WalkResult::advance();

    // Allow only single write access per buffer.
    if (write->second.size() != 1)
      return WalkResult::interrupt();

    // Check that the load indices of secondPloop coincide with store indices of
    // firstPloop for the same memrefs.
    auto storeIndices = write->second.front();
    auto loadIndices = load.indices();
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
                   const BlockAndValueMapping &firstToSecondPloopIndices) {
  if (!haveNoReadsAfterWriteExceptSameIndex(firstPloop, secondPloop,
                                            firstToSecondPloopIndices))
    return failure();

  BlockAndValueMapping secondToFirstPloopIndices;
  secondToFirstPloopIndices.map(secondPloop.getBody()->getArguments(),
                                firstPloop.getBody()->getArguments());
  return success(haveNoReadsAfterWriteExceptSameIndex(
      secondPloop, firstPloop, secondToFirstPloopIndices));
}

static bool
isFusionLegal(scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
              const BlockAndValueMapping &firstToSecondPloopIndices) {
  return !hasNestedParallelOp(firstPloop) &&
         !hasNestedParallelOp(secondPloop) &&
         equalIterationSpaces(firstPloop, secondPloop) &&
         succeeded(verifyDependencies(firstPloop, secondPloop,
                                      firstToSecondPloopIndices));
}

/// Prepends operations of firstPloop's body into secondPloop's body.
static bool fuseIfLegal(scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
                        OpBuilder &b) {
  BlockAndValueMapping firstToSecondPloopIndices;
  firstToSecondPloopIndices.map(firstPloop.getBody()->getArguments(),
                                secondPloop.getBody()->getArguments());

  if (!isFusionLegal(firstPloop, secondPloop, firstToSecondPloopIndices))
    return false;

  b.setInsertionPointToStart(secondPloop.getBody());
  for (auto &op : firstPloop.getBody()->without_terminator())
    b.clone(op, firstToSecondPloopIndices);
  firstPloop.erase();
  return true;
}

bool hasNoEffect(mlir::Operation* op)
{
    if (op->getNumRegions() != 0)
    {
        return false;
    }
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(op))
    {
        return !interface.hasEffect<mlir::MemoryEffects::Read>() &&
               !interface.hasEffect<mlir::MemoryEffects::Write>();
    }
    return !op->hasTrait<::mlir::OpTrait::HasRecursiveSideEffects>();
}
}

mlir::LogicalResult plier::naivelyFuseParallelOps(Region &region) {
  OpBuilder b(region);
  // Consider every single block and attempt to fuse adjacent loops.
  bool changed = false;
  for (auto &block : region) {
    SmallVector<SmallVector<scf::ParallelOp, 8>, 1> ploopChains{{}};
    // Not using `walk()` to traverse only top-level parallel loops and also
    // make sure that there are no side-effecting ops between the parallel
    // loops.
    bool noSideEffects = true;
    for (auto &op : block) {
      if (auto ploop = dyn_cast<scf::ParallelOp>(op)) {
        if (noSideEffects) {
          ploopChains.back().push_back(ploop);
        } else {
          ploopChains.push_back({ploop});
          noSideEffects = true;
        }
        continue;
      }
      // TODO: Handle region side effects properly.
      noSideEffects &= hasNoEffect(&op);
    }
    for (llvm::ArrayRef<scf::ParallelOp> ploops : ploopChains) {
      for (size_t i = 0, e = ploops.size(); i + 1 < e; ++i)
        if (fuseIfLegal(ploops[i], ploops[i + 1], b))
            changed = true;
    }
  }
  return mlir::success(changed);
}
