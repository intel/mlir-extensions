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

#include "mlir-extensions/Conversion/SCFToAffine/SCFToAffine.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

using namespace mlir;

class SCFParallelLowering : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    // Check if steps are constants
    SmallVector<int64_t> newSteps;
    for (auto s : op.getStep())
      if (auto c = s.getDefiningOp<arith::ConstantIndexOp>())
        newSteps.push_back(c.value());

    // just for the case if we reductions
    // TODO: fill them from found scf.reduce op
    SmallVector<LoopReduction> reductions;
    auto reducedValueTypes = llvm::to_vector<4>(
        llvm::map_range(reductions, [](const LoopReduction &red) {
          return red.value.getType();
        }));

    auto reductionKinds = llvm::to_vector<4>(llvm::map_range(
        reductions, [](const LoopReduction &red) { return red.kind; }));

    auto dims = static_cast<unsigned>(op.getStep().size());
    // Creating empty affine.parallel op.
    rewriter.setInsertionPoint(op);
    auto affineMap = AffineMap::getMultiDimIdentityMap(dims, op.getContext());
    AffineParallelOp newPloop = rewriter.create<AffineParallelOp>(
        op.getLoc(), reducedValueTypes, reductionKinds,
        llvm::makeArrayRef(affineMap), op.getLowerBound(),
        llvm::makeArrayRef(affineMap), op.getUpperBound(),
        llvm::makeArrayRef(newSteps));

    // Steal the body of the old affine for op.
    newPloop.region().takeBody(op.getRegion());

    Operation *yieldOp = newPloop.getBody()->getTerminator();
    assert(yieldOp);
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<AffineYieldOp>(yieldOp, ValueRange({}));

    assert(newPloop.verifyInvariants().succeeded() &&
           "affine body is incorrectly constructed");

    for (auto &each : llvm::make_early_inc_range(*newPloop.getBody())) {
      if (auto load = dyn_cast<memref::LoadOp>(each)) {
        rewriter.setInsertionPointAfter(load);
        rewriter.replaceOpWithNewOp<AffineLoadOp>(load, load.getMemRef(),
                                                  load.indices());
      } else if (auto store = dyn_cast<memref::StoreOp>(each)) {
        rewriter.setInsertionPointAfter(store);
        rewriter.replaceOpWithNewOp<AffineStoreOp>(
            store, store.getValueToStore(), store.getMemRef(), store.indices());
      }
    }

    // TODO: handle reductions and induction variables

    rewriter.replaceOp(op, newPloop.getResults());
    return success();
  }
};

struct SCFToAffinePass
    : public mlir::PassWrapper<SCFToAffinePass, mlir::OperationPass<void>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    SmallVector<Operation *, 8> parallelOps;
    func->walk([&](Operation *op) -> void {
      if (scf::ParallelOp pOp = mlir::dyn_cast_or_null<scf::ParallelOp>(op)) {
        // Temporary disable if contains induction variables, it's not clear for
        // now what is to do with those inductions
        if (!pOp.getInitVals().empty())
          return;

        // Let's disable ND loops for now
        if (pOp.getStep().size() != 1)
          return;

        for (auto s : pOp.getStep()) {
          if (!s.getDefiningOp<arith::ConstantIndexOp>())
            return;
        }

        if (pOp.getUpperBound().size() != pOp.getLowerBound().size() ||
            pOp.getStep().size() != pOp.getUpperBound().size())
          return;

        // check for supported memory operations
        for (auto &each : pOp.getRegion().getOps())
          if (!MemoryEffectOpInterface::hasNoEffect(&each))
            if (!isa<memref::LoadOp, memref::StoreOp>(&each))
              return;

        // Awoid conversing scf.reduce, scf.if and nested scf.parallel
        // and scf.for
        if (llvm::any_of(pOp.getRegion().getOps(), [&](Operation &each) {
              return 0 != each.getNumRegions();
            }))
          return;

        parallelOps.push_back(op);
      }
    });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<SCFParallelLowering>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    (void)mlir::applyOpPatternsAndFold(parallelOps, frozenPatterns,
                                       /*strict=*/true);
  }
};

} // namespace

/// Uplifts scf operations within a function into affine representation
std::unique_ptr<Pass> mlir::createSCFToAffinePass() {
  return std::make_unique<SCFToAffinePass>();
}
