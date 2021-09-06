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

#include "plier/Conversion/SCFToAffine/SCFToAffine.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
    // Temporary disable if contains induction variables, it's not clear for now
    // what is to do with those inductions
    if (!op.initVals().empty())
      return rewriter.notifyMatchFailure(
          op, "scf.parallel constains indcution variables");

    // Let's disable ND loops for now
    if (op.step().size() != 1)
      return rewriter.notifyMatchFailure(op, "scf.parallel ND range");

    if (op.upperBound().size() != op.lowerBound().size() ||
        op.step().size() != op.upperBound().size())
      return rewriter.notifyMatchFailure(
          op, "scf.parallel inconsistend upper/lower bounds and steps");

    // Check if steps are constants
    SmallVector<int64_t> newSteps;
    for (auto s : op.step()) {
      if (auto c = s.getDefiningOp<ConstantIndexOp>()) {
        newSteps.push_back(c.getValue());
      } else {
        return rewriter.notifyMatchFailure(
            op, "scf.parallel->affine.parallel non constant step");
      }
    }

    // Awoid conversing scf.reduce, scf.if and nested scf.parallel
    // and scf.for as well as non affine memory references
    if (llvm::any_of(op.region().getOps(), [&](Operation &each) {
          return !MemoryEffectOpInterface::hasNoEffect(&each) ||
                 0 != each.getNumRegions();
        })) {
      return rewriter.notifyMatchFailure(
          op, "scf.parallel->affine.parallel reduction is detected");
    }

    // just for the case if we reductions
    // TODO: fill them from found scf.reduce op
    SmallVector<LoopReduction> reductions;
    auto reducedValueTypes = llvm::to_vector<4>(
        llvm::map_range(reductions, [](const LoopReduction &red) {
          return red.value.getType();
        }));

    auto reductionKinds = llvm::to_vector<4>(llvm::map_range(
        reductions, [](const LoopReduction &red) { return red.kind; }));

    auto dims = op.step().size();
    // Creating empty affine.parallel op.
    rewriter.setInsertionPoint(op);
    AffineParallelOp newPloop = rewriter.create<AffineParallelOp>(
        op.getLoc(), reducedValueTypes, reductionKinds,
        llvm::makeArrayRef(
            AffineMap::getMultiDimIdentityMap(dims, op.getContext())),
        op.lowerBound(),
        llvm::makeArrayRef(
            AffineMap::getMultiDimIdentityMap(dims, op.getContext())),
        op.upperBound(), llvm::makeArrayRef(newSteps));

    // Steal the body of the old affine for op.
    newPloop.region().takeBody(op.region());

    Operation *yieldOp = &newPloop.getBody()->back();
    rewriter.setInsertionPoint(&newPloop.getBody()->back());
    rewriter.replaceOpWithNewOp<AffineYieldOp>(yieldOp, ValueRange({}));

    assert(newPloop.verify().succeeded() &&
           "affine body is incorrectly constructed");

    // TODO: handle reductions and induction variables

    rewriter.replaceOp(op, newPloop.getResults());
    return success();
  }
};

struct SCFToAffinePass
    : public mlir::PassWrapper<SCFToAffinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::AffineDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::OwningRewritePatternList patterns(&getContext());
    patterns.insert<SCFParallelLowering>(&getContext());
    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

} // namespace

/// Uplifts scf operations within a function into affine representation
std::unique_ptr<Pass> mlir::createSCFToAffinePass() {
  return std::make_unique<SCFToAffinePass>();
}
