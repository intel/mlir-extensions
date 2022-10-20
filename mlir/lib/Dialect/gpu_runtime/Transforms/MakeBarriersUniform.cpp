// Copyright 2022 Intel Corporation
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

#include "imex/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.hpp"

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
struct ConvertBarrierOp
    : public mlir::OpRewritePattern<gpu_runtime::GPUBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUBarrierOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto launchOp = op->getParentOfType<mlir::gpu::LaunchOp>();

    // Must be within launch op.
    if (!launchOp)
      return mlir::failure();

    // If op must be an immediate parent
    auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op->getParentOp());
    if (!ifOp)
      return mlir::failure();

    // Launch op must be an immediate parent of ifOp.
    if (ifOp->getParentOp() != launchOp)
      return mlir::failure();

    // IfOp with else block is not yet supported;
    if (ifOp.elseBlock())
      return mlir::failure();

    mlir::Block *ifBody = ifOp.thenBlock();
    assert(ifBody);

    mlir::DominanceInfo dom;
    llvm::SmallMapVector<mlir::Value, unsigned, 8> yieldArgsMap;

    auto barrierIt = op->getIterator();
    for (auto &beforeOp : llvm::make_range(ifBody->begin(), barrierIt)) {
      for (auto result : beforeOp.getResults()) {
        for (mlir::OpOperand user : result.getUsers()) {
          auto owner = user.getOwner();
          if (dom.properlyDominates(op, owner)) {
            auto idx = static_cast<unsigned>(yieldArgsMap.size());
            yieldArgsMap.insert({result, idx});
          }
        }
      }
    }

    auto yieldArgs = [&]() {
      auto range = llvm::make_first_range(yieldArgsMap);
      return llvm::SmallVector<mlir::Value>(std::begin(range), std::end(range));
    }();

    auto afterBlock = rewriter.splitBlock(ifBody, std::next(barrierIt));
    auto beforeBlock = rewriter.splitBlock(ifBody, ifBody->begin());

    auto barrierLoc = op->getLoc();
    auto barrierFlags = op.getFlags();
    rewriter.eraseOp(op);

    rewriter.setInsertionPointToEnd(beforeBlock);
    rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc(), yieldArgs);

    rewriter.setInsertionPoint(ifOp);
    auto ifLoc = ifOp->getLoc();
    auto cond = ifOp.getCondition();

    auto thenBodyBuilder = [&](mlir::OpBuilder & /*builder*/,
                               mlir::Location /*loc*/) {
      // Nothing
    };

    auto elseBodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      llvm::SmallVector<mlir::Value> results;
      results.reserve(yieldArgs.size());
      for (auto arg : yieldArgs) {
        auto val = builder.create<imex::util::UndefOp>(loc, arg.getType());
        results.emplace_back(val);
      }

      builder.create<mlir::scf::YieldOp>(loc, results);
    };

    mlir::ValueRange yieldArgsRange(yieldArgs);
    auto beforeIf =
        rewriter.create<mlir::scf::IfOp>(ifLoc, yieldArgsRange.getTypes(), cond,
                                         thenBodyBuilder, elseBodyBuilder);
    rewriter.mergeBlocks(beforeBlock, beforeIf.thenBlock());

    rewriter.create<gpu_runtime::GPUBarrierOp>(barrierLoc, barrierFlags);

    auto beforeIfResults = beforeIf.getResults();

    auto afterIf =
        rewriter.create<mlir::scf::IfOp>(ifLoc, cond, thenBodyBuilder);
    rewriter.mergeBlocks(afterBlock, afterIf.thenBlock());

    for (auto &op : *afterIf.thenBlock()) {
      for (mlir::OpOperand &arg : op.getOpOperands()) {
        auto val = arg.get();
        auto it = yieldArgsMap.find(val);
        if (it != yieldArgsMap.end()) {
          auto i = it->second;
          assert(i < beforeIfResults.size() && "Invalid result index.");
          auto newVal = beforeIfResults[i];
          rewriter.updateRootInPlace(&op, [&]() { arg.set(newVal); });
        }
      }
    }

    rewriter.eraseOp(ifOp);
    return mlir::success();
  }
};
} // namespace

void gpu_runtime::populateMakeBarriersUniformPatterns(
    mlir::MLIRContext &context, mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertBarrierOp>(&context);
}

namespace {
struct MakeBarriersUniformPass
    : public mlir::PassWrapper<MakeBarriersUniformPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MakeBarriersUniformPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    gpu_runtime::populateMakeBarriersUniformPatterns(ctx, patterns);

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true; // We need to visit top barriers first
    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns), config);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> gpu_runtime::createMakeBarriersUniformPass() {
  return std::make_unique<MakeBarriersUniformPass>();
}
