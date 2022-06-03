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

#include "mlir-extensions/Transforms/inline_utils.hpp"

#include "mlir-extensions/Dialect/plier/dialect.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/InliningUtils.h>

namespace {
static bool mustInline(mlir::func::CallOp call, mlir::func::FuncOp func) {
  auto attr = mlir::StringAttr::get(call.getContext(),
                                    plier::attributes::getForceInlineName());
  return call->hasAttr(attr) || func->hasAttr(attr);
}

struct ForceInline : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto func = mod.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
    if (!func)
      return mlir::failure();

    if (!mustInline(op, func))
      return mlir::failure();

    auto loc = op.getLoc();
    auto reg =
        rewriter.create<mlir::scf::ExecuteRegionOp>(loc, op.getResultTypes());
    auto newCall = [&]() -> mlir::Operation * {
      auto &regBlock = reg.getRegion().emplaceBlock();
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&regBlock);
      auto call = rewriter.clone(*op);
      rewriter.create<mlir::scf::YieldOp>(loc, call->getResults());
      return call;
    }();

    mlir::InlinerInterface inlinerInterface(op->getContext());
    auto parent = op->getParentOp();
    rewriter.startRootUpdate(parent);
    auto res =
        mlir::inlineCall(inlinerInterface, newCall, func, &func.getRegion());
    if (mlir::succeeded(res)) {
      assert(newCall->getUsers().empty());
      rewriter.eraseOp(newCall);
      rewriter.replaceOp(op, reg.getResults());
      rewriter.finalizeRootUpdate(parent);
    } else {
      rewriter.eraseOp(reg);
      rewriter.cancelRootUpdate(parent);
    }
    return res;
  }
};

struct ForceInlinePass
    : public mlir::PassWrapper<ForceInlinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  virtual mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    mlir::RewritePatternSet p(context);
    p.insert<ForceInline>(context);
    patterns = std::move(p);
    return mlir::success();
  }

  virtual void runOnOperation() override {
    auto mod = getOperation();
    (void)mlir::applyPatternsAndFoldGreedily(mod, patterns);

    mod->walk([&](mlir::func::CallOp call) {
      auto func = mod.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (func && mustInline(call, func)) {
        call.emitError("Couldn't inline force-inline call");
        signalPassFailure();
      }
    });
  }

private:
  mlir::FrozenRewritePatternSet patterns;
};
} // namespace

std::unique_ptr<mlir::Pass> plier::createForceInlinePass() {
  return std::make_unique<ForceInlinePass>();
}
