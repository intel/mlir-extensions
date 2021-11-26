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

#include "plier/rewrites/force_inline.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/InliningUtils.h>

#include "plier/dialect.hpp"

mlir::LogicalResult
plier::ForceInline::matchAndRewrite(mlir::CallOp op,
                                    mlir::PatternRewriter &rewriter) const {
  auto attrName = plier::attributes::getForceInlineName();
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(mod);

  auto func = mod.lookupSymbol<mlir::FuncOp>(op.callee());
  if (!func)
    return mlir::failure();

  if (!op->hasAttr(attrName) && !func->hasAttr(attrName))
    return mlir::failure();

  if (!llvm::hasNItems(func.getRegion(), 1))
    return mlir::failure();

  auto loc = op.getLoc();
  auto reg =
      rewriter.create<mlir::scf::ExecuteRegionOp>(loc, op.getResultTypes());
  auto newCall = [&]() -> mlir::Operation * {
    auto &regBlock = reg.region().emplaceBlock();
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
