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

#include "mlir-extensions/Transforms/if_rewrites.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/SCF/SCF.h>

mlir::LogicalResult
plier::IfOpConstCond::matchAndRewrite(mlir::scf::IfOp op,
                                      mlir::PatternRewriter &rewriter) const {
  auto cond = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
      op.getCondition().getDefiningOp());
  if (!cond)
    return mlir::failure();

  auto isConst = [](mlir::Value val) {
    if (auto parent = val.getDefiningOp())
      return parent->hasTrait<mlir::OpTrait::ConstantLike>();

    return false;
  };

  auto replace = [&](mlir::Block &block, mlir::Value to_replace,
                     mlir::Value new_val) {
    for (auto &use : llvm::make_early_inc_range(to_replace.getUses())) {
      auto owner = use.getOwner();
      if (block.findAncestorOpInBlock(*owner)) {
        rewriter.updateRootInPlace(owner, [&]() { use.set(new_val); });
      }
    }
  };

  auto lhs = cond.getLhs();
  auto rhs = cond.getRhs();
  auto pred = cond.getPredicate();
  mlir::Value constVal;
  mlir::Value toReplace;
  if (isConst(lhs)) {
    constVal = lhs;
    toReplace = rhs;
  } else if (isConst(rhs)) {
    constVal = rhs;
    toReplace = lhs;
  } else {
    return mlir::failure();
  }

  if (pred == mlir::arith::CmpIPredicate::eq) {
    replace(op.getThenRegion().front(), toReplace, constVal);
  } else if (pred == mlir::arith::CmpIPredicate::ne) {
    replace(op.getElseRegion().front(), toReplace, constVal);
  } else {
    return mlir::failure();
  }

  return mlir::success();
}
