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

#include "plier/rewrites/if_rewrites.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

mlir::LogicalResult
plier::IfOpConstCond::matchAndRewrite(mlir::scf::IfOp op,
                                      mlir::PatternRewriter &rewriter) const {
  auto cond =
      mlir::dyn_cast_or_null<mlir::CmpIOp>(op.condition().getDefiningOp());
  if (!cond) {
    return mlir::failure();
  }
  auto is_const = [](mlir::Value val) {
    if (auto parent = val.getDefiningOp()) {
      return parent->hasTrait<mlir::OpTrait::ConstantLike>();
    }
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

  mlir::Value const_val;
  mlir::Value to_replace;
  if (is_const(cond.lhs())) {
    const_val = cond.lhs();
    to_replace = cond.rhs();
  } else if (is_const(cond.rhs())) {
    const_val = cond.rhs();
    to_replace = cond.lhs();
  } else {
    return mlir::failure();
  }

  if (cond.predicate() == mlir::CmpIPredicate::eq) {
    replace(op.thenRegion().front(), to_replace, const_val);
  } else if (cond.predicate() == mlir::CmpIPredicate::ne) {
    replace(op.elseRegion().front(), to_replace, const_val);
  } else {
    return mlir::failure();
  }

  return mlir::success();
}
