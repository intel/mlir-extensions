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

#include "mlir-extensions/Transforms/loop_rewrites.hpp"
#include "mlir-extensions/Transforms/const_utils.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/SCF/SCF.h>

namespace {
template <mlir::arith::CmpIPredicate SrcPred,
          mlir::arith::CmpIPredicate DstPred>
static bool normImpl2(mlir::arith::CmpIPredicate &pred, mlir::Value index,
                      mlir::Value &lhs, mlir::Value &rhs) {
  if (pred != SrcPred)
    return false;

  if (index != lhs) {
    std::swap(lhs, rhs);
    pred = DstPred;
  }
  return true;
}

template <mlir::arith::CmpIPredicate SrcPred,
          mlir::arith::CmpIPredicate DstPred>
static bool normImpl(mlir::arith::CmpIPredicate &pred, mlir::Value index,
                     mlir::Value &lhs, mlir::Value &rhs) {
  return normImpl2<SrcPred, DstPred>(pred, index, lhs, rhs) ||
         normImpl2<DstPred, SrcPred>(pred, index, lhs, rhs);
}

enum EBound {
  LowerBound,
  UpperBound,
};
template <mlir::arith::CmpIPredicate Pred, EBound Bound, int64_t Value>
static llvm::Optional<int64_t>
handlerImpl(mlir::arith::CmpIPredicate pred, mlir::Value lhs, mlir::Value rhs,
            mlir::Value index, mlir::Value lowerBound, mlir::Value upperBound) {
  if (pred != Pred)
    return {};

  auto bound = (Bound == LowerBound ? lowerBound : upperBound);
  if (rhs == bound && lhs == index)
    return Value;

  return {};
}
} // namespace

mlir::LogicalResult plier::CmpLoopBoundsSimplify::matchAndRewrite(
    mlir::scf::ForOp op, mlir::PatternRewriter &rewriter) const {
  auto indexVar = op.getLoopBody().front().getArgument(0);
  bool matched = false;
  for (auto user : llvm::make_early_inc_range(indexVar.getUsers())) {
    auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(user);
    if (cmp) {
      auto pred = cmp.getPredicate();
      auto lhs = cmp.getLhs();
      auto rhs = cmp.getRhs();
      // Normalize index and predicate (index always on the left)
      using norm_fptr_t =
          bool (*)(mlir::arith::CmpIPredicate & pred, mlir::Value index,
                   mlir::Value & lhs, mlir::Value & rhs);
      using Predicate = mlir::arith::CmpIPredicate;
      const norm_fptr_t norm_handlers[] = {
          &normImpl<Predicate::sle, Predicate::sge>,
          &normImpl<Predicate::slt, Predicate::sgt>,
          &normImpl<Predicate::ule, Predicate::uge>,
          &normImpl<Predicate::ult, Predicate::ugt>,
          &normImpl<Predicate::eq, Predicate::eq>,
          &normImpl<Predicate::ne, Predicate::ne>,
      };

      for (auto h : norm_handlers)
        if (h(pred, indexVar, lhs, rhs))
          break;

      using fptr_t = llvm::Optional<int64_t> (*)(
          Predicate pred, mlir::Value lhs, mlir::Value rhs, mlir::Value index,
          mlir::Value lowerBound, mlir::Value upperBound);
      const fptr_t handlers[] = {
          &handlerImpl<Predicate::sge, UpperBound, 0>,
          &handlerImpl<Predicate::slt, LowerBound, 0>,
          &handlerImpl<Predicate::sge, LowerBound, 1>,
          &handlerImpl<Predicate::slt, UpperBound, 1>,
      };

      for (auto h : handlers) {
        if (auto c = h(pred, lhs, rhs, indexVar, op.getLowerBound(),
                       op.getUpperBound())) {
          auto type = rewriter.getI1Type();
          auto val = rewriter.getIntegerAttr(type, *c);
          auto constVal =
              rewriter.create<mlir::arith::ConstantOp>(cmp.getLoc(), val);
          rewriter.replaceOp(cmp, constVal.getResult());
          matched = true;
          break;
        }
      }
    }
  }
  return mlir::success(matched);
}
