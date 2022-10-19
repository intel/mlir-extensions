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

#include "imex/Transforms/loop_rewrites.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>

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

struct CmpLoopBoundsSimplify
    : public mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp cmp,
                  mlir::PatternRewriter &rewriter) const override {
    auto res = [&]()
        -> llvm::Optional<std::tuple<mlir::Value, mlir::Value, mlir::Value>> {
      for (auto val : {cmp.getLhs(), cmp.getRhs()}) {
        auto blockArg = val.dyn_cast<mlir::BlockArgument>();
        if (!blockArg)
          continue;

        auto block = blockArg.getOwner();
        auto parent = block->getParentOp();
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
          mlir::Value lb = forOp.getLowerBound();
          mlir::Value ub = forOp.getUpperBound();
          return std::make_tuple(lb, ub, val);
        } else if (auto parallelOp =
                       mlir::dyn_cast<mlir::scf::ParallelOp>(parent)) {
          auto idx = blockArg.getArgNumber();
          if (idx < parallelOp.getUpperBound().size()) {
            mlir::Value lb = parallelOp.getLowerBound()[idx];
            mlir::Value ub = parallelOp.getUpperBound()[idx];
            return std::make_tuple(lb, ub, val);
          }
        }
      }
      return llvm::None;
    }();

    if (!res)
      return mlir::failure();

    auto [lowerBound, upperBound, indexVar] = *res;

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

    using fptr_t =
        llvm::Optional<int64_t> (*)(Predicate, mlir::Value, mlir::Value,
                                    mlir::Value, mlir::Value, mlir::Value);
    const fptr_t handlers[] = {
        &handlerImpl<Predicate::sge, UpperBound, 0>,
        &handlerImpl<Predicate::slt, LowerBound, 0>,
        &handlerImpl<Predicate::sge, LowerBound, 1>,
        &handlerImpl<Predicate::slt, UpperBound, 1>,
    };

    for (auto h : handlers) {
      if (auto c = h(pred, lhs, rhs, indexVar, lowerBound, upperBound)) {
        auto type = rewriter.getI1Type();
        auto val = rewriter.getIntegerAttr(type, *c);
        auto constVal =
            rewriter.create<mlir::arith::ConstantOp>(cmp.getLoc(), val);
        rewriter.replaceOp(cmp, constVal.getResult());
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};
} // namespace

void imex::populateLoopRewritesPatterns(mlir::MLIRContext &context,
                                        mlir::RewritePatternSet &patterns) {
  patterns.insert<CmpLoopBoundsSimplify>(&context);
}
