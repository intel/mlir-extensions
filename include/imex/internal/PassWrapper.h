// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _PassWrapper_H_INCLUDED_
#define _PassWrapper_H_INCLUDED_

#include <type_traits>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace imex {

// A list of dialects to be inserted into ::mlir::DialectRegistry
template <typename... Dialects> struct DialectList {
  void insert(::mlir::DialectRegistry &registry) const {
    if constexpr (sizeof...(Dialects) > 0) {
      registry.insert<Dialects...>();
    }
  }
};

#if 0
template <> struct DialectList<> {
  void operator()(::mlir::DialectRegistry & /*registry*/) const {}
};
#endif

// Convenience class: Create simple pass from dependent dialects and a list of
// rewrite patterns CRTP wrapper around a ::mlir::PassWrapper<Fwd,
// ::mlir::OperationPass<Op>>.
template <typename Fwd, typename Op, typename DependentDialects,
          typename... Rewrites>
class PassWrapper : public ::mlir::PassWrapper<Fwd, ::mlir::OperationPass<Op>> {
public:
  virtual void
  getDependentDialects(::mlir::DialectRegistry &registry) const override {
    DependentDialects().insert(registry);
  }

  // Fill pattern set with provided rewrite patterns
  virtual ::mlir::LogicalResult
  initialize(::mlir::MLIRContext *context) override {
    ::mlir::RewritePatternSet p(context);
    p.insert<Rewrites...>(context);
    patterns = std::move(p);
    return ::mlir::success();
  }

  void runOnOperation() override {
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(),
                                               this->patterns);
  }

private:
  ::mlir::FrozenRewritePatternSet patterns;
};

} // namespace imex
#endif // _PassWrapper_H_INCLUDED_
