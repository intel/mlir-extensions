// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <type_traits>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace imex {

template <typename... Dialects> struct DependentDialectsList {
  void operator()(mlir::DialectRegistry &registry) const {
    registry.insert<Dialects...>();
  }
};

template <> struct DependentDialectsList<> {
  void operator()(mlir::DialectRegistry & /*registry*/) const {}
};

template <typename Fwd, typename Op, typename DependentDialects,
          typename... Rewrites>
class RewriteWrapperPass
    : public mlir::PassWrapper<Fwd, mlir::OperationPass<Op>> {
public:
  // TODO: Cannot use MLIR_DECLARE_EXPLICIT_TYPE_ID
  static ::mlir::TypeID resolveTypeID() {
    static ::mlir::SelfOwningTypeID id;
    return id;
  }

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    using Reg = std::conditional_t<std::is_void<DependentDialects>::value,
                                   DependentDialectsList<>, DependentDialects>;
    Reg()(registry);
  }

  virtual mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    mlir::RewritePatternSet p(context);
    p.insert<Rewrites...>(context);
    patterns = std::move(p);
    return mlir::success();
  }

  void runOnOperation() override {
    (void)mlir::applyPatternsAndFoldGreedily(this->getOperation(),
                                             this->patterns);
  }

private:
  mlir::FrozenRewritePatternSet patterns;
};
} // namespace imex
