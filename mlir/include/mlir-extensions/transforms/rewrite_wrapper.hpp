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

#pragma once

#include <type_traits>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace plier {

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
} // namespace plier
