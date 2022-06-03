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

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/SCF.h>

#include "mlir-extensions/Conversion/SCFToAffine/SCFToAffine.h"
#include "mlir-extensions/Transforms/promote_to_parallel.hpp"
#include "mlir-extensions/Transforms/rewrite_wrapper.hpp"

namespace {
template <typename Op, typename Rewrite>
struct RewriteWrapper : plier::RewriteWrapperPass<RewriteWrapper<Op, Rewrite>,
                                                  Op, void, Rewrite> {};

template <typename T> struct PassWrapper : public T {
  PassWrapper(mlir::StringRef arg, mlir::StringRef desc)
      : argument(arg), description(desc) {}

  mlir::StringRef getArgument() const final { return argument; }
  mlir::StringRef getDescription() const final { return description; }

private:
  mlir::StringRef argument;
  mlir::StringRef description;
};

template <typename Pass>
struct PassRegistrationWrapper
    : public mlir::PassRegistration<PassWrapper<Pass>> {
  PassRegistrationWrapper(mlir::StringRef arg, mlir::StringRef desc)
      : mlir::PassRegistration<PassWrapper<Pass>>([arg, desc]() {
          return std::make_unique<PassWrapper<Pass>>(arg, desc);
        }) {}
};

template <typename Op, typename Rewrite>
using WrapperRegistration =
    PassRegistrationWrapper<RewriteWrapper<Op, Rewrite>>;

static WrapperRegistration<mlir::func::FuncOp, plier::PromoteToParallel>
    promoteToParallelReg("dpcomp-promote-to-parallel", "");

static mlir::PassPipelineRegistration<> scfToAffineReg(
    "scf-to-affine", "Converts SCF parallel struct into Affine parallel",
    [](mlir::OpPassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFToAffinePass());
    });

} // namespace
