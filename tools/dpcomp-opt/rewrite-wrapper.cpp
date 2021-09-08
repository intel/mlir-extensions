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
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <mlir/Dialect/SCF/SCF.h>

#include "plier/pass/rewrite_wrapper.hpp"
#include "plier/rewrites/promote_to_parallel.hpp"

namespace {
template <typename Op, typename Rewrite>
struct RewriteWrapper : plier::RewriteWrapperPass<RewriteWrapper<Op, Rewrite>,
                                                  Op, void, Rewrite> {};

template <typename Op, typename Rewrite>
using WrapperRegistration = mlir::PassRegistration<RewriteWrapper<Op, Rewrite>>;

static WrapperRegistration<mlir::FuncOp, plier::PromoteToParallel>
    promoteToParallelReg("dpcomp-promote-to-parallel", "");
} // namespace
