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

#include "plier/rewrites/common_opts.hpp"

#include "plier/rewrites/cse.hpp"
#include "plier/rewrites/force_inline.hpp"
#include "plier/rewrites/if_rewrites.hpp"
#include "plier/rewrites/index_type_propagation.hpp"
#include "plier/rewrites/loop_rewrites.hpp"
#include "plier/rewrites/memory_rewrites.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

void plier::populate_common_opts_patterns(mlir::MLIRContext &context,
                                          mlir::RewritePatternSet &patterns) {
  for (auto *op : context.getRegisteredOperations()) {
    op->getCanonicalizationPatterns(patterns, &context);
  }

  patterns.insert<
      //        LoopInvariantCodeMotion, TODO
      plier::CmpLoopBoundsSimplify, plier::IfOpConstCond,
      plier::CSERewrite<mlir::FuncOp, /*recusive*/ false>>(&context);

  plier::populate_index_propagate_patterns(context, patterns);
}
