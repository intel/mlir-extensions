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

#include <mlir/IR/PatternMatch.h>

namespace mlir {
class SelectOp;
namespace scf {
class IfOp;
}
} // namespace mlir

namespace plier {
struct IfOpConstCond : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  IfOpConstCond(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::scf::IfOp>(context, /*benefit*/ 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace plier
