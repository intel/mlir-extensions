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

#include "plier/rewrites/arg_lowering.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Transforms/DialectConversion.h>

#include "plier/dialect.hpp"

plier::ArgOpLowering::ArgOpLowering(mlir::MLIRContext *context)
    : OpRewritePattern(context) {}

mlir::LogicalResult
plier::ArgOpLowering::matchAndRewrite(plier::ArgOp op,
                                      mlir::PatternRewriter &rewriter) const {
  auto func = op->getParentOfType<mlir::FuncOp>();
  if (!func)
    return mlir::failure();

  auto index = op.index();
  if (index >= func.getNumArguments())
    return mlir::failure();

  mlir::Value arg = func.getArgument(index);
  auto opType = op.getType();
  if (opType != arg.getType())
    arg = rewriter.create<plier::CastOp>(op.getLoc(), opType, arg);

  rewriter.replaceOp(op, arg);
  return mlir::success();
}
