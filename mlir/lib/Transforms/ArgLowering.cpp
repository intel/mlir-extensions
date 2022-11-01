// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/ArgLowering.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "imex/Dialect/plier/Dialect.hpp"

imex::ArgOpLowering::ArgOpLowering(mlir::MLIRContext *context)
    : OpRewritePattern(context) {}

mlir::LogicalResult
imex::ArgOpLowering::matchAndRewrite(plier::ArgOp op,
                                     mlir::PatternRewriter &rewriter) const {
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return mlir::failure();

  auto index = op.getIndex();
  if (index >= func.getNumArguments())
    return mlir::failure();

  mlir::Value arg = func.getArgument(index);
  auto opType = op.getType();
  if (opType != arg.getType())
    arg = rewriter.create<plier::CastOp>(op.getLoc(), opType, arg);

  rewriter.replaceOp(op, arg);
  return mlir::success();
}
