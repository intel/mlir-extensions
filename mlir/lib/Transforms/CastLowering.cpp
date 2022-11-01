// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CastLowering.hpp"

#include <mlir/Transforms/DialectConversion.h>

imex::CastOpLowering::CastOpLowering(mlir::TypeConverter &typeConverter,
                                     mlir::MLIRContext *context,
                                     CastOpLowering::cast_t cast_func)
    : OpRewritePattern(context), converter(typeConverter),
      castFunc(std::move(cast_func)) {}

mlir::LogicalResult
imex::CastOpLowering::matchAndRewrite(plier::CastOp op,
                                      mlir::PatternRewriter &rewriter) const {
  auto src = op.getValue();
  auto srcType = src.getType();
  auto dstType = converter.convertType(op.getType());
  if (dstType) {
    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }
    if (nullptr != castFunc) {
      auto loc = op.getLoc();
      if (auto newOp = castFunc(rewriter, loc, src, dstType)) {
        rewriter.replaceOp(op, newOp);
        return mlir::success();
      }
    }
  }
  return mlir::failure();
}
