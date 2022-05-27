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

#include "mlir-extensions/Transforms/cast_lowering.hpp"

#include <mlir/Transforms/DialectConversion.h>

plier::CastOpLowering::CastOpLowering(mlir::TypeConverter &typeConverter,
                                      mlir::MLIRContext *context,
                                      CastOpLowering::cast_t cast_func)
    : OpRewritePattern(context), converter(typeConverter),
      castFunc(std::move(cast_func)) {}

mlir::LogicalResult
plier::CastOpLowering::matchAndRewrite(plier::CastOp op,
                                       mlir::PatternRewriter &rewriter) const {
  auto src = op.value();
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
