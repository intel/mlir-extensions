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

#include "plier/rewrites/cast_lowering.hpp"

#include <mlir/Transforms/DialectConversion.h>

plier::CastOpLowering::CastOpLowering(
    mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
    CastOpLowering::cast_t cast_func):
    OpRewritePattern(context), converter(typeConverter),
    cast_func(std::move(cast_func)) {}

mlir::LogicalResult plier::CastOpLowering::matchAndRewrite(
    plier::CastOp op, mlir::PatternRewriter& rewriter) const
{
    auto src = op.getOperand();
    auto src_type = src.getType();
    auto dst_type = converter.convertType(op.getType());
    if (dst_type)
    {
        if (src_type == dst_type)
        {
            rewriter.replaceOp(op, src);
            return mlir::success();
        }
        if (nullptr != cast_func)
        {
            if (auto new_op = cast_func(dst_type, src, rewriter))
            {
                rewriter.replaceOp(op, new_op);
                return mlir::success();
            }
        }
    }
    return mlir::failure();
}
