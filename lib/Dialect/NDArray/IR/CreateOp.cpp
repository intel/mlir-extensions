//===- CreateOp.cpp - NDArray dialect  --------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CreateOp of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>

namespace {
/// Pattern to rewrite a create op with constant arguments.
/// Propagates constant shape args to op return type.
class CreateOpConstantArgumentFolder final
    : public mlir::OpRewritePattern<imex::ndarray::CreateOp> {
public:
  using mlir::OpRewritePattern<imex::ndarray::CreateOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ndarray::CreateOp createOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = createOp.getLoc();
    mlir::ValueRange shape = createOp.getShape();
    auto rank = shape.size();

    // no shape given
    if (!rank)
      return mlir::failure();

    // infer constants
    ::mlir::SmallVector<int64_t> staticShape(rank,
                                             ::mlir::ShapedType::kDynamic);
    for (size_t i = 0; i < rank; ++i) {
      if (auto cval = ::mlir::getConstantIntValue(shape[i]); cval) {
        staticShape[i] = cval.value();
      }
    }

    // deduce return value type
    auto oldReturnType = createOp.getType();
    auto newReturnType =
        oldReturnType.cloneWith(staticShape, oldReturnType.getElementType());
    // no change
    if (oldReturnType == newReturnType)
      return mlir::failure();

    // emit create op with new return type
    auto newOp = rewriter.create<imex::ndarray::CreateOp>(
        loc, newReturnType, shape, createOp.getValue());
    // cast to original type
    rewriter.replaceOpWithNewOp<::mlir::tensor::CastOp>(createOp, oldReturnType,
                                                        newOp);

    return mlir::success();
  }
};
} // namespace

void imex::ndarray::CreateOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<CreateOpConstantArgumentFolder>(context);
}
