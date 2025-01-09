//===- LinSpaceOp.cpp - NDArray dialect  ------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the LinSpaceOp of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>

namespace {
/// Pattern to rewrite a create op with constant arguments.
/// Propagates constant shape args to op return type.
class LinSpaceOpConstantArgumentFolder final
    : public mlir::OpRewritePattern<imex::ndarray::LinSpaceOp> {
public:
  using mlir::OpRewritePattern<imex::ndarray::LinSpaceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ndarray::LinSpaceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto num = op.getNum();

    ::mlir::SmallVector<int64_t> staticShape;
    if (auto cval = ::mlir::getConstantIntValue(num); cval) {
      staticShape.emplace_back(cval.value());
    } else {
      return mlir::failure();
    }

    // deduce return value type
    auto oldReturnType = op.getType();
    auto newReturnType =
        oldReturnType.cloneWith(staticShape, oldReturnType.getElementType());
    if (oldReturnType == newReturnType)
      return mlir::failure();

    // emit create op with new return type
    auto newOp = rewriter.create<imex::ndarray::LinSpaceOp>(
        loc, newReturnType, op.getStart(), op.getStop(), num, op.getEndpoint());
    // cast to original type
    rewriter.replaceOpWithNewOp<imex::ndarray::CastOp>(op, oldReturnType,
                                                       newOp);

    return mlir::success();
  }
};
} // namespace

void imex::ndarray::LinSpaceOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<LinSpaceOpConstantArgumentFolder>(context);
}
