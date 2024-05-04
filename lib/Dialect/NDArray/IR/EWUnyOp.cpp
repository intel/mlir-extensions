//===- EWUnyOp.cpp - NDArray dialect  ---------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the EWUnyOp of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#include "EWOp.h"

namespace {
/// Pattern to rewrite a EWUnyOp replacing dynamically shaped inputs
/// by statically shaped inputs if they are defined by an appropriate castop.
class EWUnyOpInputCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::EWUnyOp> {
public:
  using mlir::OpRewritePattern<::imex::ndarray::EWUnyOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWUnyOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    if (!llvm::isa<::imex::ndarray::NDArrayType>(op.getResult().getType())) {
      return mlir::failure();
    };

    return replaceOperandInplaceWithCast(rewriter, 0, op.getSrc(), op)
               ? ::mlir::success()
               : ::mlir::failure();
  }
};

/// Pattern to rewrite a EWUnyOp replacing dynamically shaped result type
/// by statically shaped result type if input is statically shaped.
class EWUnyOpResultCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::EWUnyOp> {
public:
  using mlir::OpRewritePattern<::imex::ndarray::EWUnyOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWUnyOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto src = op.getSrc();
    auto srcPtTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto resPtTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    if (!(srcPtTyp && resPtTyp && srcPtTyp.hasStaticShape() &&
          !resPtTyp.hasStaticShape())) {
      return mlir::failure();
    }

    auto outShape = srcPtTyp.getShape();
    auto outTyp = resPtTyp.cloneWith(outShape, resPtTyp.getElementType());

    auto nOp = rewriter.create<::imex::ndarray::EWUnyOp>(op->getLoc(), outTyp,
                                                         op.getOp(), src);
    rewriter.replaceOpWithNewOp<::imex::ndarray::CastOp>(op, resPtTyp, nOp);

    return ::mlir::success();
  }
};

} // namespace

void imex::ndarray::EWUnyOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<EWUnyOpInputCanonicalizer, EWUnyOpResultCanonicalizer>(context);
}
