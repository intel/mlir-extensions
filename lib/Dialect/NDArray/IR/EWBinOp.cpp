//===- EWBinOp.cpp - NDArray dialect  ---------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the EWBinOp of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#include "EWOp.h"

namespace {
/// Pattern to rewrite a ewbin op replacing dynamically shaped inputs
/// by statically shaped inputs if they are defined by an appropriate castop.
class EWBinOpInputCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::EWBinOp> {
public:
  using mlir::OpRewritePattern<::imex::ndarray::EWBinOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWBinOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    if (!llvm::isa<::imex::ndarray::NDArrayType>(op.getResult().getType())) {
      return mlir::failure();
    };

    bool succ = replaceOperandInplaceWithCast(rewriter, 0, op.getLhs(), op) ||
                replaceOperandInplaceWithCast(rewriter, 1, op.getRhs(), op);
    return succ ? ::mlir::success() : ::mlir::failure();
  }
};

/// Pattern to rewrite a ewbin op replacing dynamically shaped result type
/// by statically shaped result type if both inputs are statically shaped.
class EWBinOpResultCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::EWBinOp> {
public:
  using mlir::OpRewritePattern<::imex::ndarray::EWBinOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWBinOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto lhs = op.getLhs();
    auto lPtTyp = lhs.getType().dyn_cast<::imex::ndarray::NDArrayType>();
    auto rhs = op.getRhs();
    auto rPtTyp = rhs.getType().dyn_cast<::imex::ndarray::NDArrayType>();
    auto resPtTyp =
        op.getResult().getType().dyn_cast<::imex::ndarray::NDArrayType>();
    if (!(lPtTyp && rPtTyp && resPtTyp && lPtTyp.hasStaticShape() &&
          rPtTyp.hasStaticShape() && !resPtTyp.hasStaticShape())) {
      return mlir::failure();
    }

    auto outShape = ::imex::broadcast(lPtTyp.getShape(), rPtTyp.getShape());
    auto outTyp = resPtTyp.cloneWith(outShape, resPtTyp.getElementType());

    auto nOp = rewriter.create<::imex::ndarray::EWBinOp>(op->getLoc(), outTyp,
                                                         op.getOp(), lhs, rhs);
    rewriter.replaceOpWithNewOp<::imex::ndarray::CastOp>(op, resPtTyp, nOp);

    return ::mlir::success();
  }
};

} // namespace

void imex::ndarray::EWBinOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<EWBinOpInputCanonicalizer, EWBinOpResultCanonicalizer>(context);
}
