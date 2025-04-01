//===- CastOp.cpp - NDArray dialect  --------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CastElemTypeOp of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>

/// Pattern to rewrite a CastElemTypeOp replacing dynamically shaped inputs
/// by statically shaped inputs if they are defined by an appropriate CastOp.
class CastElemTypeOpInputCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::CastElemTypeOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::ndarray::CastElemTypeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastElemTypeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    if (!llvm::isa<::mlir::RankedTensorType>(op.getResult().getType())) {
      return mlir::failure();
    };

    auto src = op.getInput();
    auto srcNDTyp = mlir::dyn_cast<::mlir::RankedTensorType>(src.getType());
    auto defOp = src.getDefiningOp<::mlir::tensor::CastOp>();
    if (!srcNDTyp || srcNDTyp.hasStaticShape() || !defOp) {
      return mlir::failure();
    }
    auto defOpSrc = defOp.getSource();
    auto defSrcNDTyp =
        mlir::dyn_cast<::mlir::RankedTensorType>(defOpSrc.getType());
    if (!defSrcNDTyp || !defSrcNDTyp.hasStaticShape()) {
      return mlir::failure();
    }
    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, defOpSrc); });
    return ::mlir::success();
  }
};

/// Pattern to rewrite a CastElemTypeOp replacing dynamically shaped result type
/// by statically shaped result type if input is statically shaped.
class CastElemTypeOpResultCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::CastElemTypeOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::ndarray::CastElemTypeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastElemTypeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto src = op.getInput();
    auto srcNDTyp = mlir::dyn_cast<::mlir::RankedTensorType>(src.getType());
    auto resNDTyp =
        mlir::dyn_cast<::mlir::RankedTensorType>(op.getResult().getType());
    if (!(srcNDTyp && resNDTyp && srcNDTyp.hasStaticShape() &&
          !resNDTyp.hasStaticShape())) {
      return mlir::failure();
    }

    auto resShape = srcNDTyp.getShape();
    auto resTyp = resNDTyp.cloneWith(resShape, resNDTyp.getElementType());
    auto newOp = rewriter.create<::imex::ndarray::CastElemTypeOp>(op->getLoc(),
                                                                  resTyp, src);
    rewriter.replaceOpWithNewOp<::mlir::tensor::CastOp>(op, resNDTyp, newOp);

    return ::mlir::success();
  }
};

void imex::ndarray::CastElemTypeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results
      .add<CastElemTypeOpResultCanonicalizer, CastElemTypeOpInputCanonicalizer>(
          context);
}
