//===- CopyPermuteOp.cpp - distruntime dialect  -----------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CopyPermuteOp of the DistRuntime dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>

namespace imex {
namespace distruntime {

::mlir::SmallVector<::mlir::Value> CopyPermuteOp::getDependent() {
  return {getNlArray()};
}

} // namespace distruntime
} // namespace imex

namespace {

/// Pattern to replace dynamically shaped result types
/// by statically shaped result types.
class CopyPermuteOpResultCanonicalizer final
    : public mlir::OpRewritePattern<::imex::distruntime::CopyPermuteOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::distruntime::CopyPermuteOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::CopyPermuteOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    // check input type
    auto dstArray = op.getNlArray();
    auto dstType = mlir::dyn_cast<::mlir::RankedTensorType>(dstArray.getType());
    if (!dstType) {
      return ::mlir::failure();
    }
    auto dstShape = dstType.getShape();
    if (!::mlir::ShapedType::isDynamicShape(dstShape)) {
      return ::mlir::failure();
    }

    auto dstLShape = ::imex::getShapeFromValues(op.getNlShape());
    auto elType = dstType.getElementType();
    auto nType = dstType.cloneWith(dstLShape, elType);
    auto hType = ::imex::distruntime::AsyncHandleType::get(getContext());

    auto newOp = ::imex::distruntime::CopyPermuteOp::create(
        rewriter, op.getLoc(), ::mlir::TypeRange{hType, nType},
        op.getTeamAttr(), op.getLArray(), op.getGShape(), op.getLOffsets(),
        op.getNlOffsets(), op.getNlShape(), op.getAxes());

    // cast to original types and replace op
    auto res = ::mlir::tensor::CastOp::create(rewriter, op.getLoc(), dstType,
                                              newOp.getNlArray());
    rewriter.replaceOp(op, {newOp.getHandle(), res});

    return ::mlir::success();
  }
};

/// Pattern to rewrite a subview op with CastOp arguments.
/// Ported from mlir::tensor
class CopyPermuteCastFolder final
    : public mlir::OpRewritePattern<::imex::distruntime::CopyPermuteOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::distruntime::CopyPermuteOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::CopyPermuteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getLArray();
    auto castOp = mlir::dyn_cast<::mlir::tensor::CastOp>(src.getDefiningOp());
    if (!castOp)
      return mlir::failure();

    if (!mlir::tensor::canFoldIntoConsumerOp(castOp))
      return mlir::failure();

    auto newOp = ::imex::distruntime::CopyPermuteOp::create(
        rewriter, op.getLoc(), op->getResultTypes(), op.getTeamAttr(),
        castOp.getSource(), op.getGShape(), op.getLOffsets(), op.getNlOffsets(),
        op.getNlShape(), op.getAxes());
    rewriter.replaceOp(op, newOp);

    return mlir::success();
  }
};

} // namespace

void imex::distruntime::CopyPermuteOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<CopyPermuteOpResultCanonicalizer, CopyPermuteCastFolder>(context);
}
