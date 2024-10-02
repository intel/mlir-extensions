//===- PermuteDimsOp.cpp - NDArray dialect  ---------------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the PermuteDimsOp of the NDArray dialect.
/// Copied from NTensor.
///
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

namespace {

bool isPermutation(const ::llvm::ArrayRef<int64_t> &axes) {
  auto sortedAxes = ::mlir::SmallVector<int64_t>(axes.begin(), axes.end());
  std::sort(sortedAxes.begin(), sortedAxes.end());
  for (int64_t i = 0; static_cast<size_t>(i) < axes.size(); ++i) {
    if (sortedAxes[i] != i)
      return false;
  }
  return true;
}

bool isSorted(const ::llvm::ArrayRef<int64_t> &axes) {
  for (int64_t i = 0; static_cast<size_t>(i) < axes.size(); ++i) {
    if (axes[i] != i)
      return false;
  }
  return true;
}

/// Pattern to rewrite a permute_dims op with constant arguments.
/// Propagates constant shape args to op return type.
class PermuteDimsOpConstantArgumentFolder final
    : public mlir::OpRewritePattern<imex::ndarray::PermuteDimsOp> {
public:
  using mlir::OpRewritePattern<imex::ndarray::PermuteDimsOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ndarray::PermuteDimsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto source = op.getSource();
    const auto axes = op.getAxes();
    auto rank = static_cast<int64_t>(axes.size());

    auto sourceType = source.getType();
    if (rank != sourceType.getRank())
      return ::mlir::failure();

    if (isSorted(axes)) {
      rewriter.replaceOpWithNewOp<imex::ndarray::CastOp>(op, op.getType(),
                                                         source);
      return ::mlir::success();
    }

    auto oldReturnType = op.getType();
    if (oldReturnType.hasStaticShape()) {
      return ::mlir::failure();
    }

    const auto &oldShape = sourceType.getShape();
    ::mlir::SmallVector<int64_t> newShape(rank);
    for (int64_t i = 0; i < rank; ++i) {
      newShape[i] = oldShape[axes[i]];
    }

    auto newReturnType =
        sourceType.cloneWith(newShape, sourceType.getElementType());
    if (newReturnType == oldReturnType)
      return ::mlir::failure();

    auto newOp = rewriter.create<imex::ndarray::PermuteDimsOp>(
        op->getLoc(), newReturnType, source, axes);
    rewriter.replaceOpWithNewOp<imex::ndarray::CastOp>(op, oldReturnType,
                                                       newOp);

    return mlir::success();
  }
};
} // namespace

void imex::ndarray::PermuteDimsOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  // TODO:
  //  - convert [0,1,2] & [2,1,0] to no-op
  results.add<PermuteDimsOpConstantArgumentFolder>(context);
}

mlir::LogicalResult imex::ndarray::PermuteDimsOp::verify() {
  const auto axes = getAxes();

  if (!isPermutation(axes)) {
    return mlir::failure();
  }

  auto sourceType = getSource().getType();
  auto returnType = getType();

  if (sourceType.hasStaticShape() && returnType.hasStaticShape()) {
    auto sourceShape = sourceType.getShape();
    auto returnShape = returnType.getShape();
    for (size_t i = 0; i < axes.size(); ++i) {
      if (sourceShape[axes[i]] != returnShape[i]) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}
