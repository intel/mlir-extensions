//===- CopyReshapeOp.cpp - distruntime dialect  -----------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CopyReshapeOp of the DistRuntime dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>

namespace imex {
namespace distruntime {

::mlir::SmallVector<::mlir::Value> CopyReshapeOp::getDependent() {
  return {getNlArray()};
}

} // namespace distruntime
} // namespace imex

namespace {

/// Pattern to replace dynamically shaped result types
/// by statically shaped result types.
class CopyReshapeOpResultCanonicalizer final
    : public mlir::OpRewritePattern<::imex::distruntime::CopyReshapeOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::distruntime::CopyReshapeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::CopyReshapeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    // check input type
    auto nlArray = op.getNlArray();
    auto nlType = mlir::dyn_cast<::mlir::RankedTensorType>(nlArray.getType());
    if (!nlType) {
      return ::mlir::failure();
    }
    auto resShape = nlType.getShape();
    if (!::mlir::ShapedType::isDynamicShape(resShape)) {
      return ::mlir::failure();
    }

    // we compare result shape with expected shape and bail out if no new
    // static info found
    auto nlShape = ::imex::getShapeFromValues(op.getNlShape());
    bool found = false;
    for (auto i = 0u; i < nlShape.size(); ++i) {
      if (::mlir::ShapedType::isDynamic(resShape[i]) &&
          !::mlir::ShapedType::isDynamic(nlShape[i])) {
        found = true;
      }
    }
    if (!found) {
      return ::mlir::failure();
    }

    auto elType = nlType.getElementType();
    auto nType = nlType.cloneWith(nlShape, elType);
    auto hType = ::imex::distruntime::AsyncHandleType::get(getContext());

    auto newOp = rewriter.create<::imex::distruntime::CopyReshapeOp>(
        op.getLoc(), ::mlir::TypeRange{hType, nType}, op.getTeamAttr(),
        op.getLArray(), op.getGShape(), op.getLOffsets(), op.getNgShape(),
        op.getNlOffsets(), op.getNlShape());

    // cast to original types and replace op
    auto res = rewriter.create<mlir::tensor::CastOp>(op.getLoc(), nlType,
                                                     newOp.getNlArray());
    rewriter.replaceOp(op, {newOp.getHandle(), res});

    return ::mlir::success();
  }
};

} // namespace

void imex::distruntime::CopyReshapeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<CopyReshapeOpResultCanonicalizer>(context);
}
