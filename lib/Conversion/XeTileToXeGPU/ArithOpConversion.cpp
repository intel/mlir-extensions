//===- ArithOpConversion.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ArithOpConversionPattern, used in XeTileToXeGPU
/// conversion, converting the Arith Ops.
///
//===----------------------------------------------------------------------===//

#include "ArithOpConversion.h"

namespace imex {

class SgArithConstantOpPattern
    : public SgXeTileToXeGPUConversion<mlir::arith::ConstantOp> {
  using SgXeTileToXeGPUConversion<
      mlir::arith::ConstantOp>::SgXeTileToXeGPUConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  XeGPUOneToNPatterRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = op.getResult();
    auto resultTy = result.getType();

    if (!resultTy.isa<mlir::VectorType>())
      return mlir::failure();

    auto vectorTy = resultTy.cast<mlir::VectorType>();

    // We only interesting 4D vectors
    if (vectorTy.getRank() != 4)
      return mlir::failure();

    auto shape = vectorTy.getShape();
    auto subVectorTy = ::mlir::VectorType::get({shape[2], shape[3]},
                                               vectorTy.getElementType());

    auto valueAttr = op.getValue();
    if (!valueAttr.isa<mlir::DenseElementsAttr>())
      return mlir::failure();

    auto denseElementsAttr = valueAttr.cast<mlir::DenseElementsAttr>();
    if (!denseElementsAttr.isSplat())
      return mlir::failure();

    auto splatVal = denseElementsAttr.getSplatValue<mlir::FloatAttr>();

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value> newOps;
    for (auto i = 0; i < shape[0]; i++) {
      for (auto j = 0; j < shape[1]; j++) {
        auto newOp = rewriter.create<mlir::arith::ConstantOp>(
            loc, subVectorTy,
            mlir::DenseElementsAttr::get(subVectorTy, splatVal));
        newOps.push_back(newOp);
      }
    }

    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

bool isLegalArithOp(mlir::Operation *op) {
  if (llvm::isa<mlir::arith::ConstantOp>(op)) {
    auto constOp = llvm::cast<mlir::arith::ConstantOp>(op);
    auto resultTy = constOp.getResult().getType();
    if (resultTy.isa<mlir::VectorType>() &&
        resultTy.cast<mlir::VectorType>().getRank() == 4)
      return false;
  }
  return true;
}

void populateArithOpConversionPatterns(imex::XeGPUTypeConverter &converter,
                                       mlir::RewritePatternSet &patterns) {
  patterns.add<SgArithConstantOpPattern>(patterns.getContext(), converter);
}

} // namespace imex
