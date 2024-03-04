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
    auto value =
        llvm::dyn_cast_if_present<mlir::DenseElementsAttr>(op.getValue());

    // We only interesting 4D vectors
    if (!value || value.getType().getRank() != 4)
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute> elems(
        value.value_begin<mlir::Attribute>(),
        value.value_end<mlir::Attribute>());

    auto shape = value.getType().getShape();
    auto vecTy =
        mlir::VectorType::get({shape[2], shape[3]}, value.getElementType());

    // slice a block of (shape[2], shape[3]) from elems.
    auto slice = [&](int i, int j) {
      llvm::SmallVector<mlir::Attribute> block;
      auto width = shape[1] * shape[3];
      i = i * shape[2];
      j = j * shape[3];
      for (int64_t r = 0; r < shape[2]; r++)
        for (int64_t c = 0; c < shape[3]; c++)
          block.push_back(elems[(i + r) * width + j + c]);
      return block;
    };

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value> newOps;
    for (auto i = 0; i < shape[0]; i++) {
      for (auto j = 0; j < shape[1]; j++) {
        auto values = slice(i, j);
        auto attr = mlir::DenseElementsAttr::get(vecTy, values);
        auto newOp = rewriter.create<mlir::arith::ConstantOp>(loc, attr);
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
