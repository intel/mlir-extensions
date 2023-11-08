//===- XeTileTranformBase.h -  -------*- C++ -*-===//
//===- XeTileTranformBase.h -  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/////===----------------------------------------------------------------------===//
#ifndef _XeTileTranformBase_H_INCLUDED_
#define _XeTileTranformBase_H_INCLUDED_

#include <llvm/IR/ValueMap.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h"
#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/PassWrapper.h"
#include "imex/Utils/XeCommon.h"

#include "PassDetail.h"

namespace imex {

template <typename SourceOp>
class XeTileConversion : public imex::XeConversionPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using OpPatternRewriter = typename mlir::PatternRewriter;

  XeTileConversion(mlir::MLIRContext *context, XeTypeConverter &typeConverter,
                   mlir::PatternBenefit benefit = 1)
      : XeConversionPattern(typeConverter, SourceOp::getOperationName(),
                            benefit, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override final {
    auto sourceOp = llvm::cast<SourceOp>(op);
    OpAdaptor adaptor(op->getOperands(), sourceOp);
    return matchAndRewrite(sourceOp, adaptor, rewriter);
  }

  virtual mlir::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }
};

} // namespace imex

#endif
