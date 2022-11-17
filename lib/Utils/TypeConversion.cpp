//===- TypeConversion.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines conversion patterns for lowering ControlFlow related ops
/// like func call, branch, return Ops by adding dynamic legality rules and
/// applying them while rewriting.
///
//===----------------------------------------------------------------------===//

#include "imex/Utils/TypeConversion.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Transforms.h>
#include <mlir/Transforms/DialectConversion.h>

// #include "imex/Dialect/imex_util/dialect.hpp"

namespace {
class ConvertSelectOp
    : public mlir::OpConversionPattern<mlir::arith::SelectOp> {
public:
  using mlir::OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::arith::SelectOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};
} // namespace

void imex::populateControlFlowTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);

  // Dynamic rules below accept new function, call, branch, return operations
  // as legal output of the rewriting and then populates and apply rewriting
  // rules
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) -> llvm::Optional<bool> {
        if (typeConverter.isSignatureLegal(op.getFunctionType()) &&
            typeConverter.isLegal(&op.getBody()))
          return true;

        return llvm::None;
      });

  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<mlir::arith::SelectOp, mlir::func::CallOp>(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        if (typeConverter.isLegal(op))
          return true;

        return llvm::None;
      });

  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);

  patterns.insert<ConvertSelectOp>(typeConverter, patterns.getContext());

  target.markUnknownOpDynamicallyLegal(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        if (mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
            mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
                op, typeConverter) ||
            mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter))
          return true;

        return llvm::None;
      });
}
