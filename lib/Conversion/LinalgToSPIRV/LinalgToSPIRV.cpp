//===- LinalgToSPIRV.cpp - LinalgToSPIRV conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the LinalgToSPIRV conversion, converting the Linalg
/// dialect to the SPIRV dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/LinalgToSPIRV/LinalgToSPIRV.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>

#include "../PassDetail.h"

namespace imex {

namespace {
// *******************************
// ***** Individual patterns *****
// *******************************

// MatmulLinalgOp -> SomeSPIRVOp
struct MatmulLinalgOpConverter
    : public ::mlir::OpConversionPattern<::mlir::linalg::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::linalg::MatmulOp op,
                  ::mlir::linalg::MatmulOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    assert(false); // FIXME fill in rewriting code

    return ::mlir::success();
  }
};

// MatmulLinalgOp -> SomeSPIRVOp
struct MatmulLinalgOpRewriter
    : public mlir::OpRewritePattern<::mlir::linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::linalg::MatmulOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    assert(false); // FIXME fill in rewriting code

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass to convert Linalg to SPIRV
struct ConvertLinalgToSPIRVPass
    : public ::imex::ConvertLinalgToSPIRVBase<ConvertLinalgToSPIRVPass> {
  ConvertLinalgToSPIRVPass() = default;

  void runOnOperation() override {
    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<MatmulLinalgOpRewriter>(getContext(), patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

} // namespace

/// Populate the given list with patterns that convert Linalg to SPIRV
void populateLinalgToSPIRVConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns) {
  assert(false); // FIXME
}

/// Create a pass that convert Linalg to SPIRV
std::unique_ptr<::mlir::Pass> createConvertLinalgToSPIRVPass() {
  return std::make_unique<ConvertLinalgToSPIRVPass>();
}

} // namespace imex
