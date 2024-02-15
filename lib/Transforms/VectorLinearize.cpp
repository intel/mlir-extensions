//===- VectorLinearize.cpp - VectorLinearize Pass  --------------*- C++- *-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains VectorLinearize pass.
///
//===----------------------------------------------------------------------===//

#include <imex/Transforms/Passes.h>

#include <mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace imex {
#define GEN_PASS_DEF_VECTORLINEARIZE
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {
struct VectorLinearizePass final
    : public imex::impl::VectorLinearizeBase<VectorLinearizePass> {

  void runOnOperation() override {
    auto *context = &getContext();

    mlir::TypeConverter typeConverter;
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    mlir::vector::populateVectorLinearizeTypeConversionsAndLegality(
        typeConverter, patterns, target);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createVectorLinearizePass() {
  return std::make_unique<VectorLinearizePass>();
}
