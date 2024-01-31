//===- RegionConversions.cpp - Region conversion patterns  ------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the type conversion type patterns for region ops.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Region/IR/RegionOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace imex {

namespace region {
namespace {

/// Converts the operand and result types of the EnvironmentRegionOp
struct EnvironmentRegionOpSignatureConversion
    : public ::mlir::OpConversionPattern<::imex::region::EnvironmentRegionOp> {
  using OpConversionPattern<
      ::imex::region::EnvironmentRegionOp>::OpConversionPattern;

  /// Hook for combined matching and rewriting.
  ::mlir::LogicalResult
  matchAndRewrite(EnvironmentRegionOp op, OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // Convert the original results.
    ::mlir::SmallVector<::mlir::Type, 1> convertedResults;
    if (::mlir::failed(
            typeConverter->convertTypes(op.getResultTypes(), convertedResults)))
      return ::mlir::failure();

    // If this isn't a one-to-one type mapping, we don't know how to aggregate
    // the results.
    if (op->getNumResults() != convertedResults.size())
      return ::mlir::failure();

    // Substitute with the new result types from the corresponding conversion.
    auto loc = op.getLoc();
    auto newOp = rewriter.create<::imex::region::EnvironmentRegionOp>(
        loc, adaptor.getEnvironment(), adaptor.getArgs(), convertedResults);

    // Erase block created by builder.
    auto &newRegion = newOp.getRegion();
    rewriter.eraseBlock(&newRegion.front());

    // move old block into new region
    auto &oldRegion = op.getRegion();
    rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
    rewriter.replaceOp(op, newOp.getResults());

    return ::mlir::success();
  }
};

/// This pattern ensures that the branch operation arguments matches up with the
/// successor block arguments.
class EnvironmentRegionYieldOpTypeConversion
    : public ::mlir::OpConversionPattern<EnvironmentRegionYieldOp> {
public:
  using ::mlir::OpConversionPattern<
      EnvironmentRegionYieldOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::region::EnvironmentRegionYieldOp op,
                  OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const final {
    // For a yield, all operands go to the results of the parent, so rewrite
    // them all.
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return ::mlir::success();
  }
};

} // namespace
} // namespace region

void populateRegionTypeConversionPatterns(
    ::mlir::RewritePatternSet &patterns, ::mlir::TypeConverter &typeConverter) {
  patterns.add<region::EnvironmentRegionOpSignatureConversion,
               region::EnvironmentRegionYieldOpTypeConversion>(
      typeConverter, patterns.getContext());
}

} // namespace imex
