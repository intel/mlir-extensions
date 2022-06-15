// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Converting PTensor operations to LinAlg (for now)

#ifndef _PTensorToLinalg_H_INCLUDED_
#define _PTensorToLinalg_H_INCLUDED_

#include <mlir/Dialect/PTensor/IR/PTensorOps.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/PatternMatch.h>

namespace ptensor {

    // Convert PTensor's arange to Linalg
    struct ARangeLowering : public ::mlir::OpConversionPattern<::ptensor::ARangeOp>
    {
        using OpConversionPattern::OpConversionPattern;

        ::mlir::LogicalResult
              matchAndRewrite(::ptensor::ARangeOp op,
                              ::ptensor::ARangeOp::Adaptor adaptor,
                              ::mlir::ConversionPatternRewriter &rewriter) const override;
    };

    // Convert PTensor's elementwise binary operations to Linalg
    struct EWBinOpLowering : public ::mlir::OpConversionPattern<::ptensor::EWBinOp>
    {
        using OpConversionPattern::OpConversionPattern;

        ::mlir::LogicalResult
              matchAndRewrite(::ptensor::EWBinOp op,
                              ::ptensor::EWBinOp::Adaptor adaptor,
                              ::mlir::ConversionPatternRewriter &rewriter) const override;
    };

    // Convert PTensor's reduction operations to Linalg
    struct ReductionOpLowering : public ::mlir::OpConversionPattern<::ptensor::ReductionOp>
    {
        using OpConversionPattern::OpConversionPattern;

        ::mlir::LogicalResult
              matchAndRewrite(::ptensor::ReductionOp op,
                              ::ptensor::ReductionOp::Adaptor adaptor,
                              ::mlir::ConversionPatternRewriter &rewriter) const override;
    };
} // namespace ptensor

#endif // _PTensorToLinalg_H_INCLUDED_
