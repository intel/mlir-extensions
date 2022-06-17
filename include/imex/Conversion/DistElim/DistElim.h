// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Eliminating dist calls (falling back to local compute).

#ifndef _DistElim_H_INCLUDED_
#define _DistElim_H_INCLUDED_

#include <imex/Dialect/Dist/IR/DistOps.h>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace dist {

// RegisterPTensorOp -> no-op
struct ElimRegisterPTensorOp
    : public mlir::OpRewritePattern<::dist::RegisterPTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::dist::RegisterPTensorOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

// LocalOffsetsOp -> const(0)
struct ElimLocalOffsetsOp
    : public mlir::OpRewritePattern<::dist::LocalOffsetsOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::dist::LocalOffsetsOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

// LocalShapeOp -> global shape
struct ElimLocalShapeOp : public mlir::OpRewritePattern<::dist::LocalShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::dist::LocalShapeOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

// AllReduceOp -> identity cast
struct ElimAllReduceOp : public mlir::OpRewritePattern<::dist::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::dist::AllReduceOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace dist

#endif // _DistElim_H_INCLUDED_
