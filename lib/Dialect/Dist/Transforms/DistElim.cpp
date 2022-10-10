//===- DistElim.h - PTensorToLinalg conversion  ---------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements elimination of the Dist dialects, leading to local-only
/// operation
///
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/internal/PassWrapper.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

namespace imex {

namespace {

// dummy: constant op
template <typename Op>
static void _toConst(Op op, mlir::PatternRewriter &rewriter, int64_t v = 0) {
  auto attr = rewriter.getIndexAttr(v);
  rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attr);
}

// *******************************
// ***** Individual patterns *****
// *******************************

// RegisterPTensorOp -> no-op
struct ElimRegisterPTensorOp
    : public mlir::OpRewritePattern<::imex::dist::RegisterPTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RegisterPTensorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    _toConst(op, rewriter);
    return ::mlir::success();
  }
};

// LocalOffsetsOp -> const(0)
struct ElimLocalOffsetsOp
    : public mlir::OpRewritePattern<::imex::dist::LocalOffsetsOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalOffsetsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    _toConst(op, rewriter);
    return ::mlir::success();
  }
};

// LocalShapeOp -> global shape
struct ElimLocalShapeOp
    : public mlir::OpRewritePattern<::imex::dist::LocalShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto x = op.getPtensor().getDefiningOp<::imex::dist::RegisterPTensorOp>();
    assert(x);
    rewriter.replaceOp(op, x.getShape());
    return ::mlir::success();
  }
};

// AllReduceOp -> identity cast
struct ElimAllReduceOp
    : public mlir::OpRewritePattern<::imex::dist::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::AllReduceOp op,
                  mlir::PatternRewriter &rewriter) const override {
#if 0
    auto loc = op.getLoc();

    ::mlir::ModuleOp module = op->getParentOfType<::mlir::ModuleOp>();
    auto *context = module.getContext();
    constexpr auto _f = "printf";
    if(!module.lookupSymbol<::mlir::func::FuncOp>(_f)) {
        // auto st = rewriter.getStringAttr("dummy");
        // auto fn = ::mlir::FlatSymbolRefAttr::get(st);
        auto fn = ::llvm::StringRef(_f);
        auto ft = rewriter.getFunctionType({}, {});
        // Insert the printf function into the body of the parent module.
        ::mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<::mlir::func::FuncOp>(loc, _f, ft);
    }
    auto fa = ::mlir::SymbolRefAttr::get(context, _f);
    auto fc = rewriter.create<::mlir::func::CallOp>(loc, fa, ::mlir::TypeRange{});
#endif
    rewriter.replaceOpWithNewOp<::mlir::tensor::CastOp>(
        op, op.getTensor().getType(), op.getTensor());
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct DistElimPass : public ::imex::DistElimBase<DistElimPass> {

  DistElimPass() = default;

  void runOnOperation() override {
    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<ElimRegisterPTensorOp, ElimLocalOffsetsOp, ElimLocalShapeOp,
                   ElimAllReduceOp>(getContext(), patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);
  }
};

} // namespace

/// Populate the given list with patterns that eliminate Dist ops
void populateDistElimPatterns(::mlir::LLVMTypeConverter &converter,
                              ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createDistElimPass() {
  return std::make_unique<DistElimPass>();
}

} // namespace imex
