//===- DistToStandard.cpp - DistToStandard conversion  ----------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DistToStandard conversion, converting the Dist
/// dialect to standard dialects, mostly by creating runtime calls.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/DistToStandard/DistToStandard.h>
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/internal/PassUtils.h>
#include <imex/internal/PassWrapper.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>

#include <iostream>

#include "../PassDetail.h"

namespace imex {
namespace {

// create call to given function with 3 args: guid, new rank-shaped 1d tensor
// and rank
inline ::mlir::Value
createCallGetRankedData(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        const char *func, ::mlir::Value guid, uint64_t rank) {
  auto rankV = createIndex(loc, builder, rank);
  auto tnsr = builder.create<::mlir::tensor::EmptyOp>(
      loc, ::mlir::ArrayRef<::mlir::OpFoldResult>({rankV}),
      builder.getI64Type());
  auto fsa = builder.getStringAttr(func);
  (void)builder.create<::mlir::func::CallOp>(
      loc, fsa, ::mlir::TypeRange(),
      ::mlir::ValueRange({guid, tnsr, createInt(loc, builder, rank)}));
  return tnsr;
}

// create function prototype fo given function name, arg-types and
// return-types
inline void requireFunc(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        ::mlir::ModuleOp module, const char *fname,
                        ::mlir::TypeRange args, ::mlir::TypeRange results) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Insert before module terminator.
  builder.setInsertionPoint(module.getBody(),
                            std::prev(module.getBody()->end()));
  auto funcType = builder.getFunctionType(args, results);
  auto func = builder.create<::mlir::func::FuncOp>(loc, fname, funcType);
  func.setPrivate();
}

// *******************************
// ***** Individual patterns *****
// *******************************

// RuntimePrototypesOp -> func.func ops
struct RuntimePrototypesOpConverter
    : public mlir::OpRewritePattern<::imex::dist::RuntimePrototypesOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RuntimePrototypesOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto mod = op->getParentOp();
    assert(::mlir::isa<mlir::ModuleOp>(mod));
    ::mlir::ModuleOp module = ::mlir::cast<mlir::ModuleOp>(mod);
    auto dtype = rewriter.getI64Type();
    auto i64Type = rewriter.getI64Type();
    auto dtypeType = rewriter.getIntegerType(sizeof(int) * 8);
    auto opType =
        rewriter.getIntegerType(sizeof(::imex::ptensor::ReduceOpId) * 8);
    requireFunc(loc, rewriter, module, "_idtr_init_dtensor",
                {::mlir::RankedTensorType::get({-1}, dtype), i64Type},
                {i64Type});
    requireFunc(loc, rewriter, module, "_idtr_local_shape",
                {i64Type, ::mlir::RankedTensorType::get({-1}, dtype), i64Type},
                {});
    requireFunc(loc, rewriter, module, "_idtr_local_offsets",
                {i64Type, ::mlir::RankedTensorType::get({-1}, dtype), i64Type},
                {});
    requireFunc(loc, rewriter, module, "_idtr_reduce_all",
                {::mlir::RankedTensorType::get({}, dtype), dtypeType, opType},
                {});
    rewriter.eraseOp(op);
    return ::mlir::success();
  }
};

// RegisterPTensorOp -> call into _idtr_init_dtensor
struct RegisterPTensorOpConverter
    : public mlir::OpRewritePattern<::imex::dist::RegisterPTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RegisterPTensorOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // get RankedTensor and rank and call registration function
    auto loc = op.getLoc();
    auto shape = op.getShape();
    auto rank = rewriter.create<::mlir::shape::RankOp>(loc, shape);
    auto i64Rank = rewriter.create<::mlir::arith::IndexCastOp>(
        loc, rewriter.getI64Type(), rank);
    auto shapeTnsr = rewriter.create<::mlir::tensor::CastOp>(
        loc, ::mlir::RankedTensorType::get({-1}, rewriter.getIndexType()),
        shape);
    auto i64Tnsr = rewriter.create<::mlir::arith::IndexCastOp>(
        loc, ::mlir::RankedTensorType::get({-1}, rewriter.getI64Type()),
        shapeTnsr);
    auto fsa = rewriter.getStringAttr("_idtr_init_dtensor");
    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, fsa, rewriter.getI64Type(), ::mlir::ValueRange({i64Tnsr, i64Rank}));
    return ::mlir::success();
  }
};

// LocalOffsetsOp -> call into _idtr_local_offsets
struct LocalOffsetsOpConverter
    : public mlir::OpRewritePattern<::imex::dist::LocalOffsetsOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalOffsetsOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    rewriter.replaceOp(
        op, createCallGetRankedData(loc, rewriter, "_idtr_local_offsets",
                                    op.getPtensor(), op.getRank()));
    return ::mlir::success();
  }
};

// LocalShapeOp -> call into _idtr_local_shape
struct LocalShapeOpConverter
    : public mlir::OpRewritePattern<::imex::dist::LocalShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalShapeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    rewriter.replaceOp(
        op, createCallGetRankedData(loc, rewriter, "_idtr_local_shape",
                                    op.getPtensor(), op.getRank()));
    return ::mlir::success();
  }
};

// AllReduceOp -> call into _idtr_reduce_all
struct AllReduceOpConverter
    : public mlir::OpRewritePattern<::imex::dist::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::AllReduceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto opV = rewriter.create<::mlir::arith::ConstantOp>(loc, op.getOp());
    auto rTnsr = op.getTensor();
    auto dtype = createInt<sizeof(int) * 8>(loc, rewriter, 5); // FIXME getDType
    auto fsa = rewriter.getStringAttr("_idtr_reduce_all");
    rewriter.create<::mlir::func::CallOp>(
        loc, fsa, ::mlir::TypeRange(), ::mlir::ValueRange({rTnsr, dtype, opV}));
    rewriter.replaceOp(op, rTnsr);
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct ConvertDistToStandardPass
    : public ::imex::ConvertDistToStandardBase<ConvertDistToStandardPass> {
  ConvertDistToStandardPass() = default;

  void runOnOperation() override {
    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<RuntimePrototypesOpConverter, RegisterPTensorOpConverter,
                   LocalOffsetsOpConverter, LocalShapeOpConverter,
                   AllReduceOpConverter>(getContext(), patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);
  }
};

} // namespace

/// Populate the given list with patterns that convert Dist to Standard
void populateDistToStandardConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass that convert Dist to Standard
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertDistToStandardPass() {
  return std::make_unique<ConvertDistToStandardPass>();
}

} // namespace imex
