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

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
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
  auto tnsr = builder.create<::mlir::linalg::InitTensorOp>(
      loc, ::mlir::ValueRange({rankV}), builder.getI64Type());
  auto fsa = builder.getStringAttr(func);
  (void)builder.create<::mlir::func::CallOp>(
      loc, fsa, ::mlir::TypeRange(),
      ::mlir::ValueRange({guid, tnsr, createInt(loc, builder, rank)}));
  return tnsr;
}

// Declaring runtime function prototypes
// Each pass adding calls into the runtime should instantiate a static var
// of this type. All functions will be declared exactly once.
struct RequiredRTFuncs {
  // find framing module op and declare function prototypes if not done yet
  template <typename T>
  RequiredRTFuncs(::mlir::Location &loc, ::mlir::OpBuilder &builder, T op) {
    mlir::ModuleOp module;
    auto mod = op->getParentOp();
    while (mod) {
      if (::mlir::isa<mlir::ModuleOp>(mod)) {
        module = ::mlir::cast<mlir::ModuleOp>(mod);
        break;
      }
      mod = mod->getParentOp();
    }
    requireAll(loc, builder, module);
  }

  // each runtime function is prototype exactly once using static vars
  // names and type signatures are currently hard-coded.
  static void requireAll(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                         mlir::ModuleOp module) {
    auto dtype = builder.getI64Type();
    auto i64Type = builder.getI64Type();
    auto dtypeType = builder.getIntegerType(sizeof(int) * 8);
    auto opType =
        builder.getIntegerType(sizeof(::imex::ptensor::ReduceOpId) * 8);
    [[maybe_unused]] static bool _init = requireFunc(
        loc, builder, module, "_idtr_init_dtensor",
        {::mlir::RankedTensorType::get({-1}, dtype), i64Type}, {i64Type});
    [[maybe_unused]] static bool _lShp = requireFunc(
        loc, builder, module, "_idtr_local_shape",
        {dtype, ::mlir::RankedTensorType::get({-1}, dtype), i64Type}, {});
    [[maybe_unused]] static bool _lOffs = requireFunc(
        loc, builder, module, "_idtr_local_offsets",
        {dtype, ::mlir::RankedTensorType::get({-1}, dtype), i64Type}, {});
    [[maybe_unused]] static bool _allRed = requireFunc(
        loc, builder, module, "_idtr_reduce_all",
        {::mlir::RankedTensorType::get({}, dtype), dtypeType, opType}, {});
  }

  // create function prototype fo given function name, arg-types and
  // return-types
  static bool requireFunc(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp module, const char *fname,
                          ::mlir::TypeRange args, ::mlir::TypeRange results) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    // Insert before module terminator.
    builder.setInsertionPoint(module.getBody(),
                              std::prev(module.getBody()->end()));
    auto funcType = builder.getFunctionType(args, results);
    auto func = builder.create<::mlir::func::FuncOp>(loc, fname, funcType);
    func.setPrivate();
    return true;
  }
};

// *******************************
// ***** Individual patterns *****
// *******************************

// RegisterPTensorOp -> call into _idtr_init_dtensor
struct RegisterPTensorOpConverter
    : public mlir::OpRewritePattern<::imex::dist::RegisterPTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RegisterPTensorOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // get RankedTensor and rank and call registration function
    auto loc = op.getLoc();
    // declare our runtime funcs once (it's static)
    static RequiredRTFuncs _rtFuncs(loc, rewriter, op);

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
    // declare our runtime funcs once (it's static)
    static RequiredRTFuncs _rtFuncs(loc, rewriter, op);

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
    insertPatterns<RegisterPTensorOpConverter, LocalOffsetsOpConverter,
                   LocalShapeOpConverter, AllReduceOpConverter>(getContext(),
                                                                patterns);
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
std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createConvertDistToStandardPass() {
  return std::make_unique<ConvertDistToStandardPass>();
}

} // namespace imex
