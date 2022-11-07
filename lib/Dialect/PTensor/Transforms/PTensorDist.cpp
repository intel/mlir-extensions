//===- PTensorDist.cpp - PTensorToDist Transform  ---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transform of the PTensor dialect to a combination of
/// PTensor and Dist dialects.
///
/// PTensor operations will stay untouched unless operands are distributed
/// PTensors. If they are distributed necessary communication with the
/// runtime is performed to identify the local partition (mostly for creation
/// functions). The local tensor is extracted/created and the operation is
/// re-issued for the local part. No deep recursion happens because the operands
/// for the newly created ptensor operations are not distributed. Finally
/// additional ops are added of more communication with the runtime is needed,
/// for example to perform a final global reduction.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>
#include <imex/Util/PassWrapper.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

#include "PassDetail.h"

namespace imex {

namespace {

// *******************************
// ***** Some helper functions ***
// *******************************

// create ::imex::dist::RegisterPTensorOp
inline ::mlir::Value createInitWithRT(::mlir::Location &loc,
                                      ::mlir::OpBuilder &builder, uint64_t rank,
                                      ::mlir::Value gshape) {
  return builder.create<::imex::dist::RegisterPTensorOp>(
      loc, builder.getI64Type(), gshape);
}

// create ::imex::dist::LocalShapeOp
inline ::mlir::Value createGetLocalShape(::mlir::Location &loc,
                                         ::mlir::OpBuilder &builder,
                                         ::mlir::Value guid, uint64_t rank) {
  auto rankA = builder.getIntegerAttr(builder.getI64Type(), rank);
  return builder.create<::imex::dist::LocalShapeOp>(
      loc, ::mlir::RankedTensorType::get({(int64_t)rank}, builder.getI64Type()),
      rankA, guid);
}

// create ::imex::dist::LocalOffsetsOp
inline ::mlir::Value createGetLocalOffsets(::mlir::Location &loc,
                                           ::mlir::OpBuilder &builder,
                                           ::mlir::Value guid, uint64_t rank) {
  auto rankA = builder.getIntegerAttr(builder.getI64Type(), rank);
  return builder.create<::imex::dist::LocalOffsetsOp>(
      loc, ::mlir::RankedTensorType::get({(int64_t)rank}, builder.getI64Type()),
      rankA, guid);
}

// extract RankedTensor and create ::imex::dist::AllReduceOp
inline ::mlir::Value createAllReduce(::mlir::Location &loc,
                                     ::mlir::OpBuilder &builder,
                                     ::mlir::Attribute op,
                                     ::mlir::Value pTnsr) {
  auto pTnsrTyp = pTnsr.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(pTnsrTyp);
  auto rTnsr = builder.create<::imex::ptensor::ExtractRTensorOp>(
      loc, pTnsrTyp.getRtensor(), pTnsr);
  return builder.create<::imex::dist::AllReduceOp>(loc, rTnsr.getType(), op,
                                                   rTnsr);
}

// create ops to extract the local RankedTensor from PTensor
inline ::mlir::Value createGetLocal(::mlir::Location &loc,
                                    ::mlir::OpBuilder &builder,
                                    ::mlir::Value pt) {
  auto ptTyp = pt.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(ptTyp);
  if (ptTyp.getDist()) {
    auto rtnsr = builder.create<::imex::ptensor::ExtractRTensorOp>(
        loc, ptTyp.getRtensor(), pt);
    // FIXME: device
    return builder.create<::imex::ptensor::MkPTensorOp>(loc, rtnsr);
  }
  // not dist
  return pt;
}

// extract RankedTensor and create ::imex::ptensor::MkPTensorOp
inline ::mlir::Value createMkTnsr(::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder, ::mlir::Value pt,
                                  ::mlir::Value guid) {
  auto ptTyp = pt.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(ptTyp);
  auto rTnsr = builder.create<::imex::ptensor::ExtractRTensorOp>(
      loc, ptTyp.getRtensor(), pt);
  auto dmy = createInt<1>(loc, builder, 0);
  return builder.create<::imex::ptensor::MkPTensorOp>(loc, false, true, rTnsr,
                                                      dmy, dmy, guid);
}

// *******************************
// ***** Individual patterns *****
// *******************************

// Baseclass for Rewriters which handle recursion
// All our rewriters replace ops with series of ops icnluding the
// op-type which gets rewritten. Rewriters will not rewrite (stop recursion)
// if input PTensor operands are not distributed.
template <typename T>
struct RecOpRewritePattern : public ::mlir::OpRewritePattern<T> {
  using ::mlir::OpRewritePattern<T>::OpRewritePattern;
  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    RecOpRewritePattern<T>::setHasBoundedRewriteRecursion();
  }
};

/// Create a distributed arange if applicable.
/// Create global, distributed output PTensor as defined by operands.
/// The local partition (e.g. a RankedTensor) are wrapped in a
/// non-distributed PTensor and re-applied to arange op.
/// op gets replaced with global PTensor
struct DistARange : public RecOpRewritePattern<::imex::ptensor::ARangeOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ARangeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // nothing to do if no team
    auto team = op.getTeam();
    if (!team)
      return ::mlir::failure();

    // get operands
    auto start = op.getStart();
    auto step = op.getStep();
    // compute global count (so we know the shape)
    auto count = createCountARange(rewriter, loc, start, op.getStop(), step);
    auto dtype = rewriter.getI64Type(); // FIXME
    auto i64Typ = rewriter.getI64Type();
    // result shape is 1d
    uint64_t rank = 1;
    auto gShpTnsr = rewriter.create<::mlir::tensor::EmptyOp>(
        loc, ::mlir::ArrayRef<::mlir::OpFoldResult>({count}), dtype);
    auto gShape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, gShpTnsr);
    // so is the local shape
    llvm::SmallVector<mlir::Value> lShapeVVec(1);
    // get guid
    auto guid = createInitWithRT(loc, rewriter, 1, gShape);
    // get local shape
    auto lShapeVVec_mr = createGetLocalShape(loc, rewriter, guid, rank);
    auto zero = createIndex(loc, rewriter, 0);
    auto lSz = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, i64Typ, lShapeVVec_mr, ::mlir::ValueRange({zero}));
    // get local offsets
    auto offsets = createGetLocalOffsets(loc, rewriter, guid, rank);
    // create start from offset
    auto off = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, i64Typ, offsets, ::mlir::ValueRange({zero}));
    auto tmp =
        rewriter.create<::mlir::arith::MulIOp>(loc, off, step); // off * step
    start = rewriter.create<::mlir::arith::AddIOp>(
        loc, start, tmp); // start + (off * stride)
    // create stop
    auto tmp2 = rewriter.create<::mlir::arith::MulIOp>(
        loc, lSz, step); // step * lShape[0]
    auto stop = rewriter.create<::mlir::arith::AddIOp>(
        loc, start, tmp2); // start + (lShape[0] * stride)
    //  get type of local tensor
    ::llvm::ArrayRef<int64_t> lShape({-1});
    auto artype = ::imex::ptensor::PTensorType::get(
        rewriter.getContext(), ::mlir::RankedTensorType::get({-1}, dtype),
        false, false);
    // finally create local arange
    auto dmy = ::mlir::Value(); // createInt<1>(loc, rewriter, 0);
    auto arres = rewriter.create<::imex::ptensor::ARangeOp>(
        loc, artype, start, stop, step, op.getDevice(), dmy);
    rewriter.replaceOp(op, createMkTnsr(loc, rewriter, arres, guid));
    return ::mlir::success();
  }
};

/// Create a distributed ewbinop if operands are distributed.
/// Create global, distributed output PTensor with same shape as operands.
/// The local partitions of operands (e.g. RankedTensors) are wrapped in
/// non-distributed PTensors and re-applied to ewbinop.
/// op gets replaced with global PTensor
struct DistEWBinOp : public RecOpRewritePattern<::imex::ptensor::EWBinOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhsPtTyp =
        op.getLhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    auto rhsPtTyp =
        op.getRhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    // return success if wrong ops or not distributed
    if (!lhsPtTyp || !rhsPtTyp || !lhsPtTyp.getDist() || !lhsPtTyp.getDist()) {
      return ::mlir::failure();
    }

    // result shape
    auto gShapeARef = lhsPtTyp.getRtensor().getShape();
    auto gShapeAttr = rewriter.getIndexVectorAttr(gShapeARef);
    auto gShape = rewriter.create<::mlir::shape::ConstShapeOp>(loc, gShapeAttr);
    // auto dtype = lhsPtTyp.getRtensor().getElementType();
    // Init our new dist tensor
    auto guid = createInitWithRT(loc, rewriter, 1, gShape);
    // local ewb op
    auto lLhs = createGetLocal(loc, rewriter, op.getLhs());
    auto lRhs = createGetLocal(loc, rewriter, op.getRhs());
    // return type same as lhs for now
    auto retPtTyp = lLhs.getType(); // FIXME
    auto ewbres = rewriter.create<::imex::ptensor::EWBinOp>(
        loc, retPtTyp, op.getOp(), lLhs, lRhs);
    rewriter.replaceOp(op, createMkTnsr(loc, rewriter, ewbres, guid));
    return ::mlir::success();
  }
};

/// Create a distributed reduction if operand is distributed.
/// Create global, distributed 0d output PTensor.
/// The local partitions of operand (e.g. RankedTensors) is wrapped in
/// non-distributed PTensor and re-applied to reduction.
/// The result is then applied to a distributed allreduce.
/// op gets replaced with global PTensor
struct DistReductionOp
    : public RecOpRewritePattern<::imex::ptensor::ReductionOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReductionOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // FIXME reduction over individual dimensions is not supported
    auto loc = op.getLoc();
    // get input
    auto inpPtTyp =
        op.getInput().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!inpPtTyp || !inpPtTyp.getDist()) {
      return ::mlir::failure();
    }

    // result shape is 0d
    auto gShapeAttr = rewriter.getIndexTensorAttr({});
    auto gShape = rewriter.create<::mlir::shape::ConstShapeOp>(loc, gShapeAttr);
    // Init our new dist tensor
    auto guid = createInitWithRT(loc, rewriter, 1, gShape);
    // Local reduction
    auto local = createGetLocal(loc, rewriter, op.getInput());
    // return type 0d with same dtype as input
    auto dtype = inpPtTyp.getRtensor().getElementType();
    auto retPtTyp = ::imex::ptensor::PTensorType::get(
        rewriter.getContext(), ::mlir::RankedTensorType::get({}, dtype), false,
        false);
    auto redPTnsr = rewriter.create<::imex::ptensor::ReductionOp>(
        loc, retPtTyp, op.getOp(), local);
    // global reduction
    auto retRTnsr = createAllReduce(loc, rewriter, op.getOp(), redPTnsr);
    // finish
    auto dmy = createInt<1>(loc, rewriter, 0);
    rewriter.replaceOpWithNewOp<::imex::ptensor::MkPTensorOp>(
        op, false, true, retRTnsr, dmy, dmy, guid);
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct PTensorDistPass : public ::imex::PTensorDistBase<PTensorDistPass> {

  PTensorDistPass() = default;

  void runOnOperation() override {
    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<DistARange, DistEWBinOp, DistReductionOp>(getContext(),
                                                             patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);
  }
};

} // namespace

/// Populate the given list with patterns that eliminate Dist ops
void populatePTensorDistPatterns(::mlir::LLVMTypeConverter &converter,
                                 ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createPTensorDistPass() {
  return std::make_unique<PTensorDistPass>();
}

} // namespace imex
