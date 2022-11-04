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
/// PTensors. PTensors are converted do DistTensorTypes by creation functions,
/// for example by reacting on an input argument 'team'. When creating a
/// DistTensor additional information is attached which provides information to
/// perform distributed operations, such as shape and offsets of the local
/// partition. If operations work on distributed tensors necessary communication
/// with the runtime is performed to identify the local partition. The local
/// tensor is extracted/created and the operation is re-issued for the local
/// part. No deep recursion happens because the operands for the newly created
/// ptensor operations are not distributed. Finally additional ops are added of
/// more communication with the runtime is needed, for example to perform a
/// final global reduction.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>
#include <imex/internal/PassWrapper.h>

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

// create ::imex::dist::LocalOffsetsOp
inline ::mlir::Value createGetLocalOffsets(::mlir::Location &loc,
                                           ::mlir::OpBuilder &builder,
                                           ::mlir::Value dt) {
  return builder.create<::imex::dist::ExtractFromDistOp>(
      loc, ::imex::dist::LOFFSETS, dt);
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

// create ops to extract the local RankedTensor from DistTensor
inline ::mlir::Value createGetLocal(::mlir::Location &loc,
                                    ::mlir::OpBuilder &builder,
                                    ::mlir::Value dt) {
  return builder.create<::imex::dist::ExtractFromDistOp>(
      loc, ::imex::dist::LTENSOR, dt);
}

// Create a DistTensor from a PTensor and meta data
inline ::mlir::Value createMkTnsr(::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder,
                                  ::mlir::Value gshape, ::mlir::Value pt,
                                  ::mlir::Value loffsets, ::mlir::Value team) {
  return builder.create<::imex::dist::InitDistTensorOp>(loc, gshape, pt,
                                                        loffsets, team);
}

// extract team component from given DistTensor
inline ::mlir::Value createTeamOf(::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder,
                                  ::mlir::Value dt) {
  auto dtTyp = dt.getType().dyn_cast<::imex::dist::DistTensorType>();
  assert(dtTyp);
  return builder.create<::imex::dist::ExtractFromDistOp>(
      loc, ::imex::dist::TEAM, dt);
}

// *******************************
// ***** Individual patterns *****
// *******************************

// Base-class for RewriterPatterns which handle recursion
// All our rewriters replace ops with series of ops including the
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

/// Rewriting ::imex::ptensor::ExtractRTensorOp
/// Get PTensor from DistTensor and apply to ExtractTensorOp.
struct DistExtractRTensorOpRWP
    : public RecOpRewritePattern<::imex::ptensor::ExtractRTensorOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractRTensorOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // get input
    auto inpPtTyp =
        op.getInput().getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpPtTyp) {
      return ::mlir::failure();
    }
    auto pTnsr = createGetLocal(loc, rewriter, op.getInput());
    rewriter.replaceOpWithNewOp<::imex::ptensor::ExtractRTensorOp>(
        op, inpPtTyp.getPTensorType().getRtensor(), pTnsr);
    return ::mlir::success();
  }
};

/// Rewriting ::imex::ptensor::ARangeOp to get a distributed arange if
/// applicable. Create global, distributed output Tensor as defined by operands.
/// The local partition (e.g. a RankedTensor) are wrapped in a
/// non-distributed PTensor and re-applied to arange op.
/// op gets replaced with global DistTensor
struct DistARangeOpRWP : public RecOpRewritePattern<::imex::ptensor::ARangeOp> {
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
    // get number of procs and prank
    auto nProcs = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto pRank = rewriter.create<::imex::dist::PRankOp>(loc, team);
    // result shape is 1d
    constexpr uint64_t rank = 1;
    auto cnt = rewriter.create<::mlir::arith::IndexCastOp>(loc, i64Typ, count)
                   .getResult();
    auto gShape = rewriter.create<::mlir::tensor::FromElementsOp>(
        loc, ::mlir::RankedTensorType::get({rank}, i64Typ), cnt);

    // so is the local shape
    llvm::SmallVector<mlir::Value> lShapeVVec(rank);
    // get local shape
    auto lShapeVVec_mr = rewriter.create<::imex::dist::LocalShapeOp>(
        loc, rank, nProcs, pRank, gShape);
    auto zero = createIndex(loc, rewriter, 0);
    auto lSz = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, i64Typ, lShapeVVec_mr, ::mlir::ValueRange({zero}));
    // get local offsets
    auto offsets = rewriter.create<::imex::dist::LocalOffsetsOp>(
        loc, rank, nProcs, pRank, gShape);
    // create start from offset
    auto off = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, i64Typ, offsets, ::mlir::ValueRange({zero}));
    auto tmp =
        rewriter.create<::mlir::arith::MulIOp>(loc, off, step); // off * step
    start = rewriter.create<::mlir::arith::AddIOp>(loc, start,
                                                   tmp); // start + (off * step)
    // create stop
    auto tmp2 = rewriter.create<::mlir::arith::MulIOp>(
        loc, lSz, step); // step * lShape[0]
    auto stop = rewriter.create<::mlir::arith::AddIOp>(
        loc, start, tmp2); // start + (lShape[0] * step)
    //  get type of local tensor
    auto artype = ::imex::ptensor::PTensorType::get(
        rewriter.getContext(), ::mlir::RankedTensorType::get({-1}, dtype),
        false);
    // finally create local arange
    auto arres = rewriter.create<::imex::ptensor::ARangeOp>(
        loc, artype, start, stop, step, op.getDevice(), nullptr);
    rewriter.replaceOp(
        op, createMkTnsr(loc, rewriter, gShape, arres, offsets, team));
    return ::mlir::success();
  }
};

/// Rewrite ::imex::ptensor::EWBinOp to get a distributed ewbinop
/// if operands are distributed.
/// Create global, distributed output tensor with same shape as operands.
/// The local partitions of operands (e.g. RankedTensors) are wrapped in
/// non-distributed PTensors and re-applied to ewbinop.
/// op gets replaced with global DistTensor
struct DistEWBinOpRWP : public RecOpRewritePattern<::imex::ptensor::EWBinOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhsDtTyp =
        op.getLhs().getType().dyn_cast<::imex::dist::DistTensorType>();
    auto rhsDtTyp =
        op.getRhs().getType().dyn_cast<::imex::dist::DistTensorType>();
    // return failure if wrong ops or not distributed
    if (!lhsDtTyp || !rhsDtTyp) {
      return ::mlir::failure();
    }

    // local ewb operands
    auto lLhs = createGetLocal(loc, rewriter, op.getLhs());
    auto lRhs = createGetLocal(loc, rewriter, op.getRhs());
    // return type same as lhs for now
    auto retPtTyp = lLhs.getType(); // FIXME
    auto ewbres = rewriter.create<::imex::ptensor::EWBinOp>(
        loc, retPtTyp, op.getOp(), lLhs, lRhs);
    // get rank, global shape, offsets and team
    uint64_t rank = lhsDtTyp.getPTensorType().getRtensor().getRank();
    auto gShape = rewriter.create<::imex::dist::ExtractFromDistOp>(
        loc, ::imex::dist::GSHAPE, op.getLhs());
    auto team = createTeamOf(loc, rewriter, op.getLhs());
    auto nProcs = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto pRank = rewriter.create<::imex::dist::PRankOp>(loc, team);
    auto lOffs = rewriter.create<::imex::dist::LocalOffsetsOp>(
        loc, rank, nProcs, pRank, gShape);
    // and init our new dist tensor
    rewriter.replaceOp(
        op, createMkTnsr(loc, rewriter, gShape, ewbres, lOffs, team));
    return ::mlir::success();
  }
};

/// Rewrite ::imex::ptensor::ReductionOp to get a distributed
/// reduction if operand is distributed.
/// Create global, distributed 0d output tensor.
/// The local partitions of operand (e.g. RankedTensors) is wrapped in
/// non-distributed PTensor and re-applied to reduction.
/// The result is then applied to a distributed allreduce.
/// op gets replaced with global DistTensor
struct DistReductionOpRWP
    : public RecOpRewritePattern<::imex::ptensor::ReductionOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReductionOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // FIXME reduction over individual dimensions is not supported
    auto loc = op.getLoc();
    // get input
    auto inpDtTyp =
        op.getInput().getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpDtTyp) {
      return ::mlir::failure();
    }

    // Local reduction
    auto local = createGetLocal(loc, rewriter, op.getInput());
    // return type 0d with same dtype as input
    uint64_t rank = 0;
    auto dtype = inpDtTyp.getPTensorType().getRtensor().getElementType();
    auto retPtTyp = ::imex::ptensor::PTensorType::get(
        rewriter.getContext(), ::mlir::RankedTensorType::get({}, dtype), false);
    auto redPTnsr = rewriter.create<::imex::ptensor::ReductionOp>(
        loc, retPtTyp, op.getOp(), local);
    // global reduction
    auto retRTnsr = createAllReduce(loc, rewriter, op.getOp(), redPTnsr);
    // get global shape, offsets and team
    // result shape is 0d
    auto gShape = rewriter.create<::imex::dist::ExtractFromDistOp>(
        loc, ::imex::dist::GSHAPE, op.getInput());
    auto team = createTeamOf(loc, rewriter, op.getInput());
    auto nProcs = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto pRank = rewriter.create<::imex::dist::PRankOp>(loc, team);
    auto lOffs = rewriter.create<::imex::dist::LocalOffsetsOp>(
        loc, rank ? rank : 1, nProcs, pRank, gShape);
    // and init our new dist tensor
    auto dmy = ::imex::createInt<1>(loc, rewriter, 0); // FIXME
    auto resPTnsr = rewriter.create<::imex::ptensor::MkPTensorOp>(
        loc, false, retRTnsr, dmy, team);
    rewriter.replaceOp(
        op, createMkTnsr(loc, rewriter, gShape, resPTnsr, lOffs, team));
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
    insertPatterns<DistARangeOpRWP, DistEWBinOpRWP, DistReductionOpRWP,
                   DistExtractRTensorOpRWP>(getContext(), patterns);
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
std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createPTensorDistPass() {
  return std::make_unique<PTensorDistPass>();
}

} // namespace imex
