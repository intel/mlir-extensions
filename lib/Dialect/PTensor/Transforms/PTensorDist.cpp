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
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
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
  auto rTnsr = builder.create<::imex::ptensor::ExtractMemRefOp>(
      loc, pTnsrTyp.getMemRefType(), pTnsr);
  return builder.create<::imex::dist::AllReduceOp>(loc, rTnsr.getType(), op,
                                                   rTnsr);
}

// create ops to extract the local Tensor from DistTensor
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

/// Rewriting ::imex::ptensor::ExtractMemRefOp
/// Get PTensor from DistTensor and apply to ExtractMemRefOp.
struct DistExtractMemRefOpRWP
    : public RecOpRewritePattern<::imex::ptensor::ExtractMemRefOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractMemRefOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // get input
    auto inpPtTyp =
        op.getInput().getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpPtTyp) {
      return ::mlir::failure();
    }
    auto pTnsr = createGetLocal(loc, rewriter, op.getInput());
    rewriter.replaceOpWithNewOp<::imex::ptensor::ExtractMemRefOp>(
        op, inpPtTyp.getPTensorType().getMemRefType(), pTnsr);
    return ::mlir::success();
  }
};

// Compute local slice in dim 0, all other dims are not partitioned (yet)
// return local memref of src
::mlir::Value createGetLocalSlice(::mlir::Location &loc,
                                  ::mlir::PatternRewriter &rewriter,
                                  const ::mlir::Value &src,
                                  const ::mlir::OperandRange &slcOffs,
                                  const ::mlir::OperandRange &slcSizes,
                                  const ::mlir::OperandRange &slcStrides,
                                  ::mlir::SmallVector<::mlir::Value> &lOffsets,
                                  ::mlir::SmallVector<::mlir::Value> &lSizes,
                                  bool doOffs = true) {
  // get input
  auto inpPtTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
  assert(inpPtTyp);

  EasyIdx zeroIdx(loc, rewriter, 0);
  EasyIdx oneIdx(loc, rewriter, 1);

  // Get the local part of the global slice, team, rank, offsets
  int64_t rank = (int64_t)inpPtTyp.getPTensorType().getRank();
  auto lPTnsr = createGetLocal(loc, rewriter, src);
  auto lMemRef = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
      loc, inpPtTyp.getPTensorType().getMemRefType(), lPTnsr);
  auto lOffs = createGetLocalOffsets(loc, rewriter, src);
  EasyIdx slcOffs0(loc, rewriter, slcOffs[0]);
  EasyIdx slcSizes0(loc, rewriter, slcSizes[0]);
  EasyIdx slcStrides0(loc, rewriter, slcStrides[0]);

  // last index of slice
  auto slcEnd = slcOffs0 + slcSizes0 * slcStrides0;
  // local extent/size
  EasyIdx lExtent(
      loc, rewriter,
      rewriter.create<::mlir::memref::DimOp>(loc, lMemRef, zeroIdx.get()));
  // local offset (dim 0)
  EasyIdx lOff(loc, rewriter,
               rewriter.create<::mlir::memref::LoadOp>(
                   loc, lOffs, ::mlir::ValueRange{zeroIdx.get()}));
  // last index of local partition
  auto lEnd = lOff * lExtent;
  // check if requested slice fully before local partition
  auto beforeLocal = slcEnd.ult(lOff);
  // check if requested slice fully behind local partition
  auto behindLocal = lEnd.ule(slcOffs0);
  // check if requested slice start before local partition
  auto startsBefore = slcOffs0.ult(lOff);
  auto strOff = lOff - slcOffs0;
  // (strOff / stride) * stride
  auto nextMultiple = (strOff / slcStrides0) * slcStrides0;
  // Check if local start is on a multiple of the new slice
  auto isMultiple = nextMultiple.eq(strOff);
  // stride - (strOff - nextMultiple)
  auto off = slcStrides0 - strOff - nextMultiple;
  // offset is either 0 if multiple or off
  auto lDiff1 = isMultiple.select(zeroIdx, off);
  // if view starts within our partition: (start-lOff)
  auto lDiff2 = slcOffs0 - lOff;
  auto viewOff1 = startsBefore.select(lDiff1, lDiff2);
  // except if slice/view before or behind local partition
  auto viewOff0 = beforeLocal.select(zeroIdx, viewOff1);
  auto viewOff = behindLocal.select(zeroIdx, viewOff0);
  // min of lEnd and slice's end
  auto theEnd = lEnd.min(slcEnd);
  // range between local views start and end
  auto lRange = theEnd - viewOff - lOff;
  // number of elements in local view (range+stride-1)/stride
  auto viewSize1 = (lRange + slcStrides0 - oneIdx) / slcStrides0;
  auto viewSize0 = beforeLocal.select(zeroIdx, viewSize1);
  auto viewSize = behindLocal.select(zeroIdx, viewSize0);

  // and store in output arguments
  lOffsets[0] = viewOff.get();
  lSizes[0] = viewSize.get();
  for (auto i = 1; i < rank; ++i) {
    lOffsets[i] = slcOffs[i];
    lSizes[i] = slcSizes[i];
  }

  if (doOffs) {
    // offset of local partition in new tensor/view
    auto lViewOff1 = (lOff + viewOff - slcOffs0) / slcStrides0;
    auto lViewOff0 = beforeLocal.select(slcOffs0, lViewOff1);
    auto lViewOff = behindLocal.select(slcEnd, lViewOff0);
    // create local offsets from above computed
    auto lViewOffs =
        createAllocMR(rewriter, loc, rewriter.getIndexType(), rank);
    if (rank > 1)
      (void)::mlir::linalg::makeMemRefCopyOp(rewriter, loc, lOffs, lViewOffs);
    rewriter.create<::mlir::memref::StoreOp>(loc, lViewOff.get(), lViewOffs,
                                             zeroIdx.get());

    return lViewOffs;
  }
  return ::mlir::Value();
}

/// Rewriting ::imex::ptensor::ExtractSliceOp
/// 1. Compute local slice and offsets of dst
/// 2. Apply ExtractSliceOp to subslice and local partition
/// 3. Create new DistTensor
struct DistExtractSliceOpRWP
    : public RecOpRewritePattern<::imex::ptensor::ExtractSliceOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractSliceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSource();
    // get input
    auto inpPtTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpPtTyp) {
      return ::mlir::failure();
    }

    // Get the local part of the global slice, team, rank, offsets
    int64_t rank = (int64_t)inpPtTyp.getPTensorType().getRank();
    auto slcOffs = op.getOffsets();
    auto slcSizes = op.getSizes();
    auto slcStrides = op.getStrides();

    // Compute local part of slice
    ::mlir::SmallVector<::mlir::Value> lSlcOffsets(rank);
    ::mlir::SmallVector<::mlir::Value> lSlcSizes(rank);
    auto lViewOffsets =
        createGetLocalSlice(loc, rewriter, src, slcOffs, slcSizes, slcStrides,
                            lSlcOffsets, lSlcSizes);

    // create local view
    auto lPTnsr = createGetLocal(loc, rewriter, src);
    auto lView = rewriter.create<::imex::ptensor::ExtractSliceOp>(
        loc, inpPtTyp.getPTensorType(), lPTnsr, lSlcOffsets, lSlcSizes,
        slcStrides);

    // create global shape from slice sizes
    auto gShape = createMemRefFromElements(rewriter, loc,
                                           rewriter.getIndexType(), {slcSizes});
    auto team = createTeamOf(loc, rewriter, src);
    // init our new dist tensor
    rewriter.replaceOp(
        op, createMkTnsr(loc, rewriter, gShape, lView, lViewOffsets, team));
    return ::mlir::success();
  }
};

/// Rewriting ::imex::ptensor::InsertSliceOp
/// 1. Compute local slice of dst
/// 2. Get local PTensors (dst and src)
/// 3. Apply to ::imex::ptensor::InsertSliceOp
struct DistInsertSliceOpRWP
    : public RecOpRewritePattern<::imex::ptensor::InsertSliceOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::InsertSliceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // check if inputs are DistTensors
    auto dst = op.getDestination();
    auto src = op.getSource();
    auto dstPTTyp = dst.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto srcPTTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!dstPTTyp || !srcPTTyp) {
      return ::mlir::failure();
    }

    // get slice info and create distributed view
    int64_t rank = (int64_t)dstPTTyp.getPTensorType().getRank();
    auto slcOffs = op.getOffsets();
    auto slcSizes = op.getSizes();
    auto slcStrides = op.getStrides();
    ::mlir::SmallVector<::mlir::Value> lSlcOffsets(rank);
    ::mlir::SmallVector<::mlir::Value> lSlcSizes(rank);
    (void)createGetLocalSlice(loc, rewriter, dst, slcOffs, slcSizes, slcStrides,
                              lSlcOffsets, lSlcSizes, false);

    // get local ptensors and apply to InsertSliceOp
    auto lDst = createGetLocal(loc, rewriter, dst);
    auto lSrc = createGetLocal(loc, rewriter, src);
    rewriter.replaceOpWithNewOp<::imex::ptensor::InsertSliceOp>(
        op, lDst, lSrc, lSlcOffsets, lSlcSizes, slcStrides);

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
    EasyIdx start(loc, rewriter, op.getStart());
    EasyIdx stop(loc, rewriter, op.getStop());
    EasyIdx step(loc, rewriter, op.getStep());
    // compute global count (so we know the shape)
    auto count = createCountARange(rewriter, loc, start, stop, step);
    auto dtype = rewriter.getI64Type(); // FIXME
    auto indexTyp = rewriter.getIndexType();
    // get number of procs and prank
    auto nProcs = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto pRank = rewriter.create<::imex::dist::PRankOp>(loc, team);
    // result shape is 1d
    constexpr int64_t rank = 1;
    auto gShape = createMemRefFromElements(rewriter, loc, indexTyp, {count});

    // so is the local shape
    llvm::SmallVector<mlir::Value> lShapeVVec(rank);
    // get local shape
    auto lShapeVVec_mr = rewriter.create<::imex::dist::LocalShapeOp>(
        loc, rank, nProcs, pRank, gShape);
    auto zero = createIndex(loc, rewriter, 0);
    EasyIdx lSz(loc, rewriter,
                rewriter.create<::mlir::memref::LoadOp>(
                    loc, lShapeVVec_mr, ::mlir::ValueRange({zero})));
    // get local offsets
    auto offsets = rewriter.create<::imex::dist::LocalOffsetsOp>(
        loc, rank, nProcs, pRank, gShape);
    // create start from offset
    EasyIdx off(loc, rewriter,
                rewriter.create<::mlir::memref::LoadOp>(
                    loc, offsets, ::mlir::ValueRange({zero})));
    start = start + (off * step);
    // create stop
    stop = start + (step * lSz); // start + (lShape[0] * step)
    //  get type of local tensor
    auto artype = ::imex::ptensor::PTensorType::get(rewriter.getContext(), rank,
                                                    dtype, false);
    // finally create local arange
    auto arres = rewriter.create<::imex::ptensor::ARangeOp>(
        loc, artype, start.get(), stop.get(), step.get(), op.getDevice(),
        nullptr);
    rewriter.replaceOp(
        op, createMkTnsr(loc, rewriter, gShape, arres, offsets, team));
    return ::mlir::success();
  }
};

/// Rewrite ::imex::ptensor::EWBinOp to get a distributed ewbinop
/// if operands are distributed.
/// Create global, distributed output tensor with same shape as operands.
/// The local partitions of operands (e.g. RankedTensor) are wrapped in
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
    int64_t rank = (int64_t)lhsDtTyp.getPTensorType().getRank();
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
/// The local partitions of operand (e.g. RankedTensor) is wrapped in
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
    int64_t rank = 0;
    auto dtype = inpDtTyp.getPTensorType().getElementType();
    auto retPtTyp = ::imex::ptensor::PTensorType::get(rewriter.getContext(), 0,
                                                      dtype, false);
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
        loc, false, retRTnsr, dmy);
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
                   DistExtractMemRefOpRWP, DistExtractSliceOpRWP,
                   DistInsertSliceOpRWP>(getContext(), patterns);
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
