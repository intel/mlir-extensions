//===- DistToStandard.cpp - DistToStandard conversion  ----------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DistToStandard conversion, converting the Dist
/// dialect to standard dialects (including DistRuntime and NDArray).
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/DistToStandard/DistToStandard.h>
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Transforms/Utils.h>
#include <imex/Dialect/NDArray/Utils/Utils.h>
#include <imex/Dialect/Region/Transforms/RegionConversions.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <array>
#include <cstdlib>
#include <optional>

namespace imex {
#define GEN_PASS_DEF_CONVERTDISTTOSTANDARD
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

using ::imex::ndarray::createDType;
using ::imex::ndarray::createShapeOf;

namespace imex {
namespace dist {
namespace {

// *******************************
// ***** Individual patterns *****
// *******************************

/// Rewriting ::imex::ndarray::LinSpaceOp to get a distributed linspace if
/// applicable.
struct LinSpaceOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::LinSpaceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::LinSpaceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::LinSpaceOp op,
                  ::imex::ndarray::LinSpaceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto retArType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());
    if (!retArType)
      return ::mlir::failure();
    auto dEnv = getDistEnv(retArType);
    // nothing to do if not distributed
    if (!dEnv)
      return ::mlir::failure();

    auto team = dEnv.getTeam();
    auto start = op.getStart();
    auto stop = op.getStop();
    auto count = op.getNum();
    bool endpoint = op.getEndpoint();

    if (!(start.getType().isIntOrIndexOrFloat() &&
          stop.getType().isIntOrIndexOrFloat() &&
          count.getType().isIntOrIndex() && retArType)) {
      return ::mlir::failure();
    } // FIXME type promotion

    // cast types and get step
    auto elTyp = retArType.getElementType();
    count = createIndexCast(loc, rewriter, count);
    auto bw = elTyp.isIndex() ? 64 : elTyp.getIntOrFloatBitWidth();
    ::mlir::Type cType =
        bw > 32 ? rewriter.getF64Type()
                : (bw > 16 ? rewriter.getF32Type() : rewriter.getF16Type());
    start = createCast(loc, rewriter, start, cType);
    stop = createCast(loc, rewriter, stop, cType);
    auto step =
        createStepLinSpace(rewriter, loc, start, stop, count, endpoint, cType);

    // get number of procs and prank
    auto nProcs = createNProcs(loc, rewriter, team);
    auto pRank = createPRank(loc, rewriter, team);

    // get local shape and offsets
    auto lPart = rewriter.create<::imex::dist::DefaultPartitionOp>(
        loc, nProcs, pRank, ::mlir::ValueRange{count});
    auto lShape = lPart.getLShape();
    auto lOffs = lPart.getLOffsets();

    // use local shape and offset to compute local linspace
    auto off = createCast(loc, rewriter, lOffs[0], cType);
    auto lSz = createCast(loc, rewriter, lShape[0], cType);

    start = rewriter.createOrFold<::mlir::arith::AddFOp>(
        loc, rewriter.createOrFold<::mlir::arith::MulFOp>(loc, step, off),
        start);
    stop = rewriter.createOrFold<::mlir::arith::AddFOp>(
        loc, rewriter.createOrFold<::mlir::arith::MulFOp>(loc, step, lSz),
        start);

    // finally create local linspace
    auto res = rewriter.create<::imex::ndarray::LinSpaceOp>(
        loc, start, stop, lShape[0], false, ndarray::fromMLIR(elTyp),
        getNonDistEnvs(retArType));

    rewriter.replaceOp(op, createDistArray(loc, rewriter, team, {op.getNum()},
                                           lOffs, res.getResult()));
    return ::mlir::success();
  }
};

/// Rewriting ::imex::ndarray::CreateOp to get a distributed CreateOp if
/// applicable. Create global, distributed output array as defined by operands.
/// The local partition (e.g. a RankedTensor) are wrapped in a
/// non-distributed NDArray and re-applied to CreateOp.
/// op gets replaced with global distributed array
struct CreateOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::CreateOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::CreateOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CreateOp op,
                  ::imex::ndarray::CreateOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto retArType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());
    if (!retArType)
      return ::mlir::failure();
    auto dEnv = getDistEnv(retArType);
    // nothing to do if not distributed
    if (!dEnv)
      return ::mlir::failure();

    auto team = dEnv.getTeam();
    auto gShape = op.getShape();
    // get local shape and offsets
    auto lPart = createDefaultPartition(loc, rewriter, team, gShape);

    // finally create local array
    auto arres = rewriter.create<::imex::ndarray::CreateOp>(
        loc, lPart.getLShape(), ndarray::fromMLIR(retArType.getElementType()),
        op.getValue(), getNonDistEnvs(retArType));

    rewriter.replaceOp(op,
                       createDistArray(loc, rewriter, team, gShape,
                                       lPart.getLOffsets(), arres.getResult()));
    return ::mlir::success();
  }
};

/// Convert a CopyOp on a distributed array to CopyOps on the local data.
struct CopyOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::CopyOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::CopyOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CopyOp op,
                  ::imex::ndarray::CopyOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = op.getSource();
    auto srcDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto resDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    // return failure if wrong ops or not distributed
    if (!srcDistType || !isDist(srcDistType) || !resDistType ||
        !isDist(resDistType)) {
      return ::mlir::failure();
    }
    // FIXME: check if part shapes are compatible

    auto loc = op.getLoc();
    auto partTypes = getPartsTypes(resDistType);
    auto lParts = createPartsOf(loc, rewriter, src);
    auto lOffsets = createLocalOffsetsOf(loc, rewriter, src);

    // apply CopyOp to all parts
    ::imex::ValVec resParts;
    for (auto i = 0u; i < lParts.size(); ++i) {
      auto partOp = rewriter.create<::imex::ndarray::CopyOp>(loc, partTypes[i],
                                                             lParts[i]);
      resParts.emplace_back(partOp.getResult());
    }

    // get global shape
    auto gShape = resDistType.getShape();
    // and init our new dist array
    rewriter.replaceOp(op, createDistArray(loc, rewriter,
                                           getDistEnv(srcDistType).getTeam(),
                                           gShape, lOffsets, resParts));

    return ::mlir::success();
  }
};

/// Convert a DeleteOp on a distributed array to DeleteOps on the local data.
struct DeleteOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::DeleteOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::DeleteOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::DeleteOp op,
                  ::imex::ndarray::DeleteOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = op.getInput();
    auto srcDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    // return failure if wrong ops or not distributed
    if (!srcDistType || !isDist(srcDistType)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto lParts = createPartsOf(loc, rewriter, src);

    // apply DeleteOp to all parts
    for (auto p : lParts) {
      auto newOp = rewriter.create<::imex::ndarray::DeleteOp>(loc, p);
      newOp->setAttrs(adaptor.getAttributes());
    }

    rewriter.eraseOp(op);

    return ::mlir::success();
  }
};

// extract RankedTensor and create ::imex::dist::AllReduceOp
inline ::imex::distruntime::AllReduceOp
createAllReduce(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                ::mlir::Attribute op, ::mlir::Value ndArray) {
  auto arType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(ndArray.getType());
  assert(arType);
  auto lArray = builder.create<::imex::ndarray::ToTensorOp>(loc, ndArray);
  auto lMRef = createToMemRef(loc, builder, lArray, arType.getMemRefType());
  return builder.create<::imex::distruntime::AllReduceOp>(loc, op, lMRef);
}

/// Rewrite ::imex::ndarray::ReductionOp to get a distributed
/// reduction if operand is distributed.
/// Create global, distributed 0d output array.
/// The local partitions of operand (e.g. RankedTensor) is wrapped in
/// non-distributed NDArray and re-applied to reduction.
/// The result is then applied to a distributed allreduce.
/// op gets replaced with global distributed array
struct ReductionOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::ReductionOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::ReductionOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ReductionOp op,
                  ::imex::ndarray::ReductionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME reduction over individual dimensions is not supported
    auto loc = op.getLoc();
    auto inp = op.getInput();
    auto inpDistTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(inp.getType());
    // nothing to do if not distributed
    if (!inpDistTyp || !isDist(inpDistTyp))
      return ::mlir::failure();

    // Local reduction
    auto parts = createPartsOf(loc, rewriter, inp);
    auto local = parts.size() == 1 ? parts[0] : parts[1];
    auto retArType = cloneAsNonDist(op.getType());
    auto redArray = rewriter.create<::imex::ndarray::ReductionOp>(
        loc, retArType, op.getOp(), local);
    // global reduction
    (void)createAllReduce(loc, rewriter, op.getOp(), redArray);

    // init our new dist array
    // FIXME result shape is 0d always
    rewriter.replaceOp(op, createDistArray(loc, rewriter,
                                           getDistEnv(inpDistTyp).getTeam(),
                                           ::mlir::SmallVector<int64_t>(), {},
                                           redArray.getResult()));
    return ::mlir::success();
  }
};

/// Rewriting ::imex::ndarray::ToTensorOp
/// Get NDArray from distributed array and apply to ToTensorOp.
struct ToTensorOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::ToTensorOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::ToTensorOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ToTensorOp op,
                  ::imex::ndarray::ToTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // get input
    auto inpArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getInput().getType());
    if (!inpArTyp || !isDist(inpArTyp)) {
      return ::mlir::failure();
    }
    auto parts = createPartsOf(loc, rewriter, op.getInput());
    auto part = parts.size() == 1 ? parts[0] : parts[1];
    rewriter.replaceOpWithNewOp<::imex::ndarray::ToTensorOp>(op, part);
    return ::mlir::success();
  }
};

/// Convert a global dist::SubviewOP to ndarray::SubviewOp on the local data.
/// Computes overlap of slice, local parts and target.
/// Even though the op accepts static offs/sizes all computation
/// is done on values - only static dim-sizes of 1 are properly propagated.
/// Static strides are always propagated to NDArray.
struct SubviewOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::SubviewOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::SubviewOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::SubviewOp op,
                  ::imex::dist::SubviewOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get input and type
    auto src = op.getSource();
    auto inpDistTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto resDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    if (!inpDistTyp || !isDist(inpDistTyp) || !resDistType ||
        !isDist(resDistType)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    // Get the local part of the global slice, team, rank, offsets
    auto _slcOffs = adaptor.getOffsets();
    auto _slcSizes = adaptor.getSizes();
    auto _slcStrides = adaptor.getStrides();
    auto sSlcOffs = adaptor.getStaticOffsets();
    auto sSlcSizes = adaptor.getStaticSizes();
    auto sSlcStrides = adaptor.getStaticStrides();
    auto tOffs = adaptor.getTargetOffsets();
    auto tSizes = adaptor.getTargetSizes();
    auto rank = std::max(sSlcOffs.size(), _slcOffs.size());
    bool hasTarget = tOffs.size() > 0;

    // get offs, sizes strides as values
    auto slcOffs = getMixedAsValues(loc, rewriter, _slcOffs, sSlcOffs);
    auto slcSizes = getMixedAsValues(loc, rewriter, _slcSizes, sSlcSizes);
    auto slcStrides = getMixedAsValues(loc, rewriter, _slcStrides, sSlcStrides);

    ::imex::ValVec lViews, lVOffsets;
    auto srcParts = createPartsOf(loc, rewriter, src);
    ::imex::ValVec lOffs = createLocalOffsetsOf(loc, rewriter, src);
    ::mlir::SmallVector<EasyIdx> shift(rank, easyIdx(loc, rewriter, 0));

    // if a target is provided, crop slice to given target
    if (hasTarget) {
      for (auto i = 0u; i < rank; ++i) {
        // remember the target offset as we need to "shift back" for the local
        // offset
        shift[i] = easyIdx(loc, rewriter, tOffs[i]);
        slcOffs[i] = (easyIdx(loc, rewriter, slcOffs[i]) +
                      (shift[i] * easyIdx(loc, rewriter, slcStrides[i])))
                         .get();
        slcSizes[i] = tSizes[i];
      }
    }

    for (auto lPart : srcParts) {
      ::imex::ValVec lSlcOffsets;
      auto pShape = createShapeOf(loc, rewriter, lPart);
      // Compute local part
      auto pOverlap = createOverlap(loc, rewriter, lOffs, pShape, slcOffs,
                                    slcSizes, slcStrides);
      auto pOffsets = std::get<0>(pOverlap);
      auto pSizes_ = std::get<1>(pOverlap);

      if (lVOffsets.size() == 0) {
        lVOffsets = std::get<2>(pOverlap);
        // "shift back" the cropped target offset
        for (auto i = 0u; i < rank; ++i) {
          lVOffsets[i] =
              (easyIdx(loc, rewriter, lVOffsets[i]) + shift[i]).get();
        }
      }

      // get static size==1 and strides back
      ::mlir::SmallVector<::mlir::OpFoldResult> pOffs, pStrides, pSizes;
      for (size_t i = 0; i < rank; ++i) {
        auto pOff_ = easyIdx(loc, rewriter, pOffsets[i]);
        auto lOff_ = easyIdx(loc, rewriter, lOffs[i]);
        auto lShp_ = easyIdx(loc, rewriter, pShape[i]);
        auto lOff = (pOff_ - lOff_).min(lShp_);
        pOffs.emplace_back(lOff.get());
        auto s = sSlcStrides[i];
        pStrides.emplace_back(
            ::mlir::ShapedType::isDynamic(s)
                ? ::mlir::OpFoldResult{slcStrides[i]}
                : ::mlir::OpFoldResult{rewriter.getIndexAttr(s)});
        // this might break broadcasting since size=1 is no longer static
        pSizes.emplace_back(pSizes_[i]);
      }

      // create local view
      lViews.emplace_back(rewriter.create<::imex::ndarray::SubviewOp>(
          loc,
          mlir::dyn_cast<::imex::ndarray::NDArrayType>(lPart.getType())
              .cloneWithDynDims(),
          lPart, pOffs, pSizes, pStrides));

      // update local offset for next part
      lOffs[0] = rewriter.createOrFold<::mlir::arith::AddIOp>(loc, lOffs[0],
                                                              pShape[0]);
    }

    // init our new dist array
    auto dEnv = getDistEnv(resDistType);
    rewriter.replaceOp(op, createDistArray(loc, rewriter, dEnv.getTeam(),
                                           slcSizes, lVOffsets, lViews));
    return ::mlir::success();
  }
};

/// Convert a global dist::InsertSliceOp to ndarray::InsertSliceOp on the local
/// data. Assumes that the input is properly partitioned: the target part or if
/// none provided the default partitioning.
struct InsertSliceOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::InsertSliceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::InsertSliceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::InsertSliceOp op,
                  ::imex::ndarray::InsertSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto destArType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(
        op.getDestination().getType());
    auto srcArType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getSource().getType());
    if (!destArType || !isDist(destArType) || !srcArType ||
        !isDist(srcArType)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto dest = op.getDestination();
    auto slcOffs = ::mlir::getMixedValues(adaptor.getStaticOffsets(),
                                          adaptor.getOffsets(), rewriter);
    auto slcSizes = ::mlir::getMixedValues(adaptor.getStaticSizes(),
                                           adaptor.getSizes(), rewriter);
    auto slcStrides = ::mlir::getMixedValues(adaptor.getStaticStrides(),
                                             adaptor.getStrides(), rewriter);

    auto srcRank = srcArType.getRank();
    auto srcParts = createPartsOf(loc, rewriter, op.getSource());
    unsigned ownPartIdx = srcParts.size() == 1 ? 0 : 1;

    // the destination is assumed to be contiguous always
    auto destParts = createPartsOf(loc, rewriter, dest);
    ::imex::ValVec destOffs = createLocalOffsetsOf(loc, rewriter, dest);
    ::mlir::Value lDest;
    // get the local part
    if (destParts.size() == 1) {
      lDest = destParts[0];
    } else {
      lDest = destParts[1];
      // if it's the second part, we need to update the offset
      auto pSizes = createShapeOf(loc, rewriter, destParts[0]);
      destOffs[0] = rewriter.createOrFold<::mlir::arith::AddIOp>(
          loc, destOffs[0], pSizes[0]);
    }

    // The parts in src are in order and together form a uniform view.
    // The view must have the same shape as the local part of dest.
    // We can just insert one part of src after the other.
    // We only need to update the off into dest's local part.

    auto destSizes = createShapeOf(loc, rewriter, lDest);
    auto destOverlap = createOverlap(loc, rewriter, destOffs, destSizes,
                                     slcOffs, slcSizes, slcStrides);
    auto lOffs = std::get<0>(destOverlap);
    auto lSizes = std::get<1>(destOverlap);
    auto lo0 =
        easyIdx(loc, rewriter, lOffs[0]) - easyIdx(loc, rewriter, destOffs[0]);
    lOffs[0] = lo0.get();

    if (srcRank) {
      for (auto srcPart : srcParts) {
        auto ary = mlir::cast<::imex::ndarray::NDArrayType>(srcPart.getType());
        if (ary.hasZeroSize()) {
          continue;
        }

        // the shape of the src part is also used for Sizes in insert_slice
        auto srcSizes =
            createShapeOf<::mlir::SmallVector<::mlir::OpFoldResult>>(
                loc, rewriter, srcPart);

        // and finally insert this view into lDest
        rewriter.create<::imex::ndarray::InsertSliceOp>(
            loc, lDest, srcPart, lOffs, srcSizes, slcStrides);

        // for the next src part we have to move the offset in our lDest
        lo0 = lo0 + (easyIdx(loc, rewriter, srcSizes[0]) *
                     easyIdx(loc, rewriter, slcStrides[0]));
        lOffs[0] = lo0.get();
      }
    } else {
      // src  is a 0d array
      auto sz = easyIdx(loc, rewriter, lSizes[0]);
      auto zero = easyIdx(loc, rewriter, 0);
      rewriter.create<::mlir::scf::IfOp>(
          loc, sz.sgt(zero).get(),
          [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
            builder.create<::imex::ndarray::InsertSliceOp>(
                loc, lDest, srcParts[ownPartIdx], lOffs, lSizes, slcStrides);
            builder.create<::mlir::scf::YieldOp>(loc);
          });
    }

    rewriter.eraseOp(op);

    return ::mlir::success();
  }
};

/// Convert a global ndarray::ReshapeOp on a distributed array
/// to ndarray::ReshapeOp on the local data.
/// If needed, adds a repartition op.
/// The local partition (e.g. a RankedTensor) is wrapped in a
/// non-distributed NDArray and re-applied to ReshapeOp.
/// op gets replaced with global distributed array
struct ReshapeOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::ReshapeOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::ReshapeOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ReshapeOp op,
                  ::imex::ndarray::ReshapeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = op.getSource();
    auto srcDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto retDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    if (!(srcDistType && isDist(srcDistType) && retDistType &&
          isDist(retDistType))) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto elType = srcDistType.getElementType();
    auto dEnv = getDistEnv(srcDistType);
    auto ngShape = adaptor.getShape();
    auto gShape = createGlobalShapeOf(loc, rewriter, src);
    auto lParts = createPartsOf(loc, rewriter, src);
    auto lArray = lParts.size() == 1 ? lParts[0] : lParts[1];
    auto lOffs = createLocalOffsetsOf(loc, rewriter, src);

    // Repartitioning is needed if any of the partitions' size is not a multiple
    // of the new chunksize.
    // For now we always copy. some initial check existed in rev 3a0b97825382b

    assert(adaptor.getCopy().value_or(1) != 0 ||
           (false && "Distributed reshape currently requires copying"));

    // FIXME: Check return type: Check that static sizes are the same as the
    // default part sizes

    auto team = dEnv.getTeam();
    auto nPart = createDefaultPartition(loc, rewriter, team, ngShape);
    auto nlOffs = nPart.getLOffsets();
    auto nlShape = nPart.getLShape();
    auto shp = getShapeFromValues(nlShape);
    auto lRetType = ::imex::ndarray::NDArrayType::get(
        shp, elType, getNonDistEnvs(retDistType));

    // call the idt runtime
    auto htype = ::imex::distruntime::AsyncHandleType::get(getContext());
    auto nlArray = rewriter.create<::imex::distruntime::CopyReshapeOp>(
        loc, ::mlir::TypeRange{htype, lRetType}, team, lArray, gShape, lOffs,
        ngShape, nlOffs, nlShape);
    (void)rewriter.create<::imex::distruntime::WaitOp>(loc,
                                                       nlArray.getHandle());
    // finally init dist array
    rewriter.replaceOp(
        op, createDistArray(loc, rewriter, team, ngShape, nlOffs,
                            ::mlir::ValueRange{nlArray.getNlArray()}));

    return ::mlir::success();
  }
};

/// Convert a global dist::EWBinOp to ndarray::EWBinOp on the local data.
/// Assumes that the partitioning of the inputs are properly aligned.
struct EWBinOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::EWBinOp> {
  using ::mlir::OpConversionPattern<::imex::dist::EWBinOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  // for lhs and rhs we generate an if-cascade which yields the view
  // of the part which overlaps the current loop-slice. with static
  // array shapes/offsets canonicalizer should eliminate
  // conditions
  void getPart(::mlir::OpBuilder &builder, ::mlir::Location loc, int64_t rank,
               const EasyIdx &zero, const ::imex::ValVec &unitStrides,
               ::mlir::ValueRange parts,
               const ::mlir::SmallVector<::imex::ValVec> &shapes,
               ::imex::EasyIdx &slcOff, const ::imex::EasyIdx &slcSz,
               const ::imex::EasyIdx &pOff, unsigned i,
               ::mlir::Value &resView) const {
    auto pSz = easyIdx(loc, builder, shapes[i][0]);
    auto vOff = slcOff - pOff;
    auto pEnd = pOff + pSz;

    auto doNext = [&](const ::imex::EasyIdx &poff,
                      unsigned j) -> ::mlir::Value {
      if (j < parts.size()) {
        this->getPart(builder, loc, rank, zero, unitStrides, parts, shapes,
                      slcOff, slcSz, poff, j, resView);
        return resView;
      } else {
        // we should never get here; create a array with recognizable shape
        builder.create<::mlir::cf::AssertOp>(
            loc, createInt(loc, builder, 0, 1),
            "could not determine overlap of loop bounds and view");
        auto arType =
            mlir::cast<::imex::ndarray::NDArrayType>(resView.getType());
        static int dbg = 47110000;
        auto x = createIndex(loc, builder, ++dbg);
        return builder
            .create<::mlir::UnrealizedConversionCastOp>(
                loc, cloneAsDynNonDist(arType), x)
            .getResult(0);
      }
    };

    // create a nested if-else-block returning a view with given args if
    // condition cond is met, and returning orig resView otherwise (accepted
    // as reference!)
    auto ary = mlir::cast<::imex::ndarray::NDArrayType>(parts[i].getType());
    if (!(ary.hasUnitSize() || ary.hasZeroSize())) {
      auto overlaps = slcOff.sge(pOff).land(slcOff.slt(pEnd));
      resView =
          builder
              .create<::mlir::scf::IfOp>(
                  loc, overlaps.get(),
                  [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
                    ::imex::ValVec vOffs(rank, zero.get());
                    vOffs[0] = vOff.get();
                    auto vShape = shapes[i];
                    vShape[0] = slcSz.get();
                    auto view = builder.create<::imex::ndarray::ExtractSliceOp>(
                        loc, parts[i], vOffs, vShape, unitStrides);
                    builder.create<::mlir::scf::YieldOp>(loc, view.getResult());
                  },
                  [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
                    builder.create<::mlir::scf::YieldOp>(loc,
                                                         doNext(pEnd, i + 1));
                  })
              .getResult(0);
    } else {
      resView = doNext(pEnd, i + 1);
    }
    ++i;
  };

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::EWBinOp op,
                  ::imex::dist::EWBinOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto lhsDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getLhs().getType());
    auto rhsDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getRhs().getType());
    auto resDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    // return failure if wrong ops or not distributed
    if (!(lhsDistType && isDist(lhsDistType) && rhsDistType &&
          isDist(rhsDistType) && resDistType && isDist(resDistType))) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto rank = resDistType.getRank();
    auto lhsRank = lhsDistType.getRank();
    auto rhsRank = rhsDistType.getRank();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto resGShape = resDistType.getShape();
    auto lhsNeedsView =
        !(lhsDistType.hasUnitSize() || lhsDistType.hasZeroSize());
    auto rhsNeedsView =
        !(rhsDistType.hasUnitSize() || rhsDistType.hasZeroSize());
    auto resArType = cloneAsDynNonDist(resDistType);
    // auto core = adaptor.getCore();

    ::imex::ValVec lhsParts, rhsParts;
    int lhsOwnIdx, rhsOwnIdx;
    // create array of parts, skip 0-sized parts
    {
      auto tmp = createPartsOf(loc, rewriter, lhs);
      lhsOwnIdx = tmp.size() == 1 ? 0 : 1;
      for (int i = 0; i < (int)tmp.size(); ++i) {
        if (mlir::cast<::imex::ndarray::NDArrayType>(tmp[i].getType())
                .hasZeroSize()) {
          if (i <= lhsOwnIdx)
            --lhsOwnIdx;
        } else {
          lhsParts.emplace_back(tmp[i]);
        }
      }
      tmp = createPartsOf(loc, rewriter, rhs);
      rhsOwnIdx = tmp.size() == 1 ? 0 : 1;
      for (int i = 0; i < (int)tmp.size(); ++i) {
        if (mlir::cast<::imex::ndarray::NDArrayType>(tmp[i].getType())
                .hasZeroSize()) {
          if (i <= rhsOwnIdx)
            --rhsOwnIdx;
        } else {
          rhsParts.emplace_back(tmp[i]);
        }
      }
    }

    if (lhsRank == 0 && rhsRank == 0) {
      assert(lhsOwnIdx >= 0 && rhsOwnIdx >= 0);
      rewriter.replaceOpWithNewOp<::imex::ndarray::EWBinOp>(
          op, resArType, adaptor.getOp(), lhsParts[lhsOwnIdx],
          rhsParts[rhsOwnIdx]);
      return ::mlir::success();
    }

    auto zero = easyIdx(loc, rewriter, 0);

    // get global shape, offsets and team
    auto dEnv = getDistEnv(lhsDistType);
    auto team = dEnv.getTeam();
    ::imex::ValVec lOffs = adaptor.getTargetOffsets();
    if (lOffs.size() == 0 && resArType.getRank()) {
      if (resDistType.hasUnitSize()) {
        lOffs = ::imex::ValVec(resDistType.getRank(), zero.get());
      } else {
        ::imex::ValVec gShape;
        for (auto d : resGShape) {
          gShape.emplace_back(createIndex(loc, rewriter, d));
        }
        auto defPart = createDefaultPartition(loc, rewriter, team, gShape);
        lOffs = defPart.getLOffsets();
      }
    }

    ::imex::ValVec lhsOffs = createLocalOffsetsOf(loc, rewriter, lhs);
    ::imex::ValVec rhsOffs = createLocalOffsetsOf(loc, rewriter, rhs);

    ::mlir::SmallVector<::imex::ValVec> lhsShapes, rhsShapes;
    ::mlir::SmallVector<EasyIdx> loopStarts(1, zero);
    auto theEnd = zero;

    if (lhsRank) { // insert bounds of lhs
      for (auto p : lhsParts) {
        auto shp = createShapeOf(loc, rewriter, p);
        if (shp.size() && !lhsDistType.hasUnitSize()) {
          loopStarts.emplace_back(loopStarts.back() +
                                  easyIdx(loc, rewriter, shp[0]));
        }
        lhsShapes.emplace_back(std::move(shp));
      }
      theEnd = theEnd.max(loopStarts.back());
    }

    if (rhsRank) { // insert bounds of rhs
      auto prev = zero;
      for (auto p : rhsParts) {
        auto shp = createShapeOf(loc, rewriter, p);
        if (shp.size() && !rhsDistType.hasUnitSize()) {
          loopStarts.emplace_back(prev + easyIdx(loc, rewriter, shp[0]));
        }
        prev = loopStarts.back();
        rhsShapes.emplace_back(std::move(shp));
      }
      theEnd = theEnd.max(loopStarts.back());
    }

    auto coreOff = ::imex::EasyIdx(loc, rewriter, ::mlir::Value{});
    auto coreEnd = coreOff;
    if (true) { // insert bounds of core loop if provided
      auto cOffs = adaptor.getCoreOffsets();
      if (cOffs.size()) {
        auto cStart = easyIdx(loc, rewriter, cOffs[0]);
        coreOff = cStart.min(theEnd);
        loopStarts.emplace_back(coreOff);
        auto cEnd = cStart + easyIdx(loc, rewriter, adaptor.getCoreSizes()[0]);
        coreEnd = cEnd.min(theEnd);
        loopStarts.emplace_back(coreEnd);
      }
    }

    // sort loops by start index
    // generate compare-and-swap operation reflecting a bubble-sort
    // because the first half and second half of the list are already sorted
    // it is sufficient to reduce the outer loops to N/2 iterations
    auto N = std::max(rhsParts.size(), lhsParts.size());
    // if we have a core, we need 2 more iterations
    if (adaptor.getCoreOffsets().size() > 0) {
      N += 2;
    }
    for (unsigned i = 0; i < N; ++i) {
      for (unsigned j = 1; j < loopStarts.size(); ++j) {
        auto a = loopStarts[j];
        auto b = loopStarts[j - 1];
        loopStarts[j - 1] = a.min(b);
        loopStarts[j] = a.max(b);
      }
    }
    ::mlir::SmallVector<std::pair<EasyIdx, EasyIdx>> loops;
    for (auto i = 1u; i < loopStarts.size(); ++i) {
      if (loopStarts[i].get() != loopStarts[i - 1].get()) {
        loops.emplace_back(std::make_pair(loopStarts[i - 1], loopStarts[i]));
      }
    }

    // FIXME broadcasting
    ::imex::ValVec resShape;
    for (unsigned i = 0; i < rank; ++i) {
      auto d = resGShape[i];
      assert(d >= 0);
      if (i) {
        resShape.emplace_back(createIndex(loc, rewriter, d));
      } else {
        resShape.emplace_back(loops.back().second.get());
      }
    }

    auto res = rewriter.create<::imex::ndarray::CreateOp>(
        loc, resShape, ::imex::ndarray::fromMLIR(resDistType.getElementType()),
        nullptr, getNonDistEnvs(resDistType));
    ::mlir::Value updatedRes = res.getResult();

    ::imex::ValVec resOffs(rank, zero.get());
    ::imex::ValVec unitStrides(rank, createIndex(loc, rewriter, 1));

    // for each loop slice, determine overlap with lhs and rhs
    // apply to ndarray::ewbinop and insert into result array
    auto createLoop = [&](const std::pair<EasyIdx, EasyIdx> &lp,
                          const ::imex::EasyVal<bool> &cond) {
      auto slcOff = lp.first;
      auto slcSz = lp.second - slcOff;

      auto ifRes = rewriter.create<::mlir::scf::IfOp>(
          loc, cond.land(slcSz.sgt(zero)).get(),
          [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
            // auto lhsOff = easyIdx(loc, rewriter, lhsOffs[0]);
            // auto rhsOff = easyIdx(loc, rewriter, rhsOffs[0]);

            auto getUnitPart =
                [&builder,
                 &loc](const ::mlir::ValueRange &parts) -> ::mlir::Value {
              auto one = easyIdx(loc, builder, 1);
              auto arType = mlir::cast<::imex::ndarray::NDArrayType>(
                  parts.front().getType());
              auto rtyp = arType.cloneWith(
                  ::mlir::SmallVector<int64_t>(arType.getRank(), 1),
                  arType.getElementType());
              auto getUnitPartImpl =
                  [&builder, &loc, &one, &parts,
                   &rtyp](unsigned i, auto getUnitPart_) -> ::mlir::Value {
                if (i == parts.size() - 1) {
                  return builder.create<::imex::ndarray::CastOp>(loc, rtyp,
                                                                 parts.back());
                }
                auto p = parts[i];
                auto dims =
                    builder.createOrFold<::imex::ndarray::DimOp>(loc, p, 0);
                auto cond = easyIdx(loc, builder, dims).eq(one);
                return builder
                    .create<::mlir::scf::IfOp>(
                        loc, cond.get(),
                        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
                          auto res = builder.create<::imex::ndarray::CastOp>(
                              loc, rtyp, p);
                          builder.create<::mlir::scf::YieldOp>(loc,
                                                               res.getResult());
                        },
                        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
                          builder.create<::mlir::scf::YieldOp>(
                              loc, getUnitPart_(i + 1, getUnitPart_));
                        })
                    .getResult(0);
              };
              return getUnitPartImpl(0, getUnitPartImpl);
            };

            ::mlir::Value lhsView = lhsParts.back();
            if (lhsRank && lhsNeedsView) {
              getPart(builder, loc, rank, zero, unitStrides, lhsParts,
                      lhsShapes, slcOff, slcSz, zero, 0, lhsView);
            } else if (lhsDistType.hasUnitSize()) {
              lhsView = getUnitPart(lhsParts);
            } else {
              assert(lhsOwnIdx >= 0);
              lhsView = lhsParts[lhsOwnIdx];
            }

            ::mlir::Value rhsView = rhsParts.back();
            if (rhsRank && rhsNeedsView) {
              getPart(builder, loc, rank, zero, unitStrides, rhsParts,
                      rhsShapes, slcOff, slcSz, zero, 0, rhsView);
            } else if (rhsDistType.hasUnitSize()) {

              rhsView = getUnitPart(rhsParts);
            } else {
              assert(lhsOwnIdx >= 0);
              rhsView = rhsParts[rhsOwnIdx];
            }

            // we can now apply the ewbinop
            auto opRes = builder.create<::imex::ndarray::EWBinOp>(
                loc, resArType, op.getOp(), lhsView, rhsView);
            // and copy the result intop the result array
            resOffs[0] = slcOff.get();
            resShape[0] = slcSz.get();
            auto resAfterInsert =
                builder.create<::imex::ndarray::ImmutableInsertSliceOp>(
                    loc, updatedRes, opRes, resOffs, resShape, unitStrides);
            builder.create<::mlir::scf::YieldOp>(loc,
                                                 resAfterInsert.getResult());
          },
          [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
            builder.create<::mlir::scf::YieldOp>(loc, updatedRes);
          });
      return ifRes.getResult(0);
    };

    // create core loop first
    auto easyTrue = ::imex::EasyVal<bool>(loc, rewriter, true);
    if (coreOff.get()) {
      updatedRes = createLoop({coreOff, coreEnd}, easyTrue);
    }

    // all other loops
    for (auto l : loops) {
      // only need this loop if not core loop
      auto cond = coreOff.get() ? coreOff.ne(l.first) : easyTrue;
      updatedRes = createLoop(l, cond);
    }

    // and init our new dist array
    rewriter.replaceOp(op, createDistArray(loc, rewriter, team, resGShape,
                                           lOffs, {updatedRes}));

    return ::mlir::success();
  }
};

/// Convert a global dist::EWUnyOp to ndarray::EWUnyOp on the local data.
/// Assumes that the partitioning of the inputs are properly aligned.
struct EWUnyOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::EWUnyOp> {
  using ::mlir::OpConversionPattern<::imex::dist::EWUnyOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::EWUnyOp op,
                  ::imex::dist::EWUnyOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = op.getSrc();
    auto srcDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto resDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    // return failure if wrong ops or not distributed
    if (!srcDistType || !isDist(srcDistType) || !resDistType ||
        !isDist(resDistType)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto resArType = cloneAsDynNonDist(resDistType);
    auto lParts = createPartsOf(loc, rewriter, src);
    auto lOffsets = createLocalOffsetsOf(loc, rewriter, src);

    ::imex::ValVec resParts;
    // go through all parts and apply unyop
    for (auto part : lParts) {
      auto res = rewriter.create<::imex::ndarray::EWUnyOp>(
          loc, resArType, adaptor.getOp(), part);
      resParts.emplace_back(res.getResult());
    }

    // get global shape
    auto gShape = resDistType.getShape();
    // and init our new dist array
    rewriter.replaceOp(op, createDistArray(loc, rewriter,
                                           getDistEnv(srcDistType).getTeam(),
                                           gShape, lOffsets, resParts));

    return ::mlir::success();
  }
};

/// Convert ndarray::CastElemTypeOp if operating on distributed arrays
struct CastElemTypeOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::CastElemTypeOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::CastElemTypeOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastElemTypeOp op,
                  ::imex::ndarray::CastElemTypeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = op.getInput();
    auto srcDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto resDistType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    // return failure if wrong ops or not distributed
    if (!srcDistType || !isDist(srcDistType) || !resDistType ||
        !isDist(resDistType)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto lParts = createPartsOf(loc, rewriter, src);
    auto lOffsets = createLocalOffsetsOf(loc, rewriter, src);

    ::imex::ValVec resParts;
    // go through all parts and apply cast
    for (auto part : lParts) {
      // infer result type: non-dist, same shape, modified elem type
      auto partType =
          mlir::dyn_cast<::imex::ndarray::NDArrayType>(part.getType());
      auto resArType = cloneAsNonDist(partType).cloneWith(
          std::nullopt, resDistType.getElementType());
      auto castOp = rewriter.create<::imex::ndarray::CastElemTypeOp>(
          loc, resArType, part);
      resParts.emplace_back(castOp.getResult());
    }

    // get global shape
    auto gShape = resDistType.getShape();
    // and init our new dist array
    rewriter.replaceOp(op, createDistArray(loc, rewriter,
                                           getDistEnv(srcDistType).getTeam(),
                                           gShape, lOffsets, resParts));

    return ::mlir::success();
  }
};

/// Replace ::imex::dist::InitDistArrayOp with unrealized_conversion_cast
/// InitDistArrayOp is a dummy op used only for propagating dist infos
struct InitDistArrayOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::InitDistArrayOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::InitDistArrayOp>::OpConversionPattern;

  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::InitDistArrayOp op,
                  ::imex::dist::InitDistArrayOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto distType =
        mlir::cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    if (!distType) {
      return ::mlir::failure();
    }
    rewriter.replaceOpWithNewOp<::imex::dist::InitDistArrayOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getLOffset(),
        adaptor.getParts());
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::PartsOfOp into respective operand of defining
/// op. We assume the defining op is a InitDistArrayOp or it was converted by a
/// unrealized_conversion_cast.
struct PartsOfOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::PartsOfOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::PartsOfOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::PartsOfOp op,
                  typename ::imex::dist::PartsOfOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto base = adaptor.getArray();
    auto distType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getArray().getType());
    if (!distType || !isDist(distType)) {
      return ::mlir::failure();
    }

    auto defOp = base.getDefiningOp();
    while (defOp && defOp->getNumOperands() == 1 &&
           ::mlir::isa<::mlir::UnrealizedConversionCastOp>(defOp)) {
      defOp = defOp->getOperand(0).getDefiningOp();
    }
    if (defOp) {
      if (auto initOp =
              ::mlir::dyn_cast<::imex::dist::InitDistArrayOp>(defOp)) {
        rewriter.replaceOp(op, initOp.getParts());
        return ::mlir::success();
      }
    }
    // not a InitDistArrayOp
    return ::mlir::failure();
  }
};

/// Convert ::imex::dist::LocalOffsetsOfOp into respective operand of defining
/// op. We assume the defining op is a InitDistArrayOp or it was converted by a
/// unrealized_conversion_cast.
struct LocalOffsetsOfOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalOffsetsOfOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalOffsetsOfOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalOffsetsOfOp op,
                  typename ::imex::dist::LocalOffsetsOfOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto base = adaptor.getArray();
    auto distType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getArray().getType());
    if (!distType || !isDist(distType)) {
      return ::mlir::failure();
    }

    // 0d array
    if (op.getNumResults() == 0) {
      assert(distType.getRank() == 0);
      rewriter.eraseOp(op);
      return ::mlir::success();
    }

    auto defOp = base.getDefiningOp();
    while (defOp && defOp->getNumOperands() == 1 &&
           ::mlir::isa<::mlir::UnrealizedConversionCastOp>(defOp)) {
      defOp = defOp->getOperand(0).getDefiningOp();
    }
    if (defOp) {
      if (auto initOp =
              ::mlir::dyn_cast<::imex::dist::InitDistArrayOp>(defOp)) {
        rewriter.replaceOp(op, initOp.getLOffset());
        return ::mlir::success();
      }
    }
    // not a InitDistArrayOp
    return ::mlir::failure();
  }
};

/// Lowering ::imex::dist::DefaultPartitionOp: Compute default partition
/// for a given shape and number of processes.
/// We currently assume evenly split data.
/// We back-fill partitions if partitions are uneven (increase last to first
/// partition in prank-order by one additional item)
struct DefaultPartitionOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::DefaultPartitionOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::DefaultPartitionOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::DefaultPartitionOp op,
                  ::imex::dist::DefaultPartitionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME: non-even partitions, ndims
    auto gShape = adaptor.getGShape();
    int64_t rank = static_cast<int64_t>(gShape.size());

    if (rank == 0) {
      rewriter.eraseOp(op);
      return ::mlir::success();
    }

    auto loc = op.getLoc();
    auto sz = easyIdx(loc, rewriter, gShape.front());
    auto np = easyIdx(loc, rewriter, adaptor.getNumProcs());
    auto pr = easyIdx(loc, rewriter, adaptor.getPRank());
    auto one = easyIdx(loc, rewriter, 1);
    auto zero = easyIdx(loc, rewriter, 0);

    // compute tile size and local size (which can be greater)
    auto rem = sz % np;
    auto tSz = sz / np;
    auto lSz = tSz + (pr + rem).sge(np).select(one, zero);
    auto lOff = (pr * tSz) + zero.max(rem - (np - pr));

    // store in result range
    ::imex::ValVec res(2 * rank, zero.get());
    res[0] = lOff.get();
    res[rank] = lSz.max(zero).get();
    for (int64_t i = 1; i < rank; ++i) {
      res[rank + i] = gShape[i];
    }

    rewriter.replaceOp(op, res);
    return ::mlir::success();
  }
};

// Compute the overlap of local data and global slice and return
// as target part (global offset/size relative to requested slice)
// Currently only dim0 is cut, hence offs/sizes of all other dims
// will be identical to the ones of the requested slice
// (e.g. same size and offset 0)
struct LocalTargetOfSliceOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalTargetOfSliceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalTargetOfSliceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalTargetOfSliceOp op,
                  ::imex::dist::LocalTargetOfSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto distType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getArray().getType());
    if (!(distType && isDist(distType)))
      return ::mlir::failure();

    auto loc = op.getLoc();
    auto src = op.getArray();
    auto slcOffs = adaptor.getOffsets();
    auto slcSizes = adaptor.getSizes();
    auto slcStrides = adaptor.getStrides();
    int64_t rank = slcOffs.size();

    // Get the local part of the global slice
    auto lOffs = createLocalOffsetsOf(loc, rewriter, src);
    auto lParts = createPartsOf(loc, rewriter, src);
    ::imex::ValVec lShape = createShapeOf(loc, rewriter, lParts.front());
    for (unsigned p = 1; p < lParts.size(); ++p) {
      auto pShape = createShapeOf(loc, rewriter, lParts[p]);
      lShape[0] = rewriter.createOrFold<::mlir::arith::AddIOp>(loc, lShape[0],
                                                               pShape[0]);
    }

    auto ovlp = createOverlap<::mlir::ValueRange, ::imex::ValVec>(
        loc, rewriter, lOffs, lShape, slcOffs, slcSizes, slcStrides);
    auto lOff = std::get<2>(ovlp);
    auto lSzs = std::get<1>(ovlp);

    ::imex::ValVec results(rank * 2, createIndex(loc, rewriter, 0));
    results[0 * rank] = lOff[0];
    results[1 * rank] = lSzs[0];

    for (auto i = 1; i < rank; ++i) {
      results[1 * rank + i] = slcSizes[i];
    }

    rewriter.replaceOp(op, results);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::LocalBoundingBoxOp
/// 1. Computes offset and sizes of the provided slice when mapped to provided
/// target.
/// 2. If a bounding box is provided, computes the bounding box for it and the
/// result of 1.
struct LocalBoundingBoxOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalBoundingBoxOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalBoundingBoxOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalBoundingBoxOp op,
                  ::imex::dist::LocalBoundingBoxOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto inner = op.getInner();
    assert(!inner);
    auto vOffs = op.getOffsets();
    auto vSizes = op.getSizes();
    auto vStrides = op.getStrides();
    auto tOffs = op.getTargetOffsets();
    auto tSizes = op.getTargetSizes();
    auto bbOffs = op.getBBOffsets();
    auto bbSizes = op.getBBSizes();
    bool hasBB = !bbOffs.empty();

    auto rank = vOffs.size();

    // min start index (among all views) for each dim followed by sizes
    ::imex::ValVec oprnds(rank * 2);
    auto one = easyIdx(loc, rewriter, 1);
    auto zero = easyIdx(loc, rewriter, 0);

    // for each dim and view compute min offset and max end
    // return min offset and size (assuming stride 1 for the bb)
    for (size_t i = 0; i < rank; ++i) {
      ::mlir::SmallVector<EasyIdx> doffs;
      ::mlir::SmallVector<EasyIdx> dends;
      auto tOff = easyIdx(loc, rewriter, tOffs[i]);
      auto tSz = easyIdx(loc, rewriter, tSizes[i]);
      auto vOff = easyIdx(loc, rewriter, vOffs[i]);
      auto vSz = easyIdx(loc, rewriter, vSizes[i]);
      auto vSt = easyIdx(loc, rewriter, vStrides[i]);

      auto ttOff = vOff + tOff * vSt;
      auto ttEnd = ttOff + (tSz * vSt) - (vSt - one);
      auto has_tSz = tSz.sgt(zero); // the target might have size 0

      auto bbSz = hasBB ? easyIdx(loc, rewriter, bbSizes[i]) : zero;
      auto has_bbSz = bbSz.sgt(zero); // BB can have size 0 if BB had tSz 0
      auto bbOff =
          hasBB ? has_bbSz.select(easyIdx(loc, rewriter, bbOffs[i]), ttOff)
                : ttOff;
      auto bbEnd = bbOff + bbSz;

      auto vEnd = vOff + (vSz * vSt) - (vSt - one); // one after last element

      auto off = has_tSz.select(vOff.max(ttOff), bbOff).min(bbOff);
      auto end = has_tSz.select(vEnd.min(ttEnd), vEnd).max(bbEnd);
      auto sz = has_tSz.select(end - off, bbSz);

      oprnds[i] = off.get();
      oprnds[i + rank] = sz.get();
    }

    rewriter.replaceOp(op, oprnds);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::LocalCoreOp
/// 1. Computes offset and sizes of the local data of src as if subviewed by
/// provided slice and mapped to provided target.
/// 2. If a local core is provided, computes the intersection with the result
/// of 1.
struct LocalCoreOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalCoreOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalCoreOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalCoreOp op,
                  ::imex::dist::LocalCoreOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = op.getArray();

    auto distType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    if (!distType || !isDist(distType))
      return ::mlir::failure();

    auto rank = distType.getRank();
    if (rank == 0) {
      rewriter.eraseOp(op);
      return ::mlir::success();
    }
    int64_t resRank = op.getResults().size() / 2;
    assert(resRank == rank);

    // local part, its offsets and shape
    auto loc = op.getLoc();
    auto lOffsets = createLocalOffsetsOf(loc, rewriter, src);
    auto lParts = createPartsOf(loc, rewriter, src);
    unsigned ownPartIdx = lParts.size() == 1 ? 0 : 1;
    ::mlir::Value lData = lParts[ownPartIdx];
    ::imex::ValVec lSizes = createShapeOf(loc, rewriter, lData);
    if (ownPartIdx) {
      ::mlir::Value lhData = lParts[0];
      ::imex::ValVec lhSizes = createShapeOf(loc, rewriter, lhData);
      lOffsets[0] = (easyIdx(loc, rewriter, lOffsets[0]) +
                     easyIdx(loc, rewriter, lhSizes[0]))
                        .get();
    }

    ::imex::ValVec oprnds(rank * 2);

    auto cOffs = op.getCoreOffsets();
    auto cSizes = op.getCoreSizes();
    auto tOffs = op.getTargetOffsets();
    auto tSizes = op.getTargetSizes();
    auto sOffs = op.getSliceOffsets();
    auto sSizes = op.getSliceSizes();
    auto sStrs = op.getSliceStrides();

    auto overlap = createOverlap<::imex::ValVec>(loc, rewriter, lOffsets,
                                                 lSizes, sOffs, sSizes, sStrs);
    auto oOffs = std::get<2>(overlap);
    auto oSizes = std::get<1>(overlap);
    auto zero = easyIdx(loc, rewriter, 0);

    // for each dim compute max offset and min end
    for (auto i = 0; i < rank; ++i) {
      auto oOff = easyIdx(loc, rewriter, oOffs[i]);
      auto oSz = easyIdx(loc, rewriter, oSizes[i]);
      auto tOff = easyIdx(loc, rewriter, tOffs[i]);
      auto tSz = easyIdx(loc, rewriter, tSizes[i]);
      auto cOff = cOffs.size() ? easyIdx(loc, rewriter, cOffs[i]) : zero;
      auto cSz = cSizes.size() ? easyIdx(loc, rewriter, cSizes[i]) : tSz;

      auto shift = tOff - oOff;
      // the updated core offset is max of old and current
      auto rOff = cOff.max(zero - shift);

      // the local remainder starting at tOff
      auto lRemain = oSz - shift;
      // the local max loop sz is
      auto lMax = lRemain - rOff;
      // the target local max loop sz is
      auto tMax = tSz - rOff;

      // the updated core size is the diff of updated core off and min end
      auto rSz = (cOff + cSz - rOff).min(lMax.min(tMax));

      oprnds[i] = rOff.get(); // cOff.max(off).get();
      oprnds[i + rank] = rSz.get();
    }

    rewriter.replaceOp(op, oprnds);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::RePartitionOp
/// Creates a new array from the input array by re-partitioning it
/// according to the target part (or default). The repartitioning
/// itself happens in a library call.
struct RePartitionOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::RePartitionOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::RePartitionOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RePartitionOp op,
                  ::imex::dist::RePartitionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto base = op.getArray();

    auto distType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(base.getType());
    if (!distType || !isDist(distType))
      return ::mlir::failure();

    auto loc = op.getLoc();
    auto rank = distType.getRank();
    ::imex::ValVec bbOffs = op.getTargetOffsets();
    ::imex::ValVec bbSizes = op.getTargetSizes();

    // Get required info from base
    auto dEnv = getDistEnv(distType);
    auto team = dEnv.getTeam();
    auto gShape = createGlobalShapeOf(loc, rewriter, base);
    auto sGShape = distType.getShape();
    auto lOffsets = createLocalOffsetsOf(loc, rewriter, base);
    auto lParts = createPartsOf(loc, rewriter, base);
    // auto rank = gShape.size();

    auto zero = easyIdx(loc, rewriter, 0);
    auto one = easyIdx(loc, rewriter, 1);

    // default target partition is balanced
    if (bbSizes.empty()) {
      if (distType.hasUnitSize()) {
        bbOffs = ::imex::ValVec(rank, zero.get());
        bbSizes = ::imex::ValVec(rank, one.get());
      } else {
        auto lPart = createDefaultPartition(loc, rewriter, team, gShape);
        bbOffs = lPart.getLOffsets();
        bbSizes = lPart.getLShape();
      }
    }

    // which is the part that we own?
    assert(lParts.size() == 1 || lParts.size() == 3 ||
           (false && "Number of local parts must be 1 or 3"));
    unsigned ownPartIdx = lParts.size() == 1 ? 0 : 1;

    // Get offsets and shapes of parts
    ::mlir::Value lData = lParts[ownPartIdx];
    ::imex::ValVec lSizes = createShapeOf(loc, rewriter, lData);

    // determine overlap of new local part, we split dim 0 only
    auto bbOff = easyIdx(loc, rewriter, bbOffs[0]);
    auto bbSize = easyIdx(loc, rewriter, bbSizes[0]);
    auto oldOff = easyIdx(loc, rewriter, lOffsets[0]);
    if (ownPartIdx) {
      auto lHShape = createShapeOf(loc, rewriter, lParts[0]);
      oldOff = oldOff + easyIdx(loc, rewriter, lHShape[0]);
    }
    auto oldSize = easyIdx(loc, rewriter, lSizes[0]);
    auto tEnd = bbOff + bbSize;
    auto oldEnd = oldOff + oldSize;
    auto ownOff = oldOff.max(bbOff);
    auto ownSize = (oldEnd.min(tEnd) - ownOff).max(zero);

    // compute left and right halo sizes, we split dim 0 only
    // FIXME device
    ::imex::ValVec lHSizes(bbSizes), rHSizes(bbSizes);
    if (distType.hasUnitSize()) {
      lHSizes[0] =
          oldSize.eq(zero).land(oldOff.sgt(zero)).select(one, zero).get();
      rHSizes[0] =
          oldSize.eq(zero).land(oldOff.sle(zero)).select(one, zero).get();
    } else {
      lHSizes[0] = (ownOff.min(tEnd) - bbOff).get();
      rHSizes[0] = (tEnd - (ownOff + ownSize)).max(zero).get();
    }

    auto upHa = rewriter.create<::imex::distruntime::GetHaloOp>(
        loc, lData, gShape, lOffsets, bbOffs, bbSizes, lHSizes, rHSizes, team);

    // create subview of local part
    ::mlir::Value ownView = lData;
    if (!distType.hasUnitSize()) {
      ::imex::ValVec vSizes = bbSizes;
      vSizes[0] = ownSize.get();
      ::imex::ValVec vOffs = bbOffs;
      vOffs[0] = (ownOff - oldOff).get();
      ::imex::ValVec unitStrides(distType.getRank(),
                                 createIndex(loc, rewriter, 1));
      ownView = rewriter.create<::imex::ndarray::SubviewOp>(
          loc, lData, vOffs, vSizes, unitStrides);
    }

    // generate call to wait for halos
    // An optimizing pass might move this to the first use of a halo part
    rewriter.create<::imex::distruntime::WaitOp>(loc, upHa.getHandle());

    // init dist array
    rewriter.replaceOp(
        op, createDistArray(loc, rewriter, team, sGShape, bbOffs,
                            {upHa.getLHalo(), ownView, upHa.getRHalo()}));

    return ::mlir::success();
  }
};

/// Convert a global ndarray::PermuteDimsOp on a distributed array
/// to ndarray::PermuteDimsOp on the local data.
/// If needed, adds a repartition op.
/// The local partition (e.g. a RankedTensor) is wrapped in a
/// non-distributed NDArray and re-applied to PermuteDimsOp.
/// op gets replaced with global distributed array
struct PermuteDimsOpConverter
    : public ::mlir::OpConversionPattern<::imex::ndarray::PermuteDimsOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ndarray::PermuteDimsOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::PermuteDimsOp op,
                  ::imex::ndarray::PermuteDimsOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = op.getSource();
    auto dst = op.getResult();
    auto srcType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto dstType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(dst.getType());
    if (!(srcType && isDist(srcType) && dstType && isDist(dstType))) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto srcEnv = getDistEnv(srcType);
    auto team = srcEnv.getTeam();
    auto elementType = srcType.getElementType();

    auto srcGShape = createGlobalShapeOf(loc, rewriter, src);
    auto srcLParts = createPartsOf(loc, rewriter, src);
    auto srcLArray = srcLParts.size() == 1 ? srcLParts[0] : srcLParts[1];
    auto srcLOffsets = createLocalOffsetsOf(loc, rewriter, src);

    auto dstGShape = createGlobalShapeOf(loc, rewriter, dst);
    auto dstLPart = createDefaultPartition(loc, rewriter, team, dstGShape);
    auto dstLOffsets = dstLPart.getLOffsets();
    auto dstLShape = dstLPart.getLShape();
    auto dstLShapeIndex = getShapeFromValues(dstLShape);
    auto dstLType = ::imex::ndarray::NDArrayType::get(
        dstLShapeIndex, elementType, getNonDistEnvs(dstType));

    // call the dist runtime
    auto handleType = ::imex::distruntime::AsyncHandleType::get(getContext());
    auto distLArray = rewriter.create<::imex::distruntime::CopyPermuteOp>(
        loc, ::mlir::TypeRange{handleType, dstLType}, team, srcLArray,
        srcGShape, srcLOffsets, dstLOffsets, dstLShape, adaptor.getAxes());
    (void)rewriter.create<::imex::distruntime::WaitOp>(loc,
                                                       distLArray.getHandle());
    // finally init dist array
    rewriter.replaceOp(
        op, createDistArray(loc, rewriter, team, dstGShape, dstLOffsets,
                            ::mlir::ValueRange{distLArray.getNlArray()}));

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct ConvertDistToStandardPass
    : public ::imex::impl::ConvertDistToStandardBase<
          ConvertDistToStandardPass> {
  ConvertDistToStandardPass() = default;

  void runOnOperation() override {
    auto &ctxt = getContext();
    ::mlir::TypeConverter typeConverter;

    // Convert unknown types to itself
    typeConverter.addConversion([](::mlir::Type type) { return type; });

    // distributed array gets converted into its individual members
    typeConverter.addConversion([&ctxt](::imex::ndarray::NDArrayType type)
                                    -> std::optional<::mlir::Type> {
      if (auto dEnv = getDistEnv(type)) {
        ::mlir::SmallVector<::mlir::Type> types;
        auto rank = type.getRank();
        if (rank) {
          for (auto pttyp : getPartsTypes(type)) {
            types.emplace_back(pttyp); // parts
          }
          auto mrTyp = ::mlir::MemRefType::get(::std::array<int64_t, 1>{rank},
                                               ::mlir::IndexType::get(&ctxt));
          types.emplace_back(mrTyp); // loffs
        } else {
          auto pts = getPartsTypes(type);
          types.emplace_back(pts[pts.size() == 1 ? 0 : 1]);
        }
        return ::mlir::TupleType::get(&ctxt, types);
      }
      return type;
    });

    auto materializeArray =
        [&](::mlir::OpBuilder &builder, ::imex::ndarray::NDArrayType type,
            ::mlir::ValueRange inputs,
            ::mlir::Location loc) -> ::mlir::Value {
      assert(inputs.size() == 1);
      auto input = inputs[0];
      auto itype = input.getType();
      auto ary = mlir::dyn_cast<::imex::ndarray::NDArrayType>(itype);
      if (type != itype && ary) {
        if (isDist(ary)) {
          assert(ary.getRank() == 0);
          auto parts = createPartsOf(loc, builder, input);
          assert(parts.size() == 1);
          return parts[0];
        } else {
          return builder.create<::imex::ndarray::CastOp>(loc, type, input)
              .getResult();
        }
      }
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };

    typeConverter.addSourceMaterialization(materializeArray);

    // we need two passes because argument materialization goes after all the
    // other conversions. The first part converts all dist stuff except
    // InitDistArrayOp which should then have no use. In the second pass we
    // erase all InitDistArrayOps

    ::mlir::ConversionTarget target(ctxt);
    target.addIllegalDialect<::imex::dist::DistDialect>();
    target.addLegalDialect<
        ::imex::distruntime::DistRuntimeDialect, ::mlir::func::FuncDialect,
        ::mlir::linalg::LinalgDialect, ::mlir::arith::ArithDialect,
        ::imex::ndarray::NDArrayDialect, ::mlir::tensor::TensorDialect,
        ::mlir::memref::MemRefDialect, ::mlir::cf::ControlFlowDialect,
        ::mlir::bufferization::BufferizationDialect,
        ::imex::region::RegionDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME

    // make sure function boundaries get converted
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<
        ::mlir::func::CallOp, ::imex::ndarray::ReshapeOp,
        ::imex::ndarray::InsertSliceOp, ::imex::ndarray::EWBinOp,
        ::imex::ndarray::EWUnyOp, ::imex::ndarray::LinSpaceOp,
        ::imex::ndarray::CreateOp, ::imex::ndarray::CopyOp,
        ::imex::ndarray::ReductionOp, ::imex::ndarray::ToTensorOp,
        ::imex::ndarray::DeleteOp, ::imex::ndarray::CastElemTypeOp,
        ::imex::region::EnvironmentRegionOp,
        ::imex::region::EnvironmentRegionYieldOp,
        ::imex::ndarray::PermuteDimsOp>(
        [&](::mlir::Operation *op) { return typeConverter.isLegal(op); });
    target.addLegalOp<::imex::dist::InitDistArrayOp>();

    // All the dist conversion patterns/rewriter
    ::mlir::RewritePatternSet patterns(&ctxt);
    // all these patterns are converted
    patterns.insert<
        LinSpaceOpConverter, CreateOpConverter, CopyOpConverter,
        ReductionOpConverter, ToTensorOpConverter, InsertSliceOpConverter,
        SubviewOpConverter, EWBinOpConverter, EWUnyOpConverter,
        LocalBoundingBoxOpConverter, LocalCoreOpConverter,
        RePartitionOpConverter, ReshapeOpConverter,
        LocalTargetOfSliceOpConverter, DefaultPartitionOpConverter,
        LocalOffsetsOfOpConverter, PartsOfOpConverter, DeleteOpConverter,
        CastElemTypeOpConverter, PermuteDimsOpConverter>(typeConverter, &ctxt);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);
    ::imex::populateRegionTypeConversionPatterns(patterns, typeConverter);

    // Let's go!
    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }

    // now remove all InitDistArrayOps
    getOperation()->walk(
        [&](::imex::dist::InitDistArrayOp op) { op->erase(); });
  }
};

} // namespace
} // namespace dist

/// Populate the given list with patterns that convert Dist to Standard
void populateDistToStandardConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass that convert Dist to Standard
std::unique_ptr<::mlir::Pass> createConvertDistToStandardPass() {
  return std::make_unique<::imex::dist::ConvertDistToStandardPass>();
}

} // namespace imex
