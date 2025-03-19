//===- NDArrayDist.cpp - NDArrayToDist Transform  ---------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transform of the NDArray dialect to a combination of
/// NDArray and Dist dialects.
///
/// Replace operations in NDArray if they have a shadow definition in Dist.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Transforms/Passes.h>
#include <imex/Dialect/NDArray/Transforms/Utils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

namespace imex {
#define GEN_PASS_DEF_NDARRAYDIST
#include <imex/Dialect/NDArray/Transforms/Passes.h.inc>
} // namespace imex

namespace imex {
namespace dist {

namespace {

// *******************************
// ***** Individual patterns *****
// *******************************

// match given operation if array operands and results are distributed
// and shared the same team
template <typename FROM>
struct DistOpRWP : public ::mlir::OpRewritePattern<FROM> {
  using ::mlir::OpRewritePattern<FROM>::OpRewritePattern;

  ::mlir::LogicalResult match(FROM op) const {
    DistEnvAttr dEnv;
    if (op->getNumResults() > 0) {
      auto outDisTTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(
          op->getResultTypes().front());
      if (!outDisTTyp || !isDist(outDisTTyp)) {
        return ::mlir::failure();
      } else if (outDisTTyp) {
        // to verify same teams are used with operands
        dEnv = getDistEnv(outDisTTyp);
      }
    }

    for (auto r : op->getOperands()) {
      auto opType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(r.getType());
      if (opType) {
        auto dEnv2 = getDistEnv(opType);
        if (!dEnv2) {
          // all dist operands and the return type must use the same team
          return ::mlir::failure();
        }
        if (!dEnv) {
          dEnv = dEnv2;
        } else {
          assert(dEnv2.getTeam() == dEnv.getTeam());
        }
      }
    }

    return ::mlir::success();
  }
};

struct DistSubviewOpRWP : public DistOpRWP<::imex::ndarray::SubviewOp> {
  using DistOpRWP<::imex::ndarray::SubviewOp>::DistOpRWP;
  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::SubviewOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto empty = ::mlir::ValueRange{};
    rewriter.replaceOpWithNewOp<::imex::dist::SubviewOp>(
        op, op.getType(), op.getSource(), op.getOffsets(), op.getSizes(),
        op.getStrides(), op.getStaticOffsets(), op.getStaticSizes(),
        op.getStaticStrides(), empty, empty);
    return ::mlir::success();
  }
};

struct DistEWUnyOpRWP : public DistOpRWP<::imex::ndarray::EWUnyOp> {
  using DistOpRWP<::imex::ndarray::EWUnyOp>::DistOpRWP;
  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWUnyOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto empty = ::mlir::ValueRange{};
    rewriter.replaceOpWithNewOp<::imex::dist::EWUnyOp>(
        op, op.getType(), op.getOp(), op.getSrc(), empty, empty, empty);
    return ::mlir::success();
  }
};

/// 1. Compute local slice of dst (target part)
/// 2. Repartition input to computed target part
struct DistInsertSliceOpRWP : public DistOpRWP<::imex::ndarray::InsertSliceOp> {
  using DistOpRWP<::imex::ndarray::InsertSliceOp>::DistOpRWP;
  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::InsertSliceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = mlir::cast<::imex::ndarray::NDArrayType>(src.getType());
    if (srcType.getRank() == 0 ||
        src.getDefiningOp<::imex::dist::RePartitionOp>() ||
        ::mlir::failed(match(op))) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto dst = op.getDestination();
    auto slcOffs =
        getMixedAsValues(loc, rewriter, op.getOffsets(), op.getStaticOffsets());
    auto slcSizes =
        getMixedAsValues(loc, rewriter, op.getSizes(), op.getStaticSizes());
    auto slcStrides =
        getMixedAsValues(loc, rewriter, op.getStrides(), op.getStaticStrides());

    auto tSlice = rewriter.create<::imex::dist::LocalTargetOfSliceOp>(
        loc, dst, slcOffs, slcSizes, slcStrides);
    ::mlir::ValueRange tSlcOffs = tSlice.getTOffsets();
    ::mlir::ValueRange tSlcSizes = tSlice.getTSizes();

    // Repartition source
    auto nSrc = createRePartition(loc, rewriter, src, tSlcOffs, tSlcSizes);

    rewriter.modifyOpInPlace(op, [&]() { op.getSourceMutable().set(nSrc); });
    return ::mlir::success();
  }
};

/// Rewrite ::imex::ndarray::EWBinOp to get a distributed ewbinop
/// if operands are distributed.
/// Repartitions input arrays as needed.
struct DistEWBinOpRWP : public DistOpRWP<::imex::ndarray::EWBinOp> {
  using DistOpRWP<::imex::ndarray::EWBinOp>::DistOpRWP;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWBinOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    // get inputs and types
    auto loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto lhsDistTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(lhs.getType());
    auto rhsDistTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(rhs.getType());
    auto outDistTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());

    // Repartition if necessary
    // FIXME: this breaks with dim-sizes==1, even if statically known
    auto rbLhs =
        rhs == lhs || lhsDistTyp.getRank() == 0
            ? lhs
            : createRePartition(loc, rewriter, lhs); //, tOffs, tSizes);
    auto rbRhs =
        rhs == lhs
            ? rbLhs
            : (rhsDistTyp.getRank() == 0
                   ? rhs
                   : createRePartition(loc, rewriter, rhs)); //, tOffs, tSizes);

    auto empty = ::mlir::ValueRange{};
    rewriter.replaceOpWithNewOp<::imex::dist::EWBinOp>(
        op, outDistTyp, op.getOp(), rbLhs, rbRhs, empty, empty, empty);

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct NDArrayDistPass : public ::imex::impl::NDArrayDistBase<NDArrayDistPass> {

  NDArrayDistPass() = default;

  void runOnOperation() override {

    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<DistEWBinOpRWP, DistEWUnyOpRWP, DistSubviewOpRWP,
                   DistInsertSliceOpRWP>(getContext(), patterns);
    (void)::mlir::applyPatternsGreedily(this->getOperation(), patterns);
  }
};

} // namespace
} // namespace dist

/// Populate the given list with patterns that eliminate Dist ops
void populateNDArrayDistPatterns(::mlir::LLVMTypeConverter &converter,
                                 ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createNDArrayDistPass() {
  return std::make_unique<::imex::dist::NDArrayDistPass>();
}

} // namespace imex
