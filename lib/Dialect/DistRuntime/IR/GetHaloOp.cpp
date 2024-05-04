//===- GetHaloOp.cpp - distruntime dialect  ---------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the GetHaloOp of the DistRuntime dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>

namespace imex {
namespace distruntime {

void GetHaloOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, ::mlir::Value local,
                      ::mlir::ValueRange gShape, ::mlir::ValueRange lOffsets,
                      ::mlir::ValueRange bbOffsets, ::mlir::ValueRange bbSizes,
                      ::mlir::ValueRange lHSizes, ::mlir::ValueRange rHSizes,
                      ::mlir::Attribute team, int64_t key) {
  auto lShp = getShapeFromValues(lHSizes);
  auto rShp = getShapeFromValues(rHSizes);
  auto arType = mlir::cast<::imex::ndarray::NDArrayType>(local.getType());
  auto elType = arType.getElementType();
  build(odsBuilder, odsState,
        ::imex::distruntime::AsyncHandleType::get(elType.getContext()),
        arType.cloneWith(lShp, elType), arType.cloneWith(rShp, elType), local,
        gShape, lOffsets, bbOffsets, bbSizes, team,
        odsBuilder.getI64IntegerAttr(key));
}

::mlir::SmallVector<::mlir::Value> GetHaloOp::getDependent() {
  return {getLHalo(), getRHalo()};
}

} // namespace distruntime
} // namespace imex

namespace {

/// Pattern to replace dynamically shaped result halo types
/// by statically shaped halo result types.
/// It is assumed that for unit-sized ndarrays the halo sizes have static sizes
/// always. This is a slightly complicated canonicalization because it requires
/// computing the static sizes of the halos.
class GetHaloOpResultCanonicalizer final
    : public mlir::OpRewritePattern<::imex::distruntime::GetHaloOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::distruntime::GetHaloOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::GetHaloOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    // check input type
    auto lData = op.getLocal();
    auto lType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(lData.getType());
    auto rank = lType.getRank();
    if (!lType || rank == 0)
      return ::mlir::failure();

    // local data type
    auto arType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(lData.getType());
    auto lSizes = arType.getShape();

    // if dyn type, check if this came from a CastOp
    if (::mlir::ShapedType::isDynamicShape(lSizes)) {
      if (auto defOp = lData.getDefiningOp<::imex::ndarray::CastOp>()) {
        lData = defOp.getSource();
        arType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(lData.getType());
        lSizes = arType.getShape();
      }
    }

    // Get current halos types and shapes
    auto lHType =
        mlir::cast<::imex::ndarray::NDArrayType>(op.getLHalo().getType());
    auto rHType =
        mlir::cast<::imex::ndarray::NDArrayType>(op.getRHalo().getType());
    auto lResSzs = lHType.getShape();
    auto rResSzs = rHType.getShape();
    auto lDyn = ::mlir::ShapedType::isDynamicShape(lResSzs);
    auto rDyn = ::mlir::ShapedType::isDynamicShape(rResSzs);

    // nothing to do if the result types are already static
    if (!(lDyn || rDyn)) {
      return ::mlir::failure();
    }

    // Get all dependent values needed to compute halo sizes (bb and loffsets)
    bool moded = false;
    auto lOffsets = ::imex::getShapeFromValues(op.getLOffsets());
    auto bbOffs = ::imex::getShapeFromValues(op.getBbOffsets());
    auto bbSizes = ::imex::getShapeFromValues(op.getBbSizes());

    lDyn = ::mlir::ShapedType::isDynamicShape(lResSzs[0]);
    rDyn = ::mlir::ShapedType::isDynamicShape(rResSzs[0]);

    ::mlir::SmallVector<int64_t> lHSizes(lResSzs), rHSizes(rResSzs);
    // if the first (split) dim is non-constant for any halo -> try to determine
    // their size in first dim
    if (lDyn || rDyn) {
      // all dependent values for computation
      auto bbOff = bbOffs[0];
      auto bbSize = bbSizes[0];
      auto oldOff = lOffsets[0];
      auto oldSize = lSizes[0];

      // only if all are statically known we can compute size in first dim
      if (!::mlir::ShapedType::isDynamic(bbOff) &&
          !::mlir::ShapedType::isDynamic(bbSize) &&
          !::mlir::ShapedType::isDynamic(oldOff) &&
          !::mlir::ShapedType::isDynamic(oldSize)) {
        auto tEnd = bbOff + bbSize;
        auto oldEnd = oldOff + oldSize;
        auto ownOff = std::max(oldOff, bbOff);
        auto ownSize = std::max(std::min(oldEnd, tEnd) - ownOff, 0L);

        lHSizes[0] = std::min(ownOff, tEnd) - bbOff;
        rHSizes[0] = std::max(tEnd - (ownOff + ownSize), 0L);
        moded = true;
      }
    }

    // all other dims: if not statically known already check if bb size is
    // statically known
    for (auto i = 1; i < rank; ++i) {
      if (!::mlir::ShapedType::isDynamic(bbSizes[i])) {
        if (::mlir::ShapedType::isDynamic(lHSizes[i])) {
          lHSizes[i] = bbSizes[i];
          moded = true;
        }
        if (::mlir::ShapedType::isDynamic(rHSizes[i])) {
          rHSizes[i] = bbSizes[i];
          moded = true;
        }
      }
    }

    // no new static size determined?
    if (!moded) {
      return ::mlir::failure();
    }

    // make new halo types and create new GetHaloOp
    auto elTyp = lType.getElementType();
    auto lTyp = lType.cloneWith(lHSizes, elTyp);
    auto rTyp = lType.cloneWith(rHSizes, elTyp);

    auto newOp = rewriter.create<::imex::distruntime::GetHaloOp>(
        op.getLoc(),
        ::imex::distruntime::AsyncHandleType::get(lTyp.getContext()), lTyp,
        rTyp, lData, op.getGShape(), op.getLOffsets(), op.getBbOffsets(),
        op.getBbSizes(), op.getTeamAttr(), op.getKeyAttr());

    // cast to original types and replace op
    auto lH = rewriter.create<imex::ndarray::CastOp>(op.getLoc(), lHType,
                                                     newOp.getLHalo());
    auto rH = rewriter.create<imex::ndarray::CastOp>(op.getLoc(), rHType,
                                                     newOp.getRHalo());
    rewriter.replaceOp(op, {newOp.getHandle(), lH, rH});

    return ::mlir::success();
  }
};

} // namespace

void imex::distruntime::GetHaloOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<GetHaloOpResultCanonicalizer>(context);
}
