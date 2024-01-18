//===- DimOp.cpp - NDArray dialect  --------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DimOp of the NDArray dialect.
/// Ported from NTensor.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
/// Ported from mlir::tensor dialect
mlir::Operation *imex::ndarray::NDArrayDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  if (auto op = mlir::arith::ConstantOp::materialize(builder, value, type, loc))
    return op;
  return nullptr;
}

void imex::ndarray::DimOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "dim");
}

void imex::ndarray::DimOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &result,
                                 mlir::Value source, int64_t index) {
  auto loc = result.location;
  auto indexValue = builder.create<mlir::arith::ConstantIndexOp>(loc, index);
  build(builder, result, source, indexValue);
}

std::optional<int64_t> imex::ndarray::DimOp::getConstantIndex() {
  if (auto val = mlir::getConstantIntValue(getIndex()))
    return *val;

  return {};
}

mlir::Speculation::Speculatability imex::ndarray::DimOp::getSpeculatability() {
  auto constantIndex = getConstantIndex();
  if (!constantIndex)
    return mlir::Speculation::NotSpeculatable;

  auto rankedType =
      mlir::dyn_cast<imex::ndarray::NDArrayType>(getSource().getType());
  if (!rankedType)
    return mlir::Speculation::NotSpeculatable;

  // The verifier rejects operations that violate this assertion.
  assert(constantIndex < rankedType.getRank());
  return mlir::Speculation::Speculatable;
}

/// Ported from mlir::tensor::DimOp
mlir::OpFoldResult imex::ndarray::DimOp::fold(FoldAdaptor adaptor) {
  // All forms of folding require a known index.
  auto index = llvm::dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getIndex());
  if (!index)
    return {};

  // Folding for unranked types is not supported.
  auto ndarrayType =
      llvm::dyn_cast<imex::ndarray::NDArrayType>(getSource().getType());
  if (!ndarrayType)
    return {};

  // Out of bound indices produce undefined behavior but are still valid IR.
  // Don't choke on them.
  int64_t indexVal = index.getInt();
  if (indexVal < 0 || indexVal >= ndarrayType.getRank())
    return {};

  // Fold if the shape extent along the given index is known.
  if (!ndarrayType.isDynamicDim(index.getInt())) {
    mlir::Builder builder(getContext());
    return builder.getIndexAttr(ndarrayType.getShape()[index.getInt()]);
  }

  mlir::Operation *definingOp = getSource().getDefiningOp();

  // The size at the given index is now known to be a dynamic size.
  unsigned unsignedIndex = index.getValue().getZExtValue();

  if (auto sliceOp =
          mlir::dyn_cast_or_null<imex::ndarray::SubviewOp>(definingOp)) {
    // Fold only for non-rank reduced ops. For the rank-reduced version, rely on
    // `resolve-shaped-type-result-dims` pass.
    if (sliceOp.getType().getRank() == sliceOp.getSourceType().getRank() &&
        sliceOp.isDynamicSize(unsignedIndex)) {
      return {sliceOp.getDynamicSize(unsignedIndex)};
    }
  }

  // dim(cast) -> dim
  if (succeeded(imex::ndarray::foldArrayCast(*this)))
    return getResult();

  return {};
}

namespace {
/// Fold dim of a cast into the dim of the source of the ndarray cast.
/// Ported from mlir::tensor::DimOp
struct DimOfCastOp : public mlir::OpRewritePattern<imex::ndarray::DimOp> {
  using mlir::OpRewritePattern<imex::ndarray::DimOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ndarray::DimOp dimOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto castOp = dimOp.getSource().getDefiningOp<imex::ndarray::CastOp>();
    if (!castOp)
      return mlir::failure();
    mlir::Value newSource = castOp.getOperand();
    rewriter.replaceOpWithNewOp<imex::ndarray::DimOp>(dimOp, newSource,
                                                      dimOp.getIndex());
    return mlir::success();
  }
};

// TODO: upstream
struct LinalgGenericDimPropagate
    : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto generic = src.getDefiningOp<mlir::linalg::GenericOp>();
    if (!generic)
      return mlir::failure();

    assert(generic.getOutputs().size() == generic.getResults().size());
    auto outIndex = [&]() -> size_t {
      for (auto [i, out] : llvm::enumerate(generic.getResults())) {
        if (out == src)
          return i;
      }
      llvm_unreachable("Invalid result");
    }();

    auto out = generic.getOutputs()[outIndex];

    rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(op, out, op.getIndex());
    return mlir::success();
  }
};
} // namespace

void imex::ndarray::DimOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<DimOfCastOp, LinalgGenericDimPropagate>(context);
}
