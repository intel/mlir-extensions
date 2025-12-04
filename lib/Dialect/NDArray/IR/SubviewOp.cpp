//===- SubviewOp.cpp - NDArray dialect  -------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the SubviewOp of the NDArray dialect.
/// Copied from NTensor.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

mlir::RankedTensorType imex::ndarray::SubviewOp::inferResultType(
    mlir::RankedTensorType sourceType, mlir::ArrayRef<int64_t> staticOffsets,
    mlir::ArrayRef<int64_t> staticSizes,
    mlir::ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank && "staticOffsets length mismatch");
  assert(staticSizes.size() == rank && "staticSizes length mismatch");
  assert(staticStrides.size() == rank && "staticStrides length mismatch");
  return mlir::cast<mlir::RankedTensorType>(
      sourceType.cloneWith(staticSizes, sourceType.getElementType()));
}

mlir::RankedTensorType imex::ndarray::SubviewOp::inferResultType(
    mlir::RankedTensorType sourceShapedTensorType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubviewOp::inferResultType(sourceShapedTensorType, staticOffsets,
                                    staticSizes, staticStrides);
}

mlir::RankedTensorType imex::ndarray::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, mlir::RankedTensorType sourceType,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides) {
  auto inferredType = inferResultType(sourceType, offsets, sizes, strides);
  assert(inferredType.getRank() >= static_cast<int64_t>(resultShape.size()) &&
         "expected ");
  if (inferredType.getRank() == static_cast<int64_t>(resultShape.size()))
    return inferredType;

  assert(mlir::computeRankReductionMask(inferredType.getShape(), resultShape)
             .has_value() &&
         "invalid rank reduction");

  return mlir::cast<mlir::RankedTensorType>(
      sourceType.cloneWith(resultShape, sourceType.getElementType()));
}

mlir::RankedTensorType imex::ndarray::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, mlir::RankedTensorType sourceType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubviewOp::inferRankReducedResultType(
      resultShape, sourceType, staticOffsets, staticSizes, staticStrides);
}

/// Returns the type of the base tensor operand.
::mlir::ShapedType imex::ndarray::SubviewOp::getSourceType() {
  return mlir::dyn_cast<::mlir::ShapedType>(getSource().getType());
}

// Build a SubViewOp with mixed static and dynamic entries and custom result
// type. If the type passed is nullptr, it is inferred.
void imex::ndarray::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    mlir::RankedTensorType resultType, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceType = mlir::cast<mlir::RankedTensorType>(source.getType());
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType = imex::ndarray::SubviewOp::inferResultType(
        sourceType, staticOffsets, staticSizes, staticStrides);
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a SubViewOp with mixed static and dynamic entries and inferred result
// type.
void imex::ndarray::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, mlir::RankedTensorType(), source, offsets, sizes, strides,
        attrs);
}

// Build a SubViewOp with static entries and inferred result type.
void imex::ndarray::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, source, offsetValues, sizeValues, strideValues, attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the
// type passed is nullptr, it is inferred.
void imex::ndarray::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    mlir::RankedTensorType resultType, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues,
        attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the type
// passed is nullptr, it is inferred.
void imex::ndarray::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    mlir::RankedTensorType resultType, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues =
      llvm::to_vector<4>(llvm::map_range(
          offsets, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(
          sizes, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues =
      llvm::to_vector<4>(llvm::map_range(
          strides, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

// Build a SubViewOp with dynamic entries and inferred result type.
void imex::ndarray::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, mlir::RankedTensorType(), source, offsets, sizes, strides,
        attrs);
}

// Copypasted from upstream tensor.
llvm::SmallBitVector imex::ndarray::SubviewOp::getDroppedDims() {
  mlir::ArrayRef<int64_t> resultShape = getType().getShape();
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  unsigned shapePos = 0;
  for (const auto &size : enumerate(mixedSizes)) {
    std::optional<int64_t> sizeVal = getConstantIntValue(size.value());
    // If the size is not 1, or if the current matched dimension of the result
    // is the same static shape as the size value (which is 1), then the
    // dimension is preserved.
    if (!sizeVal || *sizeVal != 1 ||
        (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
      shapePos++;
      continue;
    }
    droppedDims.set(size.index());
  }
  return droppedDims;
}

namespace {
/// Pattern to rewrite a subview op with CastOp arguments.
/// Ported from mlir::tensor::ExtractSliceOp
template <typename SubviewOpTy>
class SubviewCastFolder final : public mlir::OpRewritePattern<SubviewOpTy> {
public:
  using mlir::OpRewritePattern<SubviewOpTy>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SubviewOpTy sliceOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Any constant operand, just return to let the constant folder kick in.
    if (llvm::any_of(sliceOp.getOperands(), [](mlir::Value operand) {
          return mlir::matchPattern(operand, mlir::matchConstantIndex());
        }))
      return mlir::failure();

    auto defOp = sliceOp.getSource().getDefiningOp();
    if (!defOp)
      return mlir::failure();
    auto castOp = mlir::dyn_cast<::mlir::tensor::CastOp>(defOp);
    if (!castOp)
      return mlir::failure();

    if (!mlir::tensor::canFoldIntoConsumerOp(castOp))
      return mlir::failure();

    // Create folded extract.
    mlir::Location loc = sliceOp.getLoc();
    mlir::Value newResult = SubviewOpTy::create(
        rewriter, loc, sliceOp.getType(), castOp.getSource(),
        sliceOp.getOffsets(), sliceOp.getSizes(), sliceOp.getStrides(),
        sliceOp.getStaticOffsets(), sliceOp.getStaticSizes(),
        sliceOp.getStaticStrides());
    if (newResult.getType() != sliceOp.getType())
      newResult = ::mlir::tensor::CastOp::create(rewriter, loc,
                                                 sliceOp.getType(), newResult);
    rewriter.replaceOp(sliceOp, newResult);
    return mlir::success();
  }
};

/// Slice elements from `values` into `outValues`. `counts` represents the
/// numbers of elements to stride in the original values for each dimension.
/// The output values can be used to construct a DenseElementsAttr.
/// Ported from mlir::tensor::ExtractSliceOp
template <typename IterTy, typename ElemTy>
static void sliceElements(IterTy values, mlir::ArrayRef<int64_t> counts,
                          mlir::ArrayRef<int64_t> offsets,
                          mlir::ArrayRef<int64_t> sizes,
                          mlir::ArrayRef<int64_t> strides,
                          llvm::SmallVectorImpl<ElemTy> *outValues) {
  assert(offsets.size() == sizes.size());
  assert(offsets.size() == strides.size());
  if (offsets.empty())
    return;

  int64_t offset = offsets.front();
  int64_t size = sizes.front();
  int64_t stride = strides.front();
  if (offsets.size() == 1) {
    for (int64_t i = 0; i < size; ++i, offset += stride)
      outValues->push_back(*(values + offset));
    return;
  }

  for (int64_t i = 0; i < size; ++i, offset += stride) {
    auto begin = values + offset * counts.front();
    sliceElements<IterTy, ElemTy>(begin, counts.drop_front(),
                                  offsets.drop_front(), sizes.drop_front(),
                                  strides.drop_front(), outValues);
  }
}

} // namespace

/// Return the canonical type of the result of an subview op.
/// Ported from mlir::tensor::ExtractSliceOp
template <typename SubviewOpTy> struct SliceReturnTypeCanonicalizer {
  mlir::RankedTensorType
  operator()(SubviewOpTy op, mlir::ArrayRef<mlir::OpFoldResult> mixedOffsets,
             mlir::ArrayRef<mlir::OpFoldResult> mixedSizes,
             mlir::ArrayRef<mlir::OpFoldResult> mixedStrides) {
    auto sourceType =
        mlir::cast<mlir::RankedTensorType>(op.getSource().getType());
    return imex::ndarray::SubviewOp::inferRankReducedResultType(
        op.getType().getShape(), sourceType, mixedOffsets, mixedSizes,
        mixedStrides);
  }
};

/// A canonicalizer wrapper to replace SubviewOps.
/// Ported from mlir::tensor::ExtractSliceOp
template <typename SubviewOpTy> struct SliceCanonicalizer {
  void operator()(mlir::PatternRewriter &rewriter, SubviewOpTy op,
                  SubviewOpTy newOp) {
    mlir::Value replacement = newOp.getResult();
    if (replacement.getType() != op.getType())
      replacement = ::mlir::tensor::CastOp::create(rewriter, op.getLoc(),
                                                   op.getType(), replacement);
    rewriter.replaceOp(op, replacement);
  }
};

void imex::ndarray::SubviewOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<mlir::OpWithOffsetSizesAndStridesConstantArgumentFolder<
                  imex::ndarray::SubviewOp,
                  SliceReturnTypeCanonicalizer<imex::ndarray::SubviewOp>,
                  SliceCanonicalizer<imex::ndarray::SubviewOp>>,
              SubviewCastFolder<imex::ndarray::SubviewOp>>(context);
}
