//===- InsertSliceOp.cpp - NDArray dialect  ---------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the InsertSliceOp of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>

#include <llvm/ADT/SmallBitVector.h>

#include <optional>

unsigned imex::ndarray::InsertSliceOp::getDestinationRank() {
  auto dstType = getDestination().getType();
  return mlir::dyn_cast<mlir::RankedTensorType>(dstType).getRank();
}

// Build an InsertSliceOp with mixed static and dynamic entries.
void imex::ndarray::InsertSliceOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value destination,
    mlir::Value source, mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  build(b, result, destination, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build an InsertSliceOp with dynamic entries.
void imex::ndarray::InsertSliceOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value destination,
    mlir::Value source, mlir::ValueRange offsets, mlir::ValueRange sizes,
    mlir::ValueRange strides, mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues =
      llvm::to_vector<4>(llvm::map_range(
          offsets, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(
          sizes, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues =
      llvm::to_vector<4>(llvm::map_range(
          strides, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  build(b, result, destination, source, offsetValues, sizeValues, strideValues,
        attrs);
}

// Build an InsertSliceOp with static entries.
void imex::ndarray::InsertSliceOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value destination,
    mlir::Value source, mlir::ArrayRef<int64_t> offsets,
    mlir::ArrayRef<int64_t> sizes, mlir::ArrayRef<int64_t> strides,
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
  build(b, result, destination, source, offsetValues, sizeValues, strideValues,
        attrs);
}

namespace {

/// Compute the dimensions of a rank-extending insert_slice op which are dropped
/// in its source, i.e. which of the (destination-rank many) `mixedSizes` have
/// no corresponding dimension in `reducedShape` (the source shape).
/// Ported from mlir::tensor (getDroppedDims), but reports a mismatch instead of
/// asserting so that the caller can simply bail out.
static std::optional<llvm::SmallBitVector>
computeDroppedDims(mlir::ArrayRef<int64_t> reducedShape,
                   mlir::ArrayRef<mlir::OpFoldResult> mixedSizes) {
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  int64_t shapePos = static_cast<int64_t>(reducedShape.size()) - 1;

  for (const auto &size : llvm::enumerate(llvm::reverse(mixedSizes))) {
    size_t idx = mixedSizes.size() - size.index() - 1;
    // Rank-reduced dims must have a static unit dimension.
    bool isStaticUnitSize = mlir::getConstantIntValue(size.value()) == 1;

    if (shapePos < 0) {
      // There are no more dims in the reduced shape. All remaining sizes must
      // be rank-reduced dims.
      if (!isStaticUnitSize)
        return std::nullopt;
      droppedDims.set(idx);
      continue;
    }

    // Dim is preserved if the size is not a static 1 or if the reduced shape
    // dim is also 1.
    if (!isStaticUnitSize || reducedShape[shapePos] == 1) {
      --shapePos;
      continue;
    }

    // Otherwise: Dim is dropped.
    droppedDims.set(idx);
  }

  // Dimension mismatch: not all source dims were matched.
  if (shapePos >= 0)
    return std::nullopt;
  return droppedDims;
}

/// Pattern to rewrite a insert_slice op with constant arguments.
/// Ported from mlir::tensor::InsertSliceOp
template <typename InsertOpTy>
class InsertSliceOpConstantArgumentFolder final
    : public mlir::OpRewritePattern<InsertOpTy> {
public:
  using mlir::OpRewritePattern<InsertOpTy>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(InsertOpTy insertSliceOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::OpFoldResult> mixedOffsets(
        insertSliceOp.getMixedOffsets());
    mlir::SmallVector<mlir::OpFoldResult> mixedSizes(
        insertSliceOp.getMixedSizes());
    mlir::SmallVector<mlir::OpFoldResult> mixedStrides(
        insertSliceOp.getMixedStrides());

    // No constant operands were folded, just return;
    if (mlir::failed(foldDynamicIndexList(mixedOffsets)) &&
        mlir::failed(foldDynamicIndexList(mixedSizes)) &&
        mlir::failed(foldDynamicIndexList(mixedStrides)))
      return mlir::failure();

    auto sourceType = insertSliceOp.getSourceType();

    // Create the new op in canonical form. Apply the rank transformation of
    // the original op (instead of inferring a canonical one) so that the
    // folded source type keeps matching the sliced destination dimensions.
    auto droppedDims = computeDroppedDims(sourceType.getShape(), mixedSizes);
    if (!droppedDims)
      return mlir::failure();
    auto newSourceType =
        mlir::tensor::inferSliceType(sourceType, mixedSizes, *droppedDims);

    mlir::Value toInsert = insertSliceOp.getSource();
    if (newSourceType != sourceType) {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      toInsert = mlir::tensor::CastOp::create(rewriter, insertSliceOp.getLoc(),
                                              newSourceType, toInsert);
    }

    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, insertSliceOp.getDestination(), toInsert, mixedOffsets,
        mixedSizes, mixedStrides);
    return mlir::success();
  }
};

/// Fold NDArray cast with insert_slice operations.
/// Ported from mlir::tensor::InsertSliceOp
template <typename InsertOpTy, bool hasReturnValue>
struct InsertSliceOpCastFolder final
    : public mlir::OpRewritePattern<InsertOpTy> {
  using mlir::OpRewritePattern<InsertOpTy>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(InsertOpTy insertSliceOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (llvm::any_of(insertSliceOp.getOperands(), [](mlir::Value operand) {
          return matchPattern(operand, mlir::matchConstantIndex());
        }))
      return mlir::failure();

    auto getSourceOfCastOp = [](mlir::Value v) -> std::optional<mlir::Value> {
      auto castOp = v.getDefiningOp<mlir::tensor::CastOp>();
      if (!castOp || !mlir::tensor::canFoldIntoConsumerOp(castOp))
        return std::nullopt;
      return castOp.getSource();
    };
    std::optional<mlir::Value> sourceCastSource =
        getSourceOfCastOp(insertSliceOp.getSource());
    std::optional<mlir::Value> destCastSource =
        getSourceOfCastOp(insertSliceOp.getDestination());
    if (!sourceCastSource && !destCastSource)
      return mlir::failure();

    auto src =
        (sourceCastSource ? *sourceCastSource : insertSliceOp.getSource());
    auto dst =
        (destCastSource ? *destCastSource : insertSliceOp.getDestination());

    mlir::Operation *replacement = InsertOpTy::create(
        rewriter, insertSliceOp.getLoc(), dst, src,
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());

    if (hasReturnValue &&
        (dst.getType() != insertSliceOp.getDestinationType())) {
      replacement = mlir::tensor::CastOp::create(
          rewriter, insertSliceOp.getLoc(), insertSliceOp.getDestinationType(),
          replacement->getResult(0));
    }
    rewriter.replaceOp(insertSliceOp, replacement->getResults());
    return mlir::success();
  }
};

} // namespace

void imex::ndarray::InsertSliceOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<InsertSliceOpConstantArgumentFolder<InsertSliceOp>,
              InsertSliceOpCastFolder<InsertSliceOp, false>>(context);
}
