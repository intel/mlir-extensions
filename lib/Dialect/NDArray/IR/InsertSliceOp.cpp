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
    auto dstTnsrType = insertSliceOp.getDestinationType();

    // Create the new op in canonical form.
    auto sourceTnsrType =
        mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            insertSliceOp.getSourceType().getRank(), dstTnsrType, mixedOffsets,
            mixedSizes, mixedStrides);
    auto newSourceType = sourceType.cloneWith(sourceTnsrType.getShape(),
                                              sourceTnsrType.getElementType());

    mlir::Value toInsert = insertSliceOp.getSource();
    if (newSourceType != sourceType) {
      if (sourceType.getRank() == 0) {
        if (newSourceType.getRank() > 1) {
          return mlir::failure();
        }
      } else if (newSourceType.getRank() != sourceType.getRank()) {
        return mlir::failure();
      } else {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        toInsert = rewriter.create<mlir::tensor::CastOp>(
            insertSliceOp.getLoc(), newSourceType, toInsert);
      }
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

    mlir::Operation *replacement = rewriter.create<InsertOpTy>(
        insertSliceOp.getLoc(), dst, src, insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());

    if (hasReturnValue &&
        (dst.getType() != insertSliceOp.getDestinationType())) {
      replacement = rewriter.create<mlir::tensor::CastOp>(
          insertSliceOp.getLoc(), insertSliceOp.getDestinationType(),
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
