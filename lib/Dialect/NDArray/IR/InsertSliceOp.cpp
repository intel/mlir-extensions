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

// Build an ImmutableInsertSliceOp with mixed static and dynamic entries.
void imex::ndarray::ImmutableInsertSliceOp::build(
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

// Build an ImmutableInsertSliceOp with dynamic entries.
void imex::ndarray::ImmutableInsertSliceOp::build(
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

// Build an ImmutableInsertSliceOp with static entries.
void imex::ndarray::ImmutableInsertSliceOp::build(
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

/// Pattern to rewrite a insert_slice op with constant 0-sized input.
template <typename InsertOpTy>
class InsertSliceOpZeroFolder final
    : public mlir::OpRewritePattern<InsertOpTy> {
public:
  using mlir::OpRewritePattern<InsertOpTy>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(InsertOpTy insertSliceOp,
                  mlir::PatternRewriter &rewriter) const override {
#if 0 // FIXME
    auto srcTyp = ::mlir::dyn_cast<mlir::RankedTensorType>(
        insertSliceOp.getSource().getType());
    if (srcTyp && srcTyp.hasZeroSize()) {
      if (insertSliceOp->getNumResults() == 0) {
        rewriter.eraseOp(insertSliceOp);
      } else {
        assert(insertSliceOp->getNumResults() == 1);
        rewriter.replaceOp(insertSliceOp, insertSliceOp.getDestination());
      }
      return ::mlir::success();
    }
#endif
    return mlir::failure();
  }
};

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
    auto dstTnsrType = insertSliceOp.getDestinationType(); //.getTensorType();
    // Create the new op in canonical form.
    auto sourceTnsrType =
        mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            insertSliceOp.getSourceType().getRank(), dstTnsrType, mixedOffsets,
            mixedSizes, mixedStrides);
    auto newSourceType = sourceType.cloneWith(sourceTnsrType.getShape(),
                                              sourceTnsrType.getElementType());
    mlir::Value toInsert = insertSliceOp.getSource();
    if (newSourceType != sourceType) {
      if (newSourceType.getRank() != sourceType.getRank())
        return mlir::failure();
      mlir::OpBuilder::InsertionGuard g(rewriter);
      toInsert = rewriter.create<mlir::tensor::CastOp>(insertSliceOp.getLoc(),
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
      auto castOp = v.getDefiningOp<imex::ndarray::CastOp>();
      if (!castOp || !imex::ndarray::canFoldIntoConsumerOp(castOp))
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
      replacement = rewriter.create<imex::ndarray::CastOp>(
          insertSliceOp.getLoc(), insertSliceOp.getDestinationType(),
          replacement->getResult(0));
    }
    rewriter.replaceOp(insertSliceOp, replacement->getResults());
    return mlir::success();
  }
};

class ImmutableInsertSliceOpExtractSliceFolder final
    : public mlir::OpRewritePattern<::imex::ndarray::ImmutableInsertSliceOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::ndarray::ImmutableInsertSliceOp>::OpRewritePattern;

  // follow insert_slice chain until hitting something that's
  // not an insert_slice or an extract_slice.
  // If hitting an extract_slice return true only if all(!) extracted slices
  // do not intersect with my slice and no other op was hit.
  // In all other cases return false.
  static bool isStale(::mlir::Operation *x, llvm::ArrayRef<int64_t> myOffs,
                      llvm::ArrayRef<int64_t> mySizes,
                      llvm::ArrayRef<int64_t> myStrides) {
    if (::mlir::isa<::imex::ndarray::ImmutableInsertSliceOp>(x)) {
      for (auto u : x->getUsers()) {
        if (!isStale(u, myOffs, mySizes, myStrides))
          return false;
      }
      // none of our users is an unknown/other op and all end in
      // non-intersecting extract_slice
      return true;
    } else if (auto esOp =
                   ::mlir::dyn_cast<::imex::ndarray::ExtractSliceOp>(x)) {
      auto esOffs = esOp.getStaticOffsets();
      auto esSizes = esOp.getStaticSizes();
      auto esStrides = esOp.getStaticStrides();
      // require statically known offsets/sizeS/strides
      for (auto i = 0u; i < esOffs.size(); ++i) {
        if (esOffs[i] == ::mlir::ShapedType::kDynamic ||
            esSizes[i] == ::mlir::ShapedType::kDynamic ||
            esStrides[i] == ::mlir::ShapedType::kDynamic) {
          return false;
        }
      }

      for (auto i = 0u; i < myOffs.size(); ++i) {
        auto myOff = myOffs[i];
        auto myEnd = myOff + mySizes[i] * myStrides[i];
        auto esOff = esOffs[i];
        auto esEnd = esOff + esSizes[i] * esStrides[i];
        if (!(esOff < myEnd && esEnd > myOff && esEnd > esOff)) {
          // overwrite requires all dimensions to intersect
          // we have no overwrite if at least one dim does not intersect
          return true;
        }
      }
    }
    return false;
  }

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ImmutableInsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return mlir::failure();

    auto myOffs = op.getStaticOffsets();
    auto mySizes = op.getStaticSizes();
    auto myStrides = op.getStaticStrides();

    // require statically known offsets/sizeS/strides
    for (auto i = 0u; i < myOffs.size(); ++i) {
      if (myOffs[i] == ::mlir::ShapedType::kDynamic ||
          mySizes[i] == ::mlir::ShapedType::kDynamic ||
          myStrides[i] == ::mlir::ShapedType::kDynamic) {
        return mlir::failure();
      }
    }

    if (!isStale(*op->user_begin(), myOffs, mySizes, myStrides))
      return mlir::failure();

    rewriter.replaceOp(op, op.getDestination());
    return ::mlir::success();
  }
};

} // namespace

void imex::ndarray::InsertSliceOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<InsertSliceOpConstantArgumentFolder<InsertSliceOp>,
              InsertSliceOpCastFolder<InsertSliceOp, false>,
              InsertSliceOpZeroFolder<InsertSliceOp>>(context);
}

void imex::ndarray::ImmutableInsertSliceOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<InsertSliceOpConstantArgumentFolder<ImmutableInsertSliceOp>,
              InsertSliceOpCastFolder<ImmutableInsertSliceOp, true>,
              InsertSliceOpZeroFolder<ImmutableInsertSliceOp>,
              ImmutableInsertSliceOpExtractSliceFolder>(context);
}
