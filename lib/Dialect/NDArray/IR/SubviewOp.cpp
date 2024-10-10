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

imex::ndarray::NDArrayType imex::ndarray::SubviewOp::inferResultType(
    imex::ndarray::NDArrayType sourceType,
    mlir::ArrayRef<int64_t> staticOffsets, mlir::ArrayRef<int64_t> staticSizes,
    mlir::ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank && "staticOffsets length mismatch");
  assert(staticSizes.size() == rank && "staticSizes length mismatch");
  assert(staticStrides.size() == rank && "staticStrides length mismatch");
  return mlir::cast<imex::ndarray::NDArrayType>(
      sourceType.cloneWith(staticSizes, sourceType.getElementType()));
}

imex::ndarray::NDArrayType imex::ndarray::SubviewOp::inferResultType(
    imex::ndarray::NDArrayType sourceShapedTensorType,
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

imex::ndarray::NDArrayType imex::ndarray::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, imex::ndarray::NDArrayType sourceType,
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

  return mlir::cast<imex::ndarray::NDArrayType>(
      sourceType.cloneWith(resultShape, sourceType.getElementType()));
}

imex::ndarray::NDArrayType imex::ndarray::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, imex::ndarray::NDArrayType sourceType,
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
    imex::ndarray::NDArrayType resultType, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceType = mlir::cast<imex::ndarray::NDArrayType>(source.getType());
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
  build(b, result, imex::ndarray::NDArrayType(), source, offsets, sizes,
        strides, attrs);
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
    imex::ndarray::NDArrayType resultType, mlir::Value source,
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
    imex::ndarray::NDArrayType resultType, mlir::Value source,
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
  build(b, result, imex::ndarray::NDArrayType(), source, offsets, sizes,
        strides, attrs);
}

// Build a ExtractSliceOp with mixed static and dynamic entries and custom
// result type. If the type passed is nullptr, it is inferred.
void imex::ndarray::ExtractSliceOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ndarray::NDArrayType resultType, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceType = mlir::cast<imex::ndarray::NDArrayType>(source.getType());
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

// Build a ExtractSliceOp with dynamic entries and custom result type. If the
// type passed is nullptr, it is inferred.
void imex::ndarray::ExtractSliceOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ndarray::NDArrayType resultType, mlir::Value source,
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

// Build a ExtractSliceOp with dynamic entries and inferred result type.
void imex::ndarray::ExtractSliceOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, imex::ndarray::NDArrayType(), source, offsets, sizes,
        strides, attrs);
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

// static bool isIdentitySubview(imex::ndarray::SubviewOp op) {
//   auto srcType =
//   op.getSource().getType().cast<imex::ndarray::NDArrayType>(); if (srcType
//   != op.getResult().getType())
//     return false;

//   for (auto val : op.getMixedOffsets())
//     if (!mlir::isConstantIntValue(val, 0))
//       return false;

//   auto srcShape = srcType.getShape();
//   for (auto [i, val] : llvm::enumerate(op.getMixedSizes())) {
//     assert(i < srcShape.size());
//     auto shapeVal = srcShape[i];
//     if (mlir::ShapedType::isDynamic(shapeVal)) {
//       auto dim = val.dyn_cast<mlir::Value>();
//       if (!dim)
//         return false;

//       auto dimOp = dim.getDefiningOp<imex::ndarray::DimOp>();
//       if (!dimOp)
//         return false;

//       auto dimInd = dimOp.getConstantIndex();
//       if (!dimInd || *dimInd != static_cast<int64_t>(i))
//         return false;
//     } else {
//       if (!mlir::isConstantIntValue(val, shapeVal))
//         return false;
//     }
//   }

//   for (auto val : op.getMixedStrides())
//     if (!mlir::isConstantIntValue(val, 1))
//       return false;

//   return true;
// }

// mlir::OpFoldResult
// imex::ndarray::SubviewOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/)
// {
//   if (isIdentitySubview(*this))
//     return getSource();

//   return nullptr;
// }

// Copypasted from upstream tensor.
mlir::LogicalResult imex::ndarray::SubviewOp::reifyResultShapes(
    mlir::OpBuilder &builder,
    mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0].reserve(getType().getRank());
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims = getDroppedDims();
  mlir::Location loc = getLoc();
  for (const auto &size : enumerate(mixedSizes)) {
    if (droppedDims.test(size.index()))
      continue;
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(size.value())) {
      reifiedReturnShapes[0].push_back(
          builder.createOrFold<mlir::arith::ConstantIndexOp>(
              loc, mlir::cast<mlir::IntegerAttr>(attr).getInt()));
      continue;
    }
    reifiedReturnShapes[0].push_back(size.value().get<mlir::Value>());
  }
  return mlir::success();
}

namespace {
// Find source-chains of immutable_insert_slices and return matching
// source or null. Checks chains of immutable_insert_slice if they overwrite
// elements and stop following to source if so. Also stops if strides in such a
// chain are non-unit strides.
::mlir::Value
findImmutableInsertSliceSource(::imex::ndarray::ImmutableInsertSliceOp iisOp,
                               size_t rank, llvm::ArrayRef<int64_t> myOffs,
                               llvm::ArrayRef<int64_t> mySizes,
                               llvm::ArrayRef<int64_t> myStrides) {

  ::mlir::SmallVector<llvm::ArrayRef<int64_t>> laterOffs;
  ::mlir::SmallVector<llvm::ArrayRef<int64_t>> laterSizes;

  while (true) {
    if (!iisOp) {
      return {};
    }

    auto iisOffs = iisOp.getStaticOffsets();
    auto iisSizes = iisOp.getStaticSizes();
    auto iisStrides = iisOp.getStaticStrides();

    auto curr = iisOp;
    auto tmp = iisOp.getDestination().getDefiningOp();
    iisOp = tmp ? mlir::dyn_cast<::imex::ndarray::ImmutableInsertSliceOp>(tmp)
                : ::imex::ndarray::ImmutableInsertSliceOp();

    bool same = true;
    for (auto i = 0u; i < rank && same; ++i) {
      if (iisOffs[i] != myOffs[i] || iisSizes[i] != mySizes[i]) {
        same = false;
      }
    }

    // Check that we are not writing to the same elements as a later
    // insert_slice (which is downstream a consumer of me) Currently we support
    // unitstrides only and cancel otherwise
    bool okstrided = true;
    for (auto i = 0u; i < rank && okstrided; ++i) {
      if (iisStrides[i] != 1) {
        okstrided = false;
      }
    }

    // determine if our start is within any of the following insert_slices and
    // we actually write something
    auto laterN = laterOffs.size();
    if (okstrided) {
      assert(laterSizes.size() == laterN);
      for (auto p = 0u; p < laterN; ++p) {
        auto lOffs = laterOffs[p];
        auto lSizes = laterSizes[p];
        bool overwrite = true;
        for (auto i = 0u; i < rank && overwrite; ++i) {
          auto lOff = lOffs[i];
          auto lEnd = lOff + lSizes[i];
          auto iOff = iisOffs[i];
          auto iEnd = iOff + iisSizes[i];
          if (!(iOff < lEnd && iEnd > lOff && iEnd > iOff)) {
            // overwrite requires all dimensions to intersect
            // we have no overwrite if at least one dim does not intersect
            overwrite = false;
          }
        }
        if (overwrite) { // if we have an overwrite, we cannot fold
          return {};
        }
      }
    } else if (laterN == 0) {
      okstrided = true;
      for (auto i = 0u; i < rank && okstrided; ++i) {
        if (iisStrides[i] != myStrides[i]) {
          okstrided = false;
        }
      }
    }

    // Whe we reach here we know that we have no duplicate write

    // we are done if the slices match and with no overwrite
    if (same && okstrided) {
      return curr.getSource();
    }

    // we can continue with the previous insert_slice (iisOp)
    laterOffs.emplace_back(iisOffs);
    laterSizes.emplace_back(iisSizes);
  };
}
} // namespace

class ExtractSliceFolder final
    : public mlir::OpRewritePattern<::imex::ndarray::ExtractSliceOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::ndarray::ExtractSliceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ExtractSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto defOp = src.getDefiningOp();
    if (!defOp) {
      return mlir::failure();
    }
    auto iisOp = mlir::dyn_cast<::imex::ndarray::ImmutableInsertSliceOp>(defOp);
    if (!iisOp) {
      return mlir::failure();
    }

    size_t rank =
        mlir::cast<::imex::ndarray::NDArrayType>(src.getType()).getRank();
    auto myOffs = op.getStaticOffsets();
    auto mySizes = op.getStaticSizes();
    auto myStrides = op.getStaticStrides();

    for (auto i = 0u; i < rank; ++i) {
      if (myOffs[i] == ::mlir::ShapedType::kDynamic ||
          mySizes[i] == ::mlir::ShapedType::kDynamic ||
          myStrides[i] == ::mlir::ShapedType::kDynamic) {
        return mlir::failure();
      }
    }

    if (auto res = findImmutableInsertSliceSource(iisOp, rank, myOffs, mySizes,
                                                  myStrides)) {
      rewriter.replaceOp(op, res);
      return mlir::success();
    }
    return mlir::failure();
  }
};

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
    auto castOp = mlir::dyn_cast<imex::ndarray::CastOp>(defOp);
    if (!castOp)
      return mlir::failure();

    if (!imex::ndarray::canFoldIntoConsumerOp(castOp))
      return mlir::failure();

    // Create folded extract.
    mlir::Location loc = sliceOp.getLoc();
    mlir::Value newResult = rewriter.create<SubviewOpTy>(
        loc, sliceOp.getType(), castOp.getSource(), sliceOp.getOffsets(),
        sliceOp.getSizes(), sliceOp.getStrides(), sliceOp.getStaticOffsets(),
        sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
    if (newResult.getType() != sliceOp.getType())
      newResult = rewriter.create<imex::ndarray::CastOp>(loc, sliceOp.getType(),
                                                         newResult);
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
  imex::ndarray::NDArrayType
  operator()(SubviewOpTy op, mlir::ArrayRef<mlir::OpFoldResult> mixedOffsets,
             mlir::ArrayRef<mlir::OpFoldResult> mixedSizes,
             mlir::ArrayRef<mlir::OpFoldResult> mixedStrides) {
    auto sourceType =
        mlir::cast<imex::ndarray::NDArrayType>(op.getSource().getType());
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
      replacement = rewriter.create<imex::ndarray::CastOp>(
          op.getLoc(), op.getType(), replacement);
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

void imex::ndarray::ExtractSliceOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<mlir::OpWithOffsetSizesAndStridesConstantArgumentFolder<
                  imex::ndarray::ExtractSliceOp,
                  SliceReturnTypeCanonicalizer<imex::ndarray::ExtractSliceOp>,
                  SliceCanonicalizer<imex::ndarray::ExtractSliceOp>>,
              SubviewCastFolder<imex::ndarray::ExtractSliceOp>,
              ExtractSliceFolder>(context);
}
