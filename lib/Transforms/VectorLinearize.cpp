//===- VectorLinearize.cpp - VectorLinearize Pass  --------------*- C++- *-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains VectorLinearize pass.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

#include "imex/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdint>
#include <numeric>

namespace imex {
#define GEN_PASS_DEF_VECTORLINEARIZE
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {

// Cloned from upstream with isLessThanTargetBitWidth check removed.
struct ConstantOpConversion final
    : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp constOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resType =
        getTypeConverter()->convertType<mlir::VectorType>(constOp.getType());

    if (resType.isScalable() &&
        !mlir::isa<mlir::SplatElementsAttr>(constOp.getValue()))
      return rewriter.notifyMatchFailure(
          constOp,
          "Cannot linearize a constant scalable vector that's not a splat");

    if (!resType)
      return rewriter.notifyMatchFailure(constOp, "can't convert return type");
    auto dstElementsAttr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!dstElementsAttr)
      return rewriter.notifyMatchFailure(constOp, "unsupported attr type");

    dstElementsAttr = dstElementsAttr.reshape(resType);
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(constOp, resType,
                                                         dstElementsAttr);
    return mlir::success();
  }
};

// Cloned from upstream with isLessThanTargetBitWidth check removed.
struct VectorizableOpConversion final
    : public mlir::OpTraitConversionPattern<mlir::OpTrait::Vectorizable> {
  using OpTraitConversionPattern::OpTraitConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FailureOr<mlir::Operation *> newOp =
        convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return mlir::failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return mlir::success();
  }
};

// Cloned from upstream with isLessThanTargetBitWidth check removed.
static void populateVectorLinearizeTypeConversionsAndLegality(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {

  typeConverter.addConversion(
      [](mlir::VectorType type) -> std::optional<mlir::Type> {
        if (!mlir::vector::isLinearizableVector(type))
          return type;

        return mlir::VectorType::get(type.getNumElements(),
                                     type.getElementType(), type.isScalable());
      });

  auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type type,
                            mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1 ||
        !mlir::isa<mlir::VectorType>(inputs.front().getType()) ||
        !mlir::isa<mlir::VectorType>(type))
      return nullptr;

    return builder.createOrFold<mlir::vector::ShapeCastOp>(loc, type,
                                                           inputs.front());
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);
  target.markUnknownOpDynamicallyLegal(
      [=](mlir::Operation *op) -> std::optional<bool> {
        if ((mlir::isa<mlir::arith::ConstantOp>(op) ||
             op->hasTrait<mlir::OpTrait::Vectorizable>())) {
          return typeConverter.isLegal(op);
        }
        if (auto l = mlir::dyn_cast<mlir::LoopLikeOpInterface>(op)) {
          for (auto arg : l.getRegionIterArgs()) {
            if (!typeConverter.isLegal(arg.getType()))
              return false;
          }
          return true;
        }
        return std::nullopt;
      });

  patterns.add<ConstantOpConversion, VectorizableOpConversion>(
      typeConverter, patterns.getContext());
}

struct VectorLoadOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::LoadOp> {
  using mlir::OpConversionPattern<mlir::vector::LoadOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::LoadOp loadOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp->getLoc();
    auto vecType = loadOp.getVectorType();
    auto shape = vecType.getShape();

    if (shape.size() != 2) {
      return rewriter.notifyMatchFailure(loc, "Can only linearize 2D vectors.");
    }
    auto unrollCount = shape[0];
    auto vecSize = shape[1];
    auto newVecType =
        mlir::VectorType::get({vecSize}, vecType.getElementType());

    llvm::SmallVector<mlir::Value, 4> indices = adaptor.getIndices();
    mlir::Value xBaseIndex = indices[0];

    // Construct the 2D vector.
    mlir::Value resultVec = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(vecType));
    // Emit unrolled loads for each 1D vector slice.
    for (auto i = 0; i < unrollCount; i++) {
      mlir::Value xIndex = xBaseIndex;
      if (i) {
        auto increment = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
        xIndex =
            rewriter.create<mlir::arith::AddIOp>(loc, xBaseIndex, increment);
      }
      indices[0] = xIndex;
      auto vec = rewriter.create<mlir::vector::LoadOp>(
          loc, newVecType, adaptor.getBase(), indices);
      resultVec =
          rewriter.create<mlir::vector::InsertOp>(loc, vec, resultVec, i);
    }

    rewriter.replaceOp(loadOp, resultVec);
    return mlir::success();
  }
};

struct VectorStoreOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::StoreOp> {
  using mlir::OpConversionPattern<mlir::vector::StoreOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::StoreOp storeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp->getLoc();
    auto vecType = storeOp.getVectorType();
    auto shape = vecType.getShape();

    if (shape.size() != 2) {
      return rewriter.notifyMatchFailure(loc, "Can only linearize 2D vectors.");
    }

    auto unrollCount = shape[0];
    llvm::SmallVector<mlir::Value, 4> indices = adaptor.getIndices();
    mlir::Value xBaseIndex = indices[0];

    auto vec = rewriter.create<mlir::vector::ShapeCastOp>(
        loc, vecType, adaptor.getValueToStore());

    for (auto i = 0; i < unrollCount; i++) {
      auto vecSlice = rewriter.create<mlir::vector::ExtractOp>(loc, vec, i);
      mlir::Value xIndex = xBaseIndex;
      if (i) {
        auto increment = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
        xIndex =
            rewriter.create<mlir::arith::AddIOp>(loc, xBaseIndex, increment);
      }
      indices[0] = xIndex;
      rewriter.create<mlir::vector::StoreOp>(loc, vecSlice, adaptor.getBase(),
                                             indices);
    }
    rewriter.eraseOp(storeOp);
    return mlir::success();
  }
};

struct VectorExtractStridedSliceConversion final
    : public mlir::OpConversionPattern<mlir::vector::ExtractStridedSliceOp> {
  using mlir::OpConversionPattern<
      mlir::vector::ExtractStridedSliceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::ExtractStridedSliceOp extractOp,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(extractOp.getType());
    auto loc = extractOp.getLoc();
    if (!dstType)
      return rewriter.notifyMatchFailure(loc, "cannot convert type.");

    if (extractOp.getVector().getType().isScalable() ||
        mlir::cast<mlir::VectorType>(dstType).isScalable())
      return rewriter.notifyMatchFailure(loc,
                                         "scalable vectors are not supported.");
    auto offsets = extractOp.getOffsets().getValue();
    auto sizes = extractOp.getSizes().getValue();
    auto strides = extractOp.getStrides().getValue();

    if (!mlir::isConstantIntValue(strides[0], 1))
      return rewriter.notifyMatchFailure(
          extractOp, "Strided slice with stride != 1 is not supported.");

    mlir::Value srcVector = adaptor.getVector();

    // if kD offsets are specified for nd source vector (n > k), the granularity
    // of the extraction is greater than 1. In this case last (n-k) dimensions
    // form the extraction granularity. example : %0 =
    // vector.extract_strided_slice %src { offsets = [0, 0], sizes = [2, 2],
    // strides = [1, 1]} : vector<4x8x8xf32> to vector<2x2x8xf32>
    // here, extraction granularity is 8.
    int64_t extractSliceLen = 1;
    auto n = extractOp.getSourceVectorType().getRank();
    auto k = static_cast<int64_t>(offsets.size());
    if (n > k) {
      for (unsigned i = 0; i < n - k; i++) {
        extractSliceLen *= extractOp.getSourceVectorType().getShape()[i + k];
      }
    }

    // get total number of extracted slices
    int64_t nExtractedSlices = 1;
    for (auto size : sizes) {
      nExtractedSlices *= mlir::cast<mlir::IntegerAttr>(size).getInt();
    }

    // compute the strides of the source vector considering first k dimensions
    llvm::SmallVector<int64_t, 4> sourceStrides(k, extractSliceLen);
    for (int i = k - 2; i >= 0; --i) {
      sourceStrides[i] = sourceStrides[i + 1] *
                         extractOp.getSourceVectorType().getShape()[i + 1];
    }
    // final shuffle indices has nExtractedElems * extractSliceLen elements
    llvm::SmallVector<int64_t, 4> indices(nExtractedSlices * extractSliceLen);
    // compute the strides of the extracted kD vector
    llvm::SmallVector<int64_t, 4> extractedStrides(k, 1);
    // compute extractedStrides
    for (int i = k - 2; i >= 0; --i) {
      extractedStrides[i] =
          extractedStrides[i + 1] *
          mlir::cast<mlir::IntegerAttr>(sizes[i + 1]).getInt();
    }
    // iterate over all extracted slices from 0 to nExtractedElems-1
    // and compute the multi-dimensional index and the corresponding linearized
    // index within the source vector
    for (int64_t i = 0; i < nExtractedSlices; ++i) {
      int64_t index = i;
      // compute the corresponding multi-dimensional index
      llvm::SmallVector<int64_t, 4> multiDimIndex(k, 0);
      for (int64_t j = 0; j < k; ++j) {
        multiDimIndex[j] = (index / extractedStrides[j]);
        index -= multiDimIndex[j] * extractedStrides[j];
      }
      // compute the corresponding linearized index in the source vector
      // i.e. shift the multiDimIndex by the offsets
      int64_t linearizedIndex = 0;
      for (int64_t j = 0; j < k; ++j) {
        linearizedIndex += (mlir::cast<mlir::IntegerAttr>(offsets[j]).getInt() +
                            multiDimIndex[j]) *
                           sourceStrides[j];
      }
      // fill the indices array form linearizedIndex to linearizedIndex +
      // sliceLen
      for (int64_t j = 0; j < extractSliceLen; ++j) {
        indices[i * extractSliceLen + j] = linearizedIndex + j;
      }
    }

    // If indices just has one element, we will continue to use
    // ExtractStridedSliceOp. Avoid using vector.shuffle on <1xT>
    // vector, as vector-to-spirv pass does not handle it well.
    if (indices.size() == 1) {
      int64_t sizes[] = {1};
      int64_t strides[] = {1};
      rewriter.replaceOpWithNewOp<mlir::vector::ExtractStridedSliceOp>(
          extractOp, srcVector, indices, sizes, strides);
    } else {
      // perform a shuffle to extract the kD vector
      rewriter.replaceOpWithNewOp<mlir::vector::ShuffleOp>(
          extractOp, dstType, srcVector, srcVector,
          rewriter.getDenseI64ArrayAttr(indices));
    }
    return mlir::success();
  }
};

// clang-format off
// linearize InsertStridedSliceOp by extracting rows from the source vector
// using extract_strided_slice and inserting them into the destination vector
// using insert_strided_slice. For example.
//   vector.insert_strided_slice %s, %d {offsets=[0, 0]}: vector<2x4xf32> into vector<4x4xf32>
// will lowered into (both s and d are linearized to 1D):
//   %0 = vector.extract_strided_slice %s {offsets=[0], sizes=[4], strides=[1]} : vector<4xf32> from vector<8xf32>
//   %1 = vector.insert_strided_slice %0, %d {offsets=[0], strides=[1]} : vector<4xf32> into vector<16xf32>
//   %2 = vector.extract_strided_slice %s {offsets=[4], sizes=[4], strides=[1]} : vector<4xf32> from vector<8xf32>
//   %3 = vector.insert_strided_slice %2, %1 {offsets=[4], strides=[1]} : vector<4xf32> into vector<16xf32>
// clang-format on
struct VectorInsertStridedSliceConversion final
    : public mlir::OpConversionPattern<mlir::vector::InsertStridedSliceOp> {
  using mlir::OpConversionPattern<
      mlir::vector::InsertStridedSliceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::InsertStridedSliceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcTy = op.getSourceVectorType();
    auto dstTy = op.getDestVectorType();

    if (op.hasNonUnitStrides()) {
      return rewriter.notifyMatchFailure(
          op, "InsertStridedSliceOp linearization only supports unit strides.");
    }

    if (srcTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "InsertStridedSliceOp linearization only supports 2D source.");
    }

    if (!srcTy.hasStaticShape() || !dstTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "InsertStridedSliceOp linerization only supports static shapes.");
    }

    auto dstShape = dstTy.getShape();
    auto dstStrides = dstShape.drop_front().vec();
    dstStrides.push_back(1);
    int64_t linearizedOffset = 0;
    for (auto [off, stride] : llvm::zip_equal(op.getOffsets(), dstStrides)) {
      linearizedOffset += mlir::getConstantIntValue(off).value() * stride;
    }

    // extracts a row from source, and insert it into the destination
    auto srcShape = srcTy.getShape();
    mlir::Value dstValue = adaptor.getDest();
    for (auto i = 0; i < srcShape[0]; i++) {
      auto srcOffset = i * srcShape[1];
      auto value = rewriter.create<mlir::vector::ExtractStridedSliceOp>(
          loc, adaptor.getValueToStore(), srcOffset, srcShape[1], 1);

      auto dstOffset = linearizedOffset + i * dstShape.back();
      dstValue = rewriter.create<mlir::vector::InsertStridedSliceOp>(
          loc, value, dstValue, dstOffset, 1);
    }

    rewriter.replaceOp(op, dstValue);
    return mlir::success();
  }
};

struct VectorShffleOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::ShuffleOp> {
  using mlir::OpConversionPattern<mlir::vector::ShuffleOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(shuffleOp.getType());
    auto loc = shuffleOp.getLoc();
    if (!dstType)
      return rewriter.notifyMatchFailure(loc, "cannot convert type.");

    auto vec1 = adaptor.getV1();
    auto vec2 = adaptor.getV2();

    int shuffleSliceLen = 1;
    int rank = shuffleOp.getV1().getType().getRank();

    // if rank > 1, we need to do the shuffle in the granularity of slices
    // instead of scalars. Size of the slice is equal to the rank-1 innermost
    // dims. Mask of the shuffle op specifies which slice to take from the
    // outermost dim.
    if (rank > 1) {
      auto shape = shuffleOp.getV1().getType().getShape();
      for (unsigned i = 1; i < shape.size(); i++) {
        shuffleSliceLen *= shape[i];
      }
    }

    auto mask = shuffleOp.getMask();
    ;
    auto totalSize = mask.size() * shuffleSliceLen;

    llvm::SmallVector<int64_t, 2> indices(totalSize);
    for (auto [i, value] : llvm::enumerate(mask)) {

      std::iota(indices.begin() + shuffleSliceLen * i,
                indices.begin() + shuffleSliceLen * (i + 1),
                shuffleSliceLen * value);
    }

    rewriter.replaceOpWithNewOp<mlir::vector::ShuffleOp>(
        shuffleOp, dstType, vec1, vec2, rewriter.getDenseI64ArrayAttr(indices));

    return mlir::success();
  }
};

struct VectorExtractOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::ExtractOp extractOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstTy = getTypeConverter()->convertType(extractOp.getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(extractOp, "cannot convert type.");

    // dynamic position is not supported
    if (extractOp.hasDynamicPosition())
      return rewriter.notifyMatchFailure(extractOp,
                                         "dynamic position is not supported.");

    auto shape = extractOp.getVector().getType().getShape();
    auto size = extractOp.getVector().getType().getNumElements();

    // compute linearized offset
    int64_t linearizedOffset = 0;
    auto offsets = extractOp.getStaticPosition();
    for (auto [i, off] : llvm::enumerate(offsets)) {
      size /= shape[i];
      linearizedOffset += offsets[i] * size;
    }

    auto srcVector = adaptor.getVector();

    // ExtractOp also supports a semantic with result as a scalar, in which case
    // We need to use ExtractElementOp instead of ShuffleOp.
    if (dstTy.isIntOrIndexOrFloat()) {
      auto pos = rewriter.create<mlir::arith::ConstantOp>(
          extractOp.getLoc(), rewriter.getI32IntegerAttr(linearizedOffset));
      rewriter.replaceOpWithNewOp<mlir::vector::ExtractElementOp>(
          extractOp, srcVector, pos);
    } else {
      llvm::SmallVector<int64_t, 2> indices(size);
      std::iota(indices.begin(), indices.end(), linearizedOffset);
      rewriter.replaceOpWithNewOp<mlir::vector::ShuffleOp>(
          extractOp, dstTy, srcVector, srcVector,
          rewriter.getDenseI64ArrayAttr(indices));
    }

    return mlir::success();
  }
};

struct VectorInsertOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::InsertOp insertOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstTy = getTypeConverter()->convertType(insertOp.getDestVectorType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(insertOp, "cannot convert type.");

    // dynamic position is not supported
    if (insertOp.hasDynamicPosition())
      return rewriter.notifyMatchFailure(insertOp,
                                         "dynamic position is not supported.");
    auto srcTy = insertOp.getValueToStoreType();
    auto srcAsVec = mlir::dyn_cast<mlir::VectorType>(srcTy);
    uint64_t srcSize = 0;
    if (srcAsVec) {
      srcSize = srcAsVec.getNumElements();
    } else {
      return rewriter.notifyMatchFailure(insertOp,
                                         "scalars are not supported.");
    }

    auto dstShape = insertOp.getDestVectorType().getShape();
    const auto dstSize = insertOp.getDestVectorType().getNumElements();
    auto dstSizeForOffsets = dstSize;

    // compute linearized offset
    int64_t linearizedOffset = 0;
    auto offsetsNd = insertOp.getStaticPosition();
    for (auto [dim, offset] : llvm::enumerate(offsetsNd)) {
      dstSizeForOffsets /= dstShape[dim];
      linearizedOffset += offset * dstSizeForOffsets;
    }

    llvm::SmallVector<int64_t, 2> indices(dstSize);
    auto origValsUntil = indices.begin();
    std::advance(origValsUntil, linearizedOffset);
    std::iota(indices.begin(), origValsUntil,
              0); // original values that remain [0, offset)
    auto newValsUntil = origValsUntil;
    std::advance(newValsUntil, srcSize);
    std::iota(origValsUntil, newValsUntil,
              dstSize); // new values [offset, offset+srcNumElements)
    std::iota(newValsUntil, indices.end(),
              linearizedOffset + srcSize); // the rest of original values
                                           // [offset+srcNumElements, end)
    // NOTE : LLVM (and IGC) only supports shuffling vectors with the same
    // number of elements. Therefore, we need to modify the source vector to
    // have the same number of elements as the destination vector. eg.
    // %newSource = vector.shuffle %source, %source, [ {0-srcSize} ... fill with
    // 0 ]
    //     %dest = vector.shuffle %dest, %newSource, [ insert shuffle indices ]
    llvm::SmallVector<int64_t> modifiedSrcIndices(dstSize, 0);
    std::iota(modifiedSrcIndices.begin(), modifiedSrcIndices.begin() + srcSize,
              0);
    auto modifiedSource = rewriter.create<mlir::vector::ShuffleOp>(
        insertOp.getLoc(), dstTy, adaptor.getValueToStore(),
        adaptor.getValueToStore(), modifiedSrcIndices);

    rewriter.replaceOpWithNewOp<mlir::vector::ShuffleOp>(
        insertOp, dstTy, adaptor.getDest(), modifiedSource,
        rewriter.getDenseI64ArrayAttr(indices));

    return mlir::success();
  }
};

struct VectorSplatOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::SplatOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::SplatOp splatOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstTy = getTypeConverter()->convertType(splatOp.getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(splatOp, "cannot convert type.");
    rewriter.replaceOpWithNewOp<mlir::vector::SplatOp>(
        splatOp, adaptor.getInput(), dstTy);
    return mlir::success();
  }
};

struct VectorCreateMaskOpConversion final
    : mlir::OpConversionPattern<mlir::vector::CreateMaskOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::CreateMaskOp createMaskOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcTy = createMaskOp.getType();
    auto srcShape = srcTy.getShape();
    if (srcShape.size() != 2)
      return rewriter.notifyMatchFailure(createMaskOp,
                                         "only 2D mask is supported.");

    if (srcShape[0] != 1)
      return rewriter.notifyMatchFailure(
          createMaskOp, "only unit outer dimension is supported.");

    auto dstTy = getTypeConverter()->convertType(srcTy);
    if (!dstTy)
      return rewriter.notifyMatchFailure(createMaskOp, "cannot convert type.");

    rewriter.replaceOpWithNewOp<mlir::vector::CreateMaskOp>(
        createMaskOp, dstTy, adaptor.getOperands().back());
    return mlir::success();
  }
};

struct VectorBitCastOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::BitCastOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::BitCastOp castOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstTy = getTypeConverter()->convertType(castOp.getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(castOp, "cannot convert type.");

    rewriter.replaceOpWithNewOp<mlir::vector::BitCastOp>(castOp, dstTy,
                                                         adaptor.getSource());
    return mlir::success();
  }
};

struct UBPoisonOpConversion final
    : public mlir::OpConversionPattern<mlir::ub::PoisonOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::ub::PoisonOp poisonOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstTy = getTypeConverter()->convertType(poisonOp.getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(poisonOp, "cannot convert type.");

    rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(poisonOp, dstTy);
    return mlir::success();
  }
};

struct LinearizeRegionBranchOp final
    : public mlir::OpInterfaceConversionPattern<mlir::RegionBranchOpInterface> {
  using OpInterfaceConversionPattern<
      mlir::RegionBranchOpInterface>::OpInterfaceConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::RegionBranchOpInterface op,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto converter = getTypeConverter();

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.startOpModification(op);

    llvm::SmallVector<mlir::Type> convertedTypes;
    for (mlir::Type ty : op->getResultTypes()) {
      convertedTypes.push_back(converter->convertType(ty));
    }

    if (convertedTypes == op->getResultTypes() &&
        op->getOperands() == operands) {
      return mlir::failure();
    }

    op->setOperands(operands);

    // Convert region types (block arguments and yields)
    for (mlir::Region &region : op->getRegions()) {
      if (mlir::failed(rewriter.convertRegionTypes(&region, *converter))) {
        return mlir::failure();
      }

      // Process yields within each region
      for (mlir::Block &block : region) {
        if (auto *terminator = block.getTerminator()) {
          for (mlir::OpOperand &yieldOperand : terminator->getOpOperands()) {
            mlir::Value value = yieldOperand.get();
            mlir::Type type = value.getType();
            if (!converter->isLegal(type)) {
              mlir::Type newTy = converter->convertType(type);
              rewriter.setInsertionPoint(terminator);
              mlir::Value newValue =
                  rewriter.create<mlir::vector::ShapeCastOp>(loc, newTy, value);
              yieldOperand.set(newValue);
            }
          }
        }
      }
    }

    // Update result types
    rewriter.setInsertionPointAfter(op);
    llvm::SmallVector<mlir::Value> newResults;
    for (mlir::Value result : op->getResults()) {
      mlir::Type oldTy = result.getType();
      if (!converter->isLegal(oldTy)) {
        mlir::Type newTy = converter->convertType(oldTy);
        result.setType(newTy);
        mlir::Operation *castOp =
            rewriter.create<mlir::vector::ShapeCastOp>(loc, oldTy, result);
        result.replaceAllUsesExcept(castOp->getResult(0), castOp);
        newResults.push_back(castOp->getResult(0));
      } else {
        newResults.push_back(result);
      }
    }

    rewriter.finalizeOpModification(op);
    return mlir::success();
  }
};

struct VectorLinearizePass final
    : public imex::impl::VectorLinearizeBase<VectorLinearizePass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    // vector.broadcast and vector.gather requires progressive lowering
    {
      mlir::RewritePatternSet patterns(&getContext());
      mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
      mlir::vector::populateVectorGatherLoweringPatterns(patterns);
      mlir::vector::populateVectorGatherToConditionalLoadPatterns(patterns);
      (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    mlir::TypeConverter typeConverter;
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    typeConverter.addConversion([](mlir::Type type) { return type; });

    target.addDynamicallyLegalOp<mlir::vector::ShuffleOp>(
        [&](mlir::Operation *op) {
          auto ty = op->getResult(0).getType();
          auto vecTy = mlir::dyn_cast_or_null<mlir::VectorType>(ty);
          return vecTy && vecTy.getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::vector::ExtractStridedSliceOp>(
        [&](mlir::vector::ExtractStridedSliceOp op) {
          return op.getVector().getType().getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::vector::InsertStridedSliceOp>(
        [&](mlir::vector::InsertStridedSliceOp op) {
          auto srcTy = op.getSourceVectorType();
          auto dstTy = op.getDestVectorType();
          if (!op.hasNonUnitStrides() && srcTy.getRank() == 2 &&
              srcTy.hasStaticShape() && dstTy.hasStaticShape())
            return false;
          return true;
        });

    target.addDynamicallyLegalOp<mlir::vector::ExtractOp>(
        [&](mlir::vector::ExtractOp op) {
          return op.getVector().getType().getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::vector::LoadOp>(
        [&](mlir::vector::LoadOp op) {
          return op.getVectorType().getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::vector::StoreOp>(
        [&](mlir::vector::StoreOp op) {
          return op.getVectorType().getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::vector::CreateMaskOp>(
        [&](mlir::vector::CreateMaskOp op) {
          return op.getType().getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::vector::BitCastOp>(
        [&](mlir::vector::BitCastOp op) {
          return op.getType().getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::ub::PoisonOp>(
        [&](mlir::ub::PoisonOp op) {
          auto ty = op->getResult(0).getType();
          auto vecTy = mlir::dyn_cast_or_null<mlir::VectorType>(ty);
          return vecTy && vecTy.getRank() == 1;
        });

    target.addDynamicallyLegalOp<mlir::scf::IfOp, mlir::scf::ForOp,
                                 mlir::scf::WhileOp>([&](mlir::Operation *op) {
      for (mlir::Type resultType : op->getResultTypes()) {
        if (auto vecType = mlir::dyn_cast<mlir::VectorType>(resultType)) {
          if (vecType.getRank() != 1)
            return false;
        }
      }
      return true;
    });

    target.addIllegalOp<mlir::vector::TransposeOp>();
    target.addLegalOp<mlir::vector::ShapeCastOp>();
    target.addLegalOp<mlir::vector::ExtractElementOp>();
    target.addLegalDialect<mlir::xegpu::XeGPUDialect>();

    target.addDynamicallyLegalOp<mlir::vector::SplatOp>(
        [&](mlir::vector::SplatOp op) -> bool {
          return (op && op.getAggregate().getType().getRank() == 1);
        });

    patterns.add<UBPoisonOpConversion, VectorExtractStridedSliceConversion,
                 VectorInsertStridedSliceConversion, VectorShffleOpConversion,
                 VectorExtractOpConversion, VectorInsertOpConversion,
                 VectorSplatOpConversion, VectorLoadOpConversion,
                 VectorStoreOpConversion, VectorCreateMaskOpConversion,
                 VectorBitCastOpConversion, LinearizeRegionBranchOp>(
        typeConverter, context);

    // Shuffle16x16 will fallback to Shuffle1D for non 16x16 sizes.
    mlir::vector::populateVectorTransposeLoweringPatterns(
        patterns, mlir::vector::VectorTransposeLowering::Shuffle16x16);
    populateVectorLinearizeTypeConversionsAndLegality(typeConverter, patterns,
                                                      target);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createVectorLinearizePass() {
  return std::make_unique<VectorLinearizePass>();
}
