//===- XeTileOneToNConversion.cpp -- XeTileOneToNConversion  ----*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeOneToNConversion, the base class for
/// XeTileToXeGPU conversion, XeOneToNTypeConverter, converting types used in
/// XeTile dialect to types used in XeGPU dialect, XeOneToNPatternRewriter a
/// wrapper around ConversionPatterRewriter providng interface for supporting
/// OneToN replace.
///
//===----------------------------------------------------------------------===//
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <imex/Dialect/XeTile/Transforms/XeTileOneToNConversion.h>
#include <imex/Utils/DebugUtils.h>
#include <imex/Utils/PassWrapper.h>
#include <imex/Utils/XeCommon.h>

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

#include <optional>

namespace imex {

static bool isIdentityConversion(mlir::Type originalType,
                                 mlir::TypeRange convertedTypes) {
  return convertedTypes.size() == 1 && convertedTypes[0] == originalType;
}

static llvm::SmallVector<mlir::Value>
buildUnrealizedBackwardsCasts(mlir::ValueRange convertedValues,
                              const mlir::OneToNTypeMapping &typeConversion,
                              mlir::RewriterBase &rewriter) {

  // Create unrealized cast op for each converted result of the op.
  mlir::TypeRange originalTypes = typeConversion.getOriginalTypes();
  llvm::SmallVector<mlir::Value> recastValues;
  recastValues.reserve(originalTypes.size());

  if (llvm::range_size(originalTypes) == 1 &&
      llvm::range_size(convertedValues) > 1) {
    mlir::ValueRange recastValue =
        buildUnrealizedCast(rewriter, originalTypes[0], convertedValues);
    assert(recastValue.size() == 1);
    recastValues.push_back(recastValue.front());
    return recastValues;
  }

  auto convertedValueIt = convertedValues.begin();
  for (auto [idx, originalType] : llvm::enumerate(originalTypes)) {
    mlir::TypeRange convertedTypes = typeConversion.getConvertedTypes(idx);
    size_t numConvertedValues = convertedTypes.size();
    if (isIdentityConversion(originalType, convertedTypes)) {
      // Identity conversion: take result as is.
      recastValues.push_back(*convertedValueIt);
    } else {
      // Non-identity conversion: cast back to source type.
      mlir::ValueRange recastValue = buildUnrealizedCast(
          rewriter, originalType,
          mlir::ValueRange{convertedValueIt,
                           convertedValueIt + numConvertedValues});
      assert(recastValue.size() == 1);
      recastValues.push_back(recastValue.front());
    }
    convertedValueIt += numConvertedValues;
  }

  return recastValues;
}

XeOneToNTypeConverter::XeOneToNTypeConverter(mlir::MLIRContext &context)
    : XeTypeConverter(context) {
  targetOp = nullptr;

  addConversion(
      [&](mlir::IndexType type) -> std::optional<mlir::Type> { return type; });

  addConversion(
      [&](mlir::MemRefType type) -> std::optional<mlir::Type> { return type; });

  addArgumentMaterialization([&](mlir::OpBuilder &builder,
                                 mlir::Type resultType, mlir::ValueRange inputs,
                                 mlir::Location loc) -> mlir::Value {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

std::optional<mlir::LogicalResult> XeOneToNTypeConverter::convertTileType(
    xetile::TileType tileTy, llvm::SmallVectorImpl<mlir::Type> &resultTypes) {
  llvm::dbgs()
      << "convertTileType is disabled, since there is no unique "
      << "way to convert an XeTile::TileType into mlir::xegpu::TensorDescType "
      << "becasue of array_length selection.\n";
  return std::nullopt;
}

std::optional<mlir::LogicalResult> XeOneToNTypeConverter::convertVectorType(
    mlir::VectorType vectorTy, llvm::SmallVectorImpl<mlir::Type> &resultTypes) {
  if (vectorTy.getRank() == 4) {
    auto shape = vectorTy.getShape();
    auto vecTy =
        mlir::VectorType::get({shape[2], shape[3]}, vectorTy.getElementType());
    auto numElements = shape[0] * shape[1];
    resultTypes.assign(numElements, vecTy);
    return mlir::success();
  } else if (vectorTy.getRank() == 2) {
    resultTypes.push_back(vectorTy);
    return mlir::success();
  }
  return std::nullopt;
}

// It computes the mapping between types orginal values and
// converted values. The standard type conversion method doesn't
// work here because a TileType could have multiple decomposions
// into TensorDescType depending on the choice for array_len.
// For given types of inputs of
// originalTypes = [xetile.tile<32x64xf16>, vector<32x64xf16>]
// convertedTypes = [xegpu.tensor_desc<32x16xf16, arr_len=2>,
//                   xegpu.tensor_desc<32x16xf16, arr_len=2>,
//                   vector<32x32xf16>, vector<32x32xf16>]
//
// It will build a map as shown in below, and store it in resultMap.
// xetile.tile<32x64xf16> --> [xegpu.tensor_desc<32x16xf16, arr_len=2>,
//                             xegpu.tensor_desc<32x16xf16, arr_len=2>]
//
// vector<32x64xf16> --> [vector<32x16xf16>, vector<32x16xf16>]
//
// The alogrithm works by inferring the number of new types based on their
// size info. It assumes the origianlTyps and convertedTypes has been aligned.
// the originalTypes, which is the key, has been added into resultMap as
// constructor
mlir::LogicalResult
XeOneToNTypeConverter::computeTypeMapping(mlir::ValueRange original,
                                          mlir::ValueRange converted,
                                          mlir::OneToNTypeMapping &resultMap) {
  llvm::SmallVector<mlir::Type> originalTypes(original.getType());
  llvm::SmallVector<mlir::Type> convertedTypes(converted.getType());

  for (size_t i = 0, j = 0; i < originalTypes.size(); i++) {
    assert(j < convertedTypes.size());
    if (originalTypes[i] == convertedTypes[j]) {
      resultMap.addInputs(i, convertedTypes[j++]);
      continue;
    }

    if (auto tileTy = llvm::dyn_cast<xetile::TileType>(originalTypes[i])) {
      auto tdescTy =
          llvm::dyn_cast<mlir::xegpu::TensorDescType>(convertedTypes[j]);
      if (!tdescTy)
        return mlir::failure();
      auto shape = tileTy.getShape();
      auto blkSZ = tdescTy.getShape();
      auto arr_len = tdescTy.isScattered() ? 1 : tdescTy.getArrayLength();
      auto totalNumElems =
          std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
      auto blockNumElems =
          std::accumulate(blkSZ.begin(), blkSZ.end(), 1, std::multiplies<>{});
      auto size = totalNumElems / blockNumElems / arr_len;
      llvm::ArrayRef<mlir::Type> types(convertedTypes.begin() + j,
                                       convertedTypes.begin() + j + size);
      resultMap.addInputs(i, types);
      j += size;
      continue;
    }

    if (auto vecTy = llvm::dyn_cast<mlir::VectorType>(originalTypes[i])) {
      auto cvtTy = llvm::dyn_cast<mlir::VectorType>(convertedTypes[j]);
      if (vecTy.getRank() != 4 || !cvtTy)
        return mlir::failure();

      auto shape = vecTy.getShape();
      auto blkSZ = cvtTy.getShape();

      auto product = [&](llvm::ArrayRef<int64_t> arr) {
        return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies<>{});
      };

      // using the total size to check whether cvtTy is derived from vecTy
      // vnni factors restrict us from apple to apple compare.
      if (product(shape.take_back(2)) != product(blkSZ))
        return mlir::failure();

      auto size = shape[0] * shape[1];
      llvm::ArrayRef<mlir::Type> types(convertedTypes.begin() + j,
                                       convertedTypes.begin() + j + size);
      resultMap.addInputs(i, types);
      j += size;
      continue;
    }

    return mlir::failure();
  }
  return mlir::success();
}

mlir::Block *XeOneToNPatternRewriter::applySignatureConversion(
    mlir::Block *block, mlir::TypeConverter::SignatureConversion &conversion,
    const mlir::TypeConverter *converter) {
  return rewriter.applySignatureConversion(block, conversion, converter);
}

void XeOneToNPatternRewriter::replaceOp(mlir::Operation *op,
                                        mlir::ValueRange newValues) {
  // It is one-to-one mapping, let the ConvertionPatternRewriter
  // handle it directly.
  if (newValues.size() == op->getNumResults()) {
    rewriter.replaceOp(op, newValues);
    return;
  }

  // it is one-to-N mapping, so create unrealizedCasts
  // to make it as one-to-one mapping
  assert(newValues.size() > op->getNumResults() &&
         "It is unexpected that the num of new op results "
         "is less than num of orignial op results.\n");

  mlir::OneToNTypeMapping resultMapping(op->getResultTypes());
  auto status = typeConverter.computeTypeMapping(op->getResults(), newValues,
                                                 resultMapping);
  if (mlir::failed(status)) {
    (void)rewriter.notifyMatchFailure(op,
                                      "Failed to convert the result types.");
    return;
  }

  auto castValues =
      buildUnrealizedBackwardsCasts(newValues, resultMapping, rewriter);
  rewriter.replaceOp(op, castValues);
}

void XeOneToNPatternRewriter::replaceOp(
    mlir::Operation *op, mlir::ValueRange newValues,
    const mlir::OneToNTypeMapping &resultMapping) {
  // It is one-to-one mapping, let the ConvertionPatternRewriter
  // handle it directly.
  if (newValues.size() == op->getNumResults()) {
    rewriter.replaceOp(op, newValues);
    return;
  }

  // it is one-to-N mapping, so create unrealizedCasts
  // to make it as one-to-one mapping
  assert(newValues.size() > op->getNumResults() &&
         "It is unexpected that the num of new op results "
         "is less than num of orignial op results.\n");

  auto castValues =
      buildUnrealizedBackwardsCasts(newValues, resultMapping, rewriter);
  rewriter.replaceOp(op, castValues);
}

} // namespace imex
