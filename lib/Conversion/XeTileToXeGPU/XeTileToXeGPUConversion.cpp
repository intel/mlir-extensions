//===- XeTileToXeGPUConversion.cpp - XeTileToXeGPU conversion  -------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the SgXeTileToXeGPUConversion, the base class for
/// XeTileToXeGPU conversion, XeGPUTypeConverter, converting types used in
/// XeTile dialect to types used in XeGPU dialect, XeGPUOneToNPatterRewriter a
/// wrapper around ConversionPatterRewriter providng interface for supporting
/// OneToN replace.
///
//===----------------------------------------------------------------------===//
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h>
#include <imex/Dialect/XeGPU/IR/XeGPU.h>
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <imex/Utils/DebugUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include <optional>

#include "../PassDetail.h"

namespace imex {

static bool isIdentityConversion(mlir::Type originalType,
                                 mlir::TypeRange convertedTypes) {
  return convertedTypes.size() == 1 && convertedTypes[0] == originalType;
}

static llvm::SmallVector<mlir::Value>
buildUnrealizedBackwardsCasts(mlir::ValueRange convertedValues,
                              const mlir::OneToNTypeMapping &typeConversion,
                              mlir::RewriterBase &rewriter) {

  // assert(typeConversion.getConvertedTypes() == convertedValues.getTypes());

  // Create unrealized cast op for each converted result of the op.
  llvm::SmallVector<mlir::Value> recastValues;
  mlir::TypeRange originalTypes = typeConversion.getOriginalTypes();
  recastValues.reserve(originalTypes.size());
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

XeGPUTypeConverter::XeGPUTypeConverter(mlir::MLIRContext &context,
                                       imex::ValueAttributeMap &map)
    : XeTypeConverter(context, map) {
  addConversion(
      [&](mlir::IndexType type) -> std::optional<mlir::Type> { return type; });

  addConversion(
      [&](mlir::MemRefType type) -> std::optional<mlir::Type> { return type; });

  addArgumentMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });

  addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
}

std::optional<mlir::LogicalResult> XeGPUTypeConverter::convertTileType(
    xetile::TileType tileTy, llvm::SmallVectorImpl<mlir::Type> &resultTypes) {
  if (tileTy.getRank() == 2) {
    resultTypes.push_back(tileTy);
    return mlir::success();
  } else if (tileTy.getRank() == 4) {
    auto shape = tileTy.getShape();
    auto tdescTy = xegpu::TensorDescType::get({shape[2], shape[3]},
                                              tileTy.getElementType());
    auto numElements = shape[0] * shape[1];
    resultTypes.assign(numElements, tdescTy);
    return mlir::success();
  }
  return std::nullopt;
}

std::optional<mlir::LogicalResult> XeGPUTypeConverter::convertVectorType(
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

mlir::Block *XeGPUOneToNPatterRewriter::applySignatureConversion(
    mlir::Region *region, mlir::TypeConverter::SignatureConversion &conversion,
    const mlir::TypeConverter *converter) {
  return rewriter.applySignatureConversion(region, conversion, converter);
}

void XeGPUOneToNPatterRewriter::replaceOp(mlir::Operation *op,
                                          mlir::ValueRange newValues) {
  // It is one-to-one mapping, let the ConvertionPatternRewriter handle it
  // directly.
  if (newValues.size() == op->getNumResults()) {
    rewriter.replaceOp(op, newValues);
  } else { // it is one-to-N mapping, so create unrealizedCasts to make it as
           // one-to-one mapping
    llvm::SmallVector<mlir::Value> recastValues;
    auto resultTys = op->getResultTypes();
    mlir::OneToNTypeMapping resultMapping(resultTys);
    if (mlir::succeeded(
            typeConverter.computeTypeMapping(resultTys, resultMapping))) {
      auto castValues =
          buildUnrealizedBackwardsCasts(newValues, resultMapping, rewriter);
      rewriter.replaceOp(op, castValues);
    } else {
      llvm_unreachable("It is an unexpected failure of failing to convert the "
                       "result types.");
    }
  }
}

} // namespace imex
