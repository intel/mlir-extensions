//===- RemoveSingleElemVector.cpp - RemoveSingleElemVector Pass -*- C++- *-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains RemoveSingleElemVector pass.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Transforms/Utils/AddDiscriminators.h"

#include "imex/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdint>
#include <numeric>

namespace imex {
#define GEN_PASS_DEF_REMOVESINGLEELEMVECTOR
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {

struct VectorExtractElementOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::ExtractElementOp> {
  using mlir::OpConversionPattern<
      mlir::vector::ExtractElementOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::ExtractElementOp extractOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vector = extractOp.getVector();
    auto vecTy = vector.getType();
    auto constOp = vector.getDefiningOp<mlir::arith::ConstantOp>();

    if (vecTy.getRank() == 1 && vecTy.getNumElements() == 1 && constOp) {
      auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
      if (!value)
        return mlir::failure();

      auto attr = value.getValues<mlir::TypedAttr>()[0];
      auto elemTy = vecTy.getElementType();

      auto newVal = rewriter.create<mlir::arith::ConstantOp>(extractOp.getLoc(),
                                                             elemTy, attr);

      rewriter.replaceOp(extractOp, newVal);
      return mlir::success();
    }
    return mlir::failure();
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

    auto resType = mlir::cast<mlir::VectorType>(extractOp.getType());
    auto srcVector = extractOp.getVector();
    auto offsets = extractOp.getOffsets();

    // We only convert ops extracting a single element from a 1D vector.
    if (resType.getNumElements() == 1 && srcVector.getType().getRank() == 1) {
      auto pos = rewriter.create<mlir::arith::ConstantOp>(
          extractOp.getLoc(), mlir::cast<mlir::IntegerAttr>(offsets[0]));
      rewriter.replaceOpWithNewOp<mlir::vector::ExtractElementOp>(
          extractOp, srcVector, pos);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct VectorizableOpPattern final
    : public mlir::OpTraitConversionPattern<mlir::OpTrait::Vectorizable> {
  using mlir::OpTraitConversionPattern<
      mlir::OpTrait::Vectorizable>::OpTraitConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // We only convert ops that are vectorizable.

    if (op->getNumResults() != 1)
      return mlir::failure();

    auto res = op->getResult(0);
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    if (!resType || resType.getNumElements() != 1)
      return mlir::failure();

    auto elemType = resType.getElementType();
    mlir::OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes({elemType});
    state.addAttributes(op->getAttrs());
    auto newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

template <typename OpTy>
static mlir::Value
createInsertElementOps(OpTy op, mlir::ValueRange operands,
                       mlir::ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto type = op.getType();
  auto elemType = type.getElementType();

  mlir::TypedAttr fillValue = rewriter.getIndexAttr(0);
  if (elemType.isInteger())
    fillValue = rewriter.getIntegerAttr(elemType, 0);
  if (mlir::isa<mlir::FloatType>(elemType))
    fillValue = rewriter.getFloatAttr(elemType, 0);
  auto denseAttr = mlir::DenseElementsAttr::get(type, fillValue);

  mlir::Value newOp =
      rewriter.create<mlir::arith::ConstantOp>(loc, type, denseAttr);
  for (auto [i, opr] : llvm::enumerate(operands)) {
    mlir::Value pos = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(i));
    newOp =
        rewriter.create<mlir::vector::InsertElementOp>(loc, opr, newOp, pos);
  }
  return newOp;
}

struct VectorShffleOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::ShuffleOp> {
  using mlir::OpConversionPattern<mlir::vector::ShuffleOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vec1 = adaptor.getV1();
    auto vec2 = adaptor.getV2();
    if (vec1.getType().isIntOrIndexOrFloat() &&
        vec2.getType().isIntOrIndexOrFloat()) {
      auto newOp = createInsertElementOps(shuffleOp, {vec1, vec2}, rewriter);
      rewriter.replaceOp(shuffleOp, newOp);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct VectorInterleaveOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::InterleaveOp> {
  using mlir::OpConversionPattern<
      mlir::vector::InterleaveOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::InterleaveOp interleaveOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vec1 = adaptor.getLhs();
    auto vec2 = adaptor.getRhs();

    if (vec1.getType().isIntOrIndexOrFloat() &&
        vec2.getType().isIntOrIndexOrFloat()) {
      auto newOp = createInsertElementOps(interleaveOp, {vec1, vec2}, rewriter);
      rewriter.replaceOp(interleaveOp, newOp);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct VectorSplatOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::SplatOp> {
  using mlir::OpConversionPattern<mlir::vector::SplatOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::SplatOp splatOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto type = mlir::cast<mlir::VectorType>(splatOp.getType());
    if (type.getNumElements() != 1)
      return mlir::failure();

    rewriter.replaceOp(splatOp, adaptor.getInput());
    return mlir::success();
  }
};

struct RemoveSingleElemVectorPass final
    : public imex::impl::RemoveSingleElemVectorBase<
          RemoveSingleElemVectorPass> {

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::TypeConverter typeConverter;

    auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type resultType,
                              mlir::ValueRange inputs, mlir::Location loc) {
      return builder
          .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };

    typeConverter.addArgumentMaterialization(materializeCast);
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);
    typeConverter.addConversion([](mlir::Type type) {
      auto vecTy = mlir::dyn_cast<mlir::VectorType>(type);
      if (vecTy && vecTy.getNumElements() == 1)
        return vecTy.getElementType();
      return type;
    });

    // typeConverter.addConversion([](mlir::Type type) { return type; });

    mlir::ConversionTarget target(*context);
    target.addLegalOp<mlir::arith::ConstantOp>();
    target.addLegalOp<mlir::vector::InsertElementOp>();
    target.addDynamicallyLegalOp<mlir::vector::ExtractElementOp>(
        [&](mlir::vector::ExtractElementOp op) {
          auto vecTy = op.getVector().getType();
          return vecTy.getNumElements() != 1;
        });

    target.markUnknownOpDynamicallyLegal(
        [=](mlir::Operation *op) -> std::optional<bool> {
          if (op->hasTrait<mlir::OpTrait::Vectorizable>()) {
            // we mark all vectorizable ops as legal except
            // those that return a vector with a single element
            bool isLegal = true;
            if (op->getNumResults() == 1) {
              auto type = op->getResult(0).getType();
              auto vecTy = mlir::dyn_cast<mlir::VectorType>(type);
              isLegal = !(vecTy && vecTy.getNumElements() == 1);
            }
            return isLegal;
          }
          return std::nullopt;
        });

    mlir::RewritePatternSet patterns(context);
    patterns.add<VectorExtractStridedSliceConversion, VectorizableOpPattern,
                 VectorShffleOpConversion, VectorInterleaveOpConversion,
                 VectorSplatOpConversion, VectorExtractElementOpConversion>(
        typeConverter, context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createRemoveSingleElemVectorPass() {
  return std::make_unique<RemoveSingleElemVectorPass>();
}
