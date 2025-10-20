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

#include "imex/Transforms/Passes.h"

namespace imex {
#define GEN_PASS_DEF_REMOVESINGLEELEMVECTOR
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {

struct VectorExtractOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::ExtractOp> {
  using mlir::OpConversionPattern<mlir::vector::ExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::ExtractOp extractOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    if (adaptor.getSource().getType() == extractOp.getType()) {
      rewriter.replaceOp(extractOp, adaptor.getSource());
      return mlir::success();
    }

    auto vector = extractOp.getSource();
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
    auto srcVector = extractOp.getSource();
    auto offsets = extractOp.getOffsets();

    // We only convert ops extracting a single element from a 1D vector.
    if (resType.getNumElements() == 1 && srcVector.getType().getRank() == 1) {
      rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(extractOp, srcVector,
                                                           offsets[0]);
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
static mlir::Value createInsertOps(OpTy op, mlir::ValueRange operands,
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
    newOp = rewriter.create<mlir::vector::InsertOp>(loc, opr, newOp, i);
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
      auto newOp = createInsertOps(shuffleOp, {vec1, vec2}, rewriter);
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
      auto newOp = createInsertOps(interleaveOp, {vec1, vec2}, rewriter);
      rewriter.replaceOp(interleaveOp, newOp);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct VectorBroadcastOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::BroadcastOp> {
  using mlir::OpConversionPattern<
      mlir::vector::BroadcastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::BroadcastOp splatOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto type = mlir::cast<mlir::VectorType>(splatOp.getType());
    if (type.getNumElements() != 1)
      return mlir::failure();

    rewriter.replaceOp(splatOp, adaptor.getSource());
    return mlir::success();
  }
};

// Vector store op transformation pattern for single element vector.
// Vector.store is converted to memref.store if the vector is a single element
// vector.

// The full transformation is as follows:

// Input:
// vector.store %vector, %memref[%idx] : memref<4xf32>, vector<1xf32>

// Output:
// %scalar = vector.extract %vector[%c0:i32] : vector<1xf32>
// memref.store %scalar, %memref[%idx] : memref<4xf32>

struct VectorStoreOpConversion final
    : public mlir::OpConversionPattern<mlir::vector::StoreOp> {
  using mlir::OpConversionPattern<mlir::vector::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::StoreOp storeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vecTy = storeOp.getVectorType();
    // Only do transformation if the vector type is 1D and has a single element
    // @TODO: Do we need to handle 0-D vector?
    if (!(vecTy.getRank() == 1 && vecTy.getNumElements() == 1)) {
      return mlir::success();
    }

    auto vector = storeOp.getValueToStore();
    auto base = storeOp.getBase();
    auto indices = storeOp.getIndices();

    // Create a i32 constant of value 0 for index

    // Extract the single element from the vector as a scalar
    auto scalar = rewriter.create<mlir::vector::ExtractOp>(
        storeOp.getLoc(), vector, rewriter.getI32IntegerAttr(0));

    // Create a memref.store op with the scalar value
    auto memrefStoreOp = rewriter.create<mlir::memref::StoreOp>(
        storeOp.getLoc(), scalar, base, indices);

    rewriter.replaceOp(storeOp, memrefStoreOp);
    return mlir::success();
  }
};

struct RemoveSingleElemVectorPass final
    : public imex::impl::RemoveSingleElemVectorBase<
          RemoveSingleElemVectorPass> {

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::TypeConverter typeConverter;
    // convert a value from vector<1xT> to T using vector::ExtractOp
    auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type resultType,
                              mlir::ValueRange inputs, mlir::Location loc) {
      if (inputs.size() != 1)
        return mlir::Value();

      auto vecTy = mlir::dyn_cast<mlir::VectorType>(inputs[0].getType());
      if (!vecTy || vecTy.getNumElements() != 1)
        return mlir::Value();

      return builder
          .create<mlir::vector::ExtractOp>(loc, inputs[0],
                                           builder.getIndexAttr(0))
          .getResult();
    };

    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    typeConverter.addConversion([](mlir::Type type) {
      auto vecTy = mlir::dyn_cast<mlir::VectorType>(type);
      if (vecTy && vecTy.getNumElements() == 1)
        return vecTy.getElementType();
      return type;
    });

    mlir::ConversionTarget target(*context);
    target.addLegalOp<mlir::memref::StoreOp>();
    target.addLegalOp<mlir::arith::ConstantOp>();
    target.addLegalOp<mlir::vector::InsertOp>();
    target.addDynamicallyLegalOp<mlir::vector::ExtractOp>(
        [&](mlir::vector::ExtractOp op) {
          auto vecTy = op.getSource().getType();
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
    // Disable ectorExtractStridedSliceConversion for now as it interferes with
    // xetile-blockop-fallback pass
    patterns.add</*VectorExtractStridedSliceConversion,*/ VectorizableOpPattern,
                 VectorShffleOpConversion, VectorInterleaveOpConversion,
                 VectorBroadcastOpConversion, VectorExtractOpConversion,
                 VectorStoreOpConversion>(typeConverter, context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createRemoveSingleElemVectorPass() {
  return std::make_unique<RemoveSingleElemVectorPass>();
}
