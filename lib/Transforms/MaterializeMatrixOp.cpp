//===-- MaterializeMatrixOp.cpp - MaterializeMatrixOpPass  ----------*-
// C++-*-===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains MaterializeMatrixOp pass used for Xe2/Xe3 architecture.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "imex/Utils/XeCommon.h"

#include <memory>

namespace imex {
#define GEN_PASS_DEF_MATERIALIZEMATRIXOP
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace imex;
using namespace mlir;
using namespace mlir::xegpu;

namespace {

// Lower xegpu::CreateMemDescOp to memref::ViewOp. Since SLM access instructions
// on Xe2 and Xe3 operate on 32-bit or 64-bit units, all data types smaller than
// 32 bits will be converted to 32 bits.
class CreateMemDescOpPattern final
    : public OpConversionPattern<xegpu::CreateMemDescOp> {
public:
  using OpConversionPattern<xegpu::CreateMemDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateMemDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypedValue<MemRefType> src = dyn_cast<TypedValue<MemRefType>>(op.getSource());
    MemDescType resTy = op.getMemDesc().getType();
    auto *converter = getTypeConverter();
    MemRefType newResTy = converter->convertType<MemRefType>(resTy);
    Value zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(op, newResTy, src, zero,
                                                ValueRange());
    return success();
  }
};

class LoadMatrixOpPattern final
    : public OpConversionPattern<xegpu::LoadMatrixOp> {
public:
  using OpConversionPattern<xegpu::LoadMatrixOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    auto loc = op.getLoc();
    MemDescType mdescTy = op.getMemDesc().getType();
    VectorType resTy = dyn_cast<VectorType>(op.getRes().getType());

    SmallVector<int64_t> blockShape = mdescTy.getBlockShape();

    if (blockShape.empty()) {
      // TODO: lowering to regular load/store
      // in case the SLM can't be blocked due to some limitation, the lowering
      // need to fall back to regular load/store. The inst_data size may be
      // bigger than regular load/store so need to be split to multiple regular
      // load/store if needed.
      return rewriter.notifyMatchFailure(
          op, "LoadMatrixOp without blocking layout are not yet supported.");
    }

    // TODO: support col-major
    if (mdescTy.isColMajor())
      return rewriter.notifyMatchFailure(op, "unsupported memory descriptor");

    int packSize = getVnniFactor(resTy.getElementType());
    int vecSize = resTy.getNumElements();

    auto converter = getTypeConverter();
    Type elemTy = converter->convertType(resTy.getElementType());
    Attribute encoding =
        BlockTensorDescAttr::get(context, xegpu::MemorySpace::SLM, 1, true);

    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    assert(blockShape.size() == 2 && "only support blocking for rank-2 matrix");
    Value linearOffset = mdescTy.getLinearOffsets(rewriter, loc, offsets);
    if (packSize > 1) {
      vecSize = vecSize / packSize;
      Value packSizeScalar = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(packSize));
      linearOffset =
          arith::DivUIOp::create(rewriter, loc, linearOffset, packSizeScalar);
    }

    auto tdescTy = TensorDescType::get(context, vecSize, elemTy, encoding,
                                       /*layout=*/nullptr);

    Value tdesc = xegpu::CreateNdDescOp::create(
        rewriter, loc, tdescTy,
        dyn_cast<TypedValue<MemRefType>>(adaptor.getMemDesc()),
        OpFoldResult(linearOffset));

    auto packAttr = UnitAttr();
    auto transAttr = DenseI64ArrayAttr();
    auto bitWidthAttr = IntegerAttr();
    VectorType newResTy = VectorType::get(vecSize, elemTy);
    auto ldOp = xegpu::LoadNdOp::create(
        rewriter, loc, newResTy, tdesc, ValueRange(), DenseI64ArrayAttr(),
        packAttr, transAttr, bitWidthAttr, nullptr, nullptr, nullptr, nullptr);

    Value result = ldOp.getResult();

    // cast back
    elemTy = resTy.getElementType();
    auto castTy = VectorType::get(resTy.getNumElements(), elemTy);
    if (castTy != newResTy)
      result = vector::BitCastOp::create(rewriter, loc, castTy, result);
    if (castTy != resTy)
      result = vector::ShapeCastOp::create(rewriter, loc, resTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert xegpu::StoreMatrixOp to xegpu::StoreNdOp if MemDesc is
// row-major or xegpu::StoreScatterOp if MemDesc is col-major.
class StoreMatrixOpPattern final
    : public OpConversionPattern<xegpu::StoreMatrixOp> {
public:
  using OpConversionPattern<xegpu::StoreMatrixOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::StoreMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value data = adaptor.getData();
    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();

    VectorType dataTy = dyn_cast<VectorType>(op.getData().getType());
    SmallVector<int64_t> dataShape(dataTy.getShape().begin(),
                                   dataTy.getShape().end());
    int packSize = getVnniFactor(dataTy.getElementType());

    MemDescType mdescTy = op.getMemDesc().getType();
    SmallVector<int64_t> blockShape = mdescTy.getBlockShape();

    if (blockShape.empty()) {
      // TODO: lowering to regular load/store
      // in case the SLM can't be blocked due to some limitation, the lowering
      // need to fall back to regular load/store. The inst_data size may be
      // bigger than regular load/store so need to be split to multiple regular
      // load/store if needed.
      return rewriter.notifyMatchFailure(
          op,
          "unblocked StoreMatrixOp are not supported on Xe2/Xe3 architecture.");
    }

    assert(blockShape.size() == 2 && "only support blocking for rank-2 matrix");
    Value linearOffset = mdescTy.getLinearOffsets(rewriter, loc, offsets);
    if (mdescTy.isColMajor()) {
      int64_t vecSize = dataShape[1];
      int64_t stride = blockShape[0];

      auto indexVecTy = VectorType::get(vecSize, rewriter.getIndexType());
      Value linearOffsetVec =
          vector::BroadcastOp::create(rewriter, loc, indexVecTy, linearOffset);

      // Generate a vector of indices [0, 1, ..., vecSize-1] as constant index
      // values
      SmallVector<Attribute> indexAttrs =
          llvm::map_to_vector(llvm::seq<int64_t>(0, vecSize), [&](int64_t i) {
            return cast<Attribute>(rewriter.getIndexAttr(i));
          });
      Value indexVec = arith::ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(indexVecTy, indexAttrs));
      Value strideConst = arith::ConstantIndexOp::create(rewriter, loc, stride);
      Value strideVec =
          vector::BroadcastOp::create(rewriter, loc, indexVecTy, strideConst);
      Value mulOp = arith::MulIOp::create(rewriter, loc, indexVec, strideVec);
      Value colOffsets =
          arith::AddIOp::create(rewriter, loc, linearOffsetVec, mulOp);

      if (packSize > 1) {
        Value packSizeScalar = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIndexAttr(packSize));
        Value packSizeVec = vector::BroadcastOp::create(
            rewriter, loc, colOffsets.getType(), packSizeScalar);
        colOffsets =
            arith::DivUIOp::create(rewriter, loc, colOffsets, packSizeVec);
      }
      bool tryFold = op.getData().hasOneUse();
      data = convertToPackedVector(rewriter, loc, data, tryFold);
      auto maskTy = VectorType::get(blockShape[1], rewriter.getI1Type());
      auto mask = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(maskTy, rewriter.getBoolAttr(true)));
      SmallVector<int64_t> permutation = {1, 0};
      data = vector::TransposeOp::create(rewriter, loc, data, permutation);
      { // using old style.
        MLIRContext *context = op.getContext();
        auto converter = getTypeConverter();
        int64_t chunkSize = dataShape[0] / packSize;

        auto encoding = xegpu::ScatterTensorDescAttr::get(
            context, xegpu::MemorySpace::SLM, chunkSize);
        Type elemTy = converter->convertType(dataTy.getElementType());
        auto tdescTy = TensorDescType::get(context, {vecSize, chunkSize},
                                           elemTy, encoding, nullptr);

        Value tdesc = xegpu::CreateDescOp::create(
            rewriter, loc, tdescTy, adaptor.getMemDesc(), colOffsets);
        rewriter.replaceOpWithNewOp<xegpu::StoreScatterOp>(
            op, data, tdesc, mask, nullptr, nullptr, nullptr);
      }
      { // new style.
        // auto chunkSize = rewriter.getI64IntegerAttr(blockShape[0] /
        // packSize); rewriter.replaceOpWithNewOp<xegpu::StoreScatterOp>(
        //     op, data, adaptor.getMemDesc(), linearOffset, mask,
        //     chunkSize, nullptr, nullptr, nullptr);
      }
    } else { // lower to 1D block TenssorDesc
      MLIRContext *context = op.getContext();
      int vecSize = dataTy.getNumElements();
      auto converter = getTypeConverter();
      Type elemTy = converter->convertType(dataTy.getElementType());
      Attribute encoding =
          BlockTensorDescAttr::get(context, xegpu::MemorySpace::SLM, 1, true);

      if (packSize > 1) {
        vecSize = vecSize / packSize;
        Value packSizeScalar = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIndexAttr(packSize));
        linearOffset =
            arith::DivUIOp::create(rewriter, loc, linearOffset, packSizeScalar);
      }

      auto tdescTy = TensorDescType::get(context, vecSize, elemTy, encoding,
                                         /*layout=*/nullptr);
      Value tdesc = xegpu::CreateNdDescOp::create(
          rewriter, loc, tdescTy,
          dyn_cast<TypedValue<MemRefType>>(adaptor.getMemDesc()),
          OpFoldResult(linearOffset));
      data = convertTo1D32BitVector(data, loc, rewriter);
      rewriter.replaceOpWithNewOp<xegpu::StoreNdOp>(op, data, tdesc, nullptr,
                                                    nullptr, nullptr);
    }
    return success();
  }
};

/// Populate the given list with patterns that convert MemDesc and related ops
void populateMatrixOpConversionPatterns(TypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns
      .add<CreateMemDescOpPattern, LoadMatrixOpPattern, StoreMatrixOpPattern>(
          converter, patterns.getContext());
}

struct MaterializeMatrixOpPass
    : public imex::impl::MaterializeMatrixOpBase<MaterializeMatrixOpPass> {
  void runOnOperation() override {
    auto mod = getOperation();
    MLIRContext &ctx = getContext();
    TypeConverter typeConverter;

    // Since SLM access instructions on Xe2 and Xe3 operate on 32-bit or
    // 64-bit units, all data types smaller than 32 bits has to be converted
    // to 32 bits.
    typeConverter.addConversion([&](Type type) -> Type {
      if (type.isInteger() && type.getIntOrFloatBitWidth() < 32)
        return IntegerType::get(type.getContext(), 32);
      if (type.isFloat() && type.getIntOrFloatBitWidth() < 32)
        return Float32Type::get(type.getContext());
      return type;
    });

    // Convert MemDescType into flattend 32-bit MemRefType for SLM
    typeConverter.addConversion([&](MemDescType type) -> Type {
      Type elemTy = type.getElementType();
      int packSize = getVnniFactor(elemTy);
      elemTy = typeConverter.convertType(elemTy);
      int numElems = type.getNumElements() / packSize;
      // TODO: Currently, an I64Attr(3) is assumed to represent the address
      // space in memref.alloc. This should be standardized for consistency
      // in XeGPU.
      return MemRefType::get(numElems, elemTy, AffineMap(), 3);
    });

    ConversionTarget target(ctx);
    target.addIllegalOp<CreateMemDescOp, LoadMatrixOp, StoreMatrixOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });

    RewritePatternSet patterns(&ctx);
    populateMatrixOpConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::createCSEPass());
    if (mlir::failed(pm.run(mod)))
      signalPassFailure();
  }
};

} // namespace

namespace imex {
std::unique_ptr<Pass> createMaterializeMatrixOpPass() {
  return std::make_unique<MaterializeMatrixOpPass>();
}
} // namespace imex
