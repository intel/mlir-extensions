//===- GPUToSPIRVPass.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file extends upstream GPUToSPIRV Pass that converts GPU ops to SPIR-V
/// by adding more conversion patterns like SCF, math and control flow. This
/// pass only converts gpu.func ops inside gpu.module op.
///
//===----------------------------------------------------------------------===//
#include "imex/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Debug.h>
#include <mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVTypes.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <mlir/Pass/Pass.h>

namespace imex {
#define GEN_PASS_DEF_CONVERTGPUXTOSPIRV
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

namespace imex {

/// Pass to lower GPU Dialect to SPIR-V. The pass only converts the gpu.func ops
/// inside gpu.module ops. i.e., the function that are referenced in
/// gpu.launch_func ops. For each such function
///
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
class GPUXToSPIRVPass : public impl::ConvertGPUXToSPIRVBase<GPUXToSPIRVPass> {
public:
  explicit GPUXToSPIRVPass(bool mapMemorySpace)
      : mapMemorySpace(mapMemorySpace) {}
  void runOnOperation() override;

private:
  bool mapMemorySpace;
};

// This op:
//   vector.create_mask %maskVal : vector<vWidth x i1>
// is lowered to:
//   if maskVal < 0
//     mask = 0
//   else if maskVal < vWidth
//     mask = (1 << maskVal) - 1
//   else
//     mask = all ones
class VectorMaskConversionPattern final
    : public mlir::OpConversionPattern<mlir::vector::CreateMaskOp> {
public:
  using OpConversionPattern<mlir::vector::CreateMaskOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::CreateMaskOp vMaskOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::VectorType vTy = vMaskOp.getVectorType();
    if (vTy.getRank() != 1)
      return mlir::failure();

    auto vWidth = vTy.getNumElements();
    auto vWidthConst = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI32IntegerAttr(vWidth));
    auto maskVal = adaptor.getOperands()[0];
    maskVal = rewriter.create<mlir::arith::TruncIOp>(
        vMaskOp.getLoc(), rewriter.getI32Type(), maskVal);

    // maskVal < vWidth
    auto cmp = rewriter.create<mlir::arith::CmpIOp>(
        vMaskOp.getLoc(), mlir::arith::CmpIPredicate::slt, maskVal,
        vWidthConst);
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    auto shift = rewriter.create<mlir::spirv::ShiftLeftLogicalOp>(
        vMaskOp.getLoc(), one, maskVal);
    auto mask1 =
        rewriter.create<mlir::arith::SubIOp>(vMaskOp.getLoc(), shift, one);
    auto mask2 = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(0xFFFFFFFF));
    mlir::Value sel = rewriter.create<mlir::arith::SelectOp>(vMaskOp.getLoc(),
                                                             cmp, mask1, mask2);

    // maskVal < 0
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto cmp2 = rewriter.create<mlir::arith::CmpIOp>(
        vMaskOp.getLoc(), mlir::arith::CmpIPredicate::slt, maskVal, zero);
    sel = rewriter.create<mlir::arith::SelectOp>(vMaskOp.getLoc(), cmp2, zero,
                                                 sel);

    sel = rewriter.create<mlir::arith::TruncIOp>(
        vMaskOp.getLoc(), rewriter.getIntegerType(vWidth), sel);
    auto res = rewriter.create<mlir::spirv::BitcastOp>(
        vMaskOp.getLoc(), mlir::VectorType::get({vWidth}, rewriter.getI1Type()),
        sel);
    vMaskOp->replaceAllUsesWith(res);
    rewriter.eraseOp(vMaskOp);
    return mlir::success();
  }
};

// Pattern to convert arith.truncf (f32 -> bf16) followed by arith.bitcast (bf16
// -> i16) to a SPIR-V convert op.
class ArithTruncFBitcastConversionPattern final
    : public mlir::OpConversionPattern<mlir::arith::TruncFOp> {
public:
  using OpConversionPattern<mlir::arith::TruncFOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::TruncFOp truncfOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    // Lamda to return the element type of truncfOp
    auto getOpElementType = [&](auto op) -> mlir::Type {
      if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(op.getType()))
        return vecTy.getElementType();
      return op.getType();
    };

    if (!getOpElementType(truncfOp).isBF16())
      return mlir::failure();

    if (!truncfOp->hasOneUse())
      return mlir::failure();

    // Check if the result of truncf is used by a bitcast op
    mlir::arith::BitcastOp bitcastOp =
        mlir::dyn_cast<mlir::arith::BitcastOp>(*(truncfOp->getUsers().begin()));
    // Check if the bitcast op is converting to i16
    if (!bitcastOp || !getOpElementType(bitcastOp).isInteger(16))
      return mlir::failure();

    mlir::arith::BitcastOpAdaptor bitcastOpAdaptor(bitcastOp);

    mlir::Value intelFToBF16ConvertionOp =
        rewriter.create<mlir::spirv::INTELConvertFToBF16Op>(
            truncfOp.getLoc(),
            getTypeConverter()->convertType(bitcastOp.getType()),
            adaptor.getOperands());

    rewriter.replaceOp(bitcastOp, intelFToBF16ConvertionOp);
    rewriter.eraseOp(truncfOp);
    return mlir::success();
  }
};

// Pattern to convert arith.bitcast (i16 -> bf16) followed by arith.extf (bf16
// -> f32) to a SPIR-V convert op.
class ArithBitcastExtFConversionPattern final
    : public mlir::OpConversionPattern<mlir::arith::BitcastOp> {
public:
  using OpConversionPattern<mlir::arith::BitcastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::BitcastOp bitcastOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Lamda to return the element type of bitcsatOp
    auto getOpElementType = [&](auto op) -> mlir::Type {
      if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(op.getType()))
        return vecTy.getElementType();
      return op.getType();
    };

    if (!getOpElementType(bitcastOp).isBF16())
      return mlir::failure();

    if (!bitcastOp->hasOneUse())
      return mlir::failure();

    // Check if the result of bitcast is used by an extf op
    mlir::arith::ExtFOp extfOp =
        mlir::dyn_cast<mlir::arith::ExtFOp>(*(bitcastOp->getUsers().begin()));
    // Check if the extf op is converting to f32
    if (!extfOp || !getOpElementType(extfOp).isF32())
      return mlir::failure();

    mlir::Value intelBF16ToFConvertionOp =
        rewriter.create<mlir::spirv::INTELConvertBF16ToFOp>(
            bitcastOp.getLoc(),
            getTypeConverter()->convertType(extfOp.getType()),
            adaptor.getOperands());

    rewriter.replaceOp(extfOp, intelBF16ToFConvertionOp);
    rewriter.eraseOp(bitcastOp);

    return mlir::success();
  }
};

void populateBF16ArithToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.add<ArithTruncFBitcastConversionPattern,
               ArithBitcastExtFConversionPattern>(typeConverter,
                                                  patterns.getContext());
}

// This pattern converts vector.from_elements op to SPIR-V CompositeInsertOp
class VectorFromElementsConversionPattern final
    : public mlir::OpConversionPattern<mlir::vector::FromElementsOp> {
public:
  using OpConversionPattern<mlir::vector::FromElementsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::FromElementsOp fromElementsOp,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::VectorType vecTy = fromElementsOp.getType();
    if (vecTy.getRank() > 1)
      return rewriter.notifyMatchFailure(fromElementsOp,
                                         "rank > 1 vectors are not supported");

    mlir::Type spirvVecTy = getTypeConverter()->convertType(vecTy);
    if (!spirvVecTy)
      return mlir::failure();

    // if the vector is just constructed from one element
    if (mlir::isa<mlir::spirv::ScalarType>(spirvVecTy)) {
      rewriter.replaceOp(fromElementsOp, adaptor.getElements()[0]);
      return mlir::success();
    }

    auto loc = fromElementsOp.getLoc();
    mlir::Value result = rewriter.create<mlir::spirv::UndefOp>(loc, spirvVecTy);
    for (auto [idx, val] : llvm::enumerate(adaptor.getElements())) {
      result = rewriter.create<mlir::spirv::CompositeInsertOp>(loc, val, result,
                                                               idx);
    }
    rewriter.replaceOp(fromElementsOp, result);
    return mlir::success();
  }
};

void populateVectorToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                   mlir::RewritePatternSet &patterns) {
  patterns
      .add<VectorFromElementsConversionPattern, VectorMaskConversionPattern>(
          typeConverter, patterns.getContext());
}

static bool isGenericVectorTy(mlir::Type type) {
  if (mlir::isa<mlir::spirv::ScalarType>(type))
    return true;
  auto vecSize = mlir::dyn_cast<mlir::VectorType>(type).getNumElements();
  return vecSize == 2 || vecSize == 3 || vecSize == 4 || vecSize == 8 ||
         vecSize == 16;
}

void GPUXToSPIRVPass::runOnOperation() {
  mlir::MLIRContext *context = &getContext();
  mlir::ModuleOp module = getOperation();

  llvm::SmallVector<mlir::Operation *, 1> gpuModules;
  mlir::OpBuilder builder(context);
  module.walk([&](mlir::gpu::GPUModuleOp moduleOp) {
    // For each kernel module (should be only 1 for now, but that is not a
    // requirement here), clone the module for conversion because the
    // gpu.launch function still needs the kernel module.
    builder.setInsertionPoint(moduleOp.getOperation());
    gpuModules.push_back(builder.clone(*moduleOp.getOperation()));
  });

  for (auto gpuModule : gpuModules) {

    // Map MemRef memory space to SPIR-V storage class first if requested.
    if (mapMemorySpace) {
      std::unique_ptr<mlir::ConversionTarget> target =
          mlir::spirv::getMemorySpaceToStorageClassTarget(*context);
      mlir::spirv::MemorySpaceToStorageClassMap memorySpaceMap =
          mlir::spirv::mapMemorySpaceToOpenCLStorageClass;
      mlir::spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);

      mlir::RewritePatternSet patterns(context);
      mlir::spirv::convertMemRefTypesAndAttrs(gpuModule, converter);

      if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
        return signalPassFailure();
    }

    auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(gpuModule);
    std::unique_ptr<mlir::ConversionTarget> target =
        mlir::SPIRVConversionTarget::get(targetAttr);

    mlir::RewritePatternSet patterns(context);
    mlir::SPIRVConversionOptions options;
    options.use64bitIndex = true;
    options.emulateLT32BitScalarTypes = false;

    mlir::SPIRVTypeConverter typeConverter(targetAttr, options);

    target->addDynamicallyLegalOp<mlir::spirv::INTELConvertBF16ToFOp>(
        [](mlir::spirv::INTELConvertBF16ToFOp) { return true; });
    target->addDynamicallyLegalOp<mlir::spirv::INTELConvertFToBF16Op>(
        [](mlir::spirv::INTELConvertFToBF16Op) { return true; });

    // SPIR-V elementwise arith/math ops require special handling if they
    // operate on large vectors. We dynamically legalize these ops based on
    // the vector size they consume.
    // FIXME: this is not an exhaustive list of arith/math ops that need
    // special handling.
    target->addDynamicallyLegalOp<mlir::spirv::CLExpOp>(
        [&](mlir::spirv::CLExpOp op) {
          return isGenericVectorTy(op.getType());
        });
    target->addDynamicallyLegalOp<mlir::spirv::CLFMaxOp>(
        [&](mlir::spirv::CLFMaxOp op) {
          return isGenericVectorTy(op.getType());
        });

    // Upstream SPIRVTypeConverter does not add conversion for
    // UnrankedMemRefType.
    // Conversion logic is the same as ranked dynamic memref type for OpenCL
    // Kernel. unranked memref type is converted to a spirv pointer type
    // with converted spirv scalar element type and spirv storage class.
    // Only scalar element type is currently supported.
    // Also vulkan should be handled differently but out of scope since this
    // conversion pass is for lowering to OpenCL spirv kernel only.
    typeConverter.addConversion(
        [&](mlir::UnrankedMemRefType type) -> std::optional<mlir::Type> {
          auto attr = mlir::dyn_cast_or_null<mlir::spirv::StorageClassAttr>(
              type.getMemorySpace());
          if (!attr)
            return nullptr;
          mlir::spirv::StorageClass storageClass = attr.getValue();

          mlir::Type elementType = type.getElementType();
          auto scalarType =
              mlir::dyn_cast<mlir::spirv::ScalarType>(elementType);
          if (!scalarType)
            return nullptr;
          mlir::Type arrayElemType = typeConverter.convertType(scalarType);
          return mlir::spirv::PointerType::get(arrayElemType, storageClass);
        });

    imex::populateBF16ArithToSPIRVPatterns(typeConverter, patterns);
    //------- Upstream Conversion------------
    mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
    mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    mlir::populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
    mlir::populateVectorToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);
    mlir::index::populateIndexToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMemRefToSPIRVPatterns(typeConverter, patterns);
    mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
    // ---------------------------------------

    // IMEX GPUToSPIRV extension
    mlir::ScfToSPIRVContext scfToSpirvCtx;
    mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);
    imex::populateVectorToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
      return signalPassFailure();
  }
}

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUXToSPIRVPass>(mapMemorySpace);
}
} // namespace imex
