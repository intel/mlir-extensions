//===-------- ConvertToSPIRV.cpp - one shot convert-to-spirv pass --------===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Generate a convert-to-spirv pass. Add all `to-spirv` patterns from all the
// dialects.

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/DebugLog.h"
#include <optional>

#include "imex/Conversion/ConvertToSPIRV/ConvertToSPIRV.h"
#include "imex/Conversion/Passes.h"

namespace imex {
#define GEN_PASS_DEF_CONVERTTOSPIRV
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;
namespace imex {
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
    assert(vWidth <= 64 && "vector.create_mask supports vector widths <= 64");
    auto vWidthConst = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(vWidth));
    auto maskVal = adaptor.getOperands()[0];
    maskVal = rewriter.create<mlir::arith::TruncIOp>(
        vMaskOp.getLoc(), rewriter.getI64Type(), maskVal);

    // maskVal < vWidth
    auto cmp = rewriter.create<mlir::arith::CmpIOp>(
        vMaskOp.getLoc(), mlir::arith::CmpIPredicate::slt, maskVal,
        vWidthConst);
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(1));
    auto shift = rewriter.create<mlir::spirv::ShiftLeftLogicalOp>(
        vMaskOp.getLoc(), one, maskVal);
    auto mask1 =
        rewriter.create<mlir::arith::SubIOp>(vMaskOp.getLoc(), shift, one);
    auto mask2 = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(-1)); // all ones
    mlir::Value sel = rewriter.create<mlir::arith::SelectOp>(vMaskOp.getLoc(),
                                                             cmp, mask1, mask2);

    // maskVal < 0
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        vMaskOp.getLoc(), rewriter.getI64IntegerAttr(0));
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

void populateIMEXVectorToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                       mlir::RewritePatternSet &patterns) {
  patterns
      .add<VectorFromElementsConversionPattern, VectorMaskConversionPattern>(
          typeConverter, patterns.getContext());
}
} // namespace imex

namespace {

// Populate upstream conversion patterns for each dialect.
void populateUpstreamConvertToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter,
    ScfToSPIRVContext &scfToSPIRVContext, RewritePatternSet &patterns) {
  arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  arith::populateArithToSPIRVPatterns(typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
  cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
  populateComplexToSPIRVPatterns(typeConverter, patterns);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateGPUToSPIRVPatterns(typeConverter, patterns);
  index::populateIndexToSPIRVPatterns(typeConverter, patterns);
  populateMathToSPIRVPatterns(typeConverter, patterns);
  populateMemRefToSPIRVPatterns(typeConverter, patterns);
  populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);
  populateTensorToSPIRVPatterns(typeConverter,
                                /*byteCountThreshold=*/64, patterns);
  ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);
  mlir::populateVectorToSPIRVPatterns(typeConverter, patterns);
}

struct ConvertToSPIRVPass
    : public imex::impl::ConvertToSPIRVBase<ConvertToSPIRVPass> {
  void runOnOperation() override;

private:
  // Queries the target environment from 'targets' attribute of the given
  // `moduleOp`.
  spirv::TargetEnvAttr lookupTargetEnvInTargets(gpu::GPUModuleOp moduleOp);

  // Queries the target environment from 'targets' attribute of the given
  // `moduleOp` or returns target environment as returned by
  // `spirv::lookupTargetEnvOrDefault` if not provided by 'targets'.
  spirv::TargetEnvAttr lookupTargetEnvOrDefault(gpu::GPUModuleOp moduleOp);
  // Map memRef memory space to SPIR-V storage class.
  void mapToMemRef(Operation *op, spirv::TargetEnvAttr &targetAttr);
  // bool mapMemorySpace;
};

spirv::TargetEnvAttr
ConvertToSPIRVPass::lookupTargetEnvInTargets(gpu::GPUModuleOp moduleOp) {
  if (ArrayAttr targets = moduleOp.getTargetsAttr()) {
    for (Attribute targetAttr : targets)
      if (auto spirvTargetEnvAttr = dyn_cast<spirv::TargetEnvAttr>(targetAttr))
        return spirvTargetEnvAttr;
  }
  return {};
}

spirv::TargetEnvAttr
ConvertToSPIRVPass::lookupTargetEnvOrDefault(gpu::GPUModuleOp moduleOp) {
  if (spirv::TargetEnvAttr targetEnvAttr = lookupTargetEnvInTargets(moduleOp))
    return targetEnvAttr;
  return spirv::lookupTargetEnvOrDefault(moduleOp);
}

// Map memRef memory space to SPIR-V storage class.
void ConvertToSPIRVPass::mapToMemRef(Operation *op,
                                     spirv::TargetEnvAttr &targetAttr) {
  spirv::TargetEnv targetEnv(targetAttr);
  bool targetEnvSupportsKernelCapability =
      targetEnv.allows(spirv::Capability::Kernel);
  spirv::MemorySpaceToStorageClassMap memorySpaceMap =
      targetEnvSupportsKernelCapability
          ? spirv::mapMemorySpaceToOpenCLStorageClass
          : spirv::mapMemorySpaceToVulkanStorageClass;
  spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);
  spirv::convertMemRefTypesAndAttrs(op, converter);

  // Check if there are any illegal ops remaining.
  std::unique_ptr<ConversionTarget> target =
      spirv::getMemorySpaceToStorageClassTarget(*op->getContext());

  op->walk([&target, this](Operation *childOp) {
    if (target->isIllegal(childOp)) {
      childOp->emitOpError("failed to legalize memory space");
      signalPassFailure(); // Now this works because it's a member function
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

void ConvertToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  SmallVector<Operation *, 1> gpuModules;
  OpBuilder builder(context);

  auto targetEnvSupportsKernelCapability = [this](gpu::GPUModuleOp moduleOp) {
    auto targetAttr = lookupTargetEnvOrDefault(moduleOp);
    spirv::TargetEnv targetEnv(targetAttr);
    return targetEnv.allows(spirv::Capability::Kernel);
  };

  op->walk([&](gpu::GPUModuleOp moduleOp) {
    // Clone each GPU kernel module for conversion, given that the GPU
    // launch op still needs the original GPU kernel module.
    // For Vulkan Shader capabilities, we insert the newly converted SPIR-V
    // module right after the original GPU module, as that's the expectation
    // of the in-tree SPIR-V CPU runner (the Vulkan runner does not use this
    // pass). For OpenCL Kernel capabilities, we insert the newly converted
    // SPIR-V module inside the original GPU module, as that's the expectaion
    // of the normal GPU compilation pipeline.
    if (targetEnvSupportsKernelCapability(moduleOp)) {
      builder.setInsertionPointToStart(moduleOp.getBody());
    } else {
      builder.setInsertionPoint(moduleOp.getOperation());
    }
    gpuModules.push_back(builder.clone(*moduleOp.getOperation()));
  });

  // Run conversion for each gpu module independently as they can have
  // different TargetEnv attributes.
  for (Operation *gpuModule : gpuModules) {
    // Configure conversion target
    auto castedGPUModule = mlir::dyn_cast<gpu::GPUModuleOp>(*gpuModule);
    spirv::TargetEnvAttr targetAttr = lookupTargetEnvOrDefault(castedGPUModule);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    // Set up type converter with SPIR-V type conversion and pass options
    SPIRVConversionOptions options;
    options.emulateLT32BitScalarTypes = this->emulateLT32BitScalarTypes;
    options.emulateUnsupportedFloatTypes = this->emulateUnsupportedFloatTypes;
    options.use64bitIndex = this->use64bitIndex;
    options.boolNumBits = this->boolNumBits;
    SPIRVTypeConverter typeConverter(targetAttr, options);

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

    // Add all to-SPIRV conversion patterns
    RewritePatternSet patterns(context);
    // Upstream patterns
    ScfToSPIRVContext scfToSPIRVContext;
    mapToMemRef(gpuModule, targetAttr);
    populateUpstreamConvertToSPIRVPatterns(typeConverter, scfToSPIRVContext,
                                           patterns);
    // IMEX patterns
    imex::populateIMEXVectorToSPIRVPatterns(typeConverter, patterns);
    // Apply conversion
    if (failed(applyFullConversion(gpuModule, *target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
  // For OpenCL, the gpu.func op in the original gpu.module op needs to be
  // replaced with an empty func.func op with the same arguments as the
  // gpu.func op. The func.func op needs gpu.kernel attribute set.
  op->walk([&](gpu::GPUModuleOp moduleOp) {
    if (targetEnvSupportsKernelCapability(moduleOp)) {
      moduleOp.walk([&](gpu::GPUFuncOp funcOp) {
        builder.setInsertionPoint(funcOp);
        auto newFuncOp =
            func::FuncOp::create(builder, funcOp.getLoc(), funcOp.getName(),
                                 funcOp.getFunctionType());
        auto entryBlock = newFuncOp.addEntryBlock();
        builder.setInsertionPointToEnd(entryBlock);
        func::ReturnOp::create(builder, funcOp.getLoc());
        newFuncOp->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                           builder.getUnitAttr());
        funcOp.erase();
      });
    }
  });
}

} // namespace

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
imex::createConvertToSPIRVPass() {
  return std::make_unique<ConvertToSPIRVPass>();
}
