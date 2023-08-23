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
#include "imex/Conversion/XeGPUToSPIRV/XeGPUToSPIRV.h"
#include "imex/Dialect/XeGPU/IR/XeGPUOps.h"

#include "../PassDetail.h"

#include <mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>

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
class GPUXToSPIRVPass : public ::imex::ConvertGPUXToSPIRVBase<GPUXToSPIRVPass> {
public:
  explicit GPUXToSPIRVPass(bool mapMemorySpace)
      : mapMemorySpace(mapMemorySpace) {}
  void runOnOperation() override;

private:
  bool mapMemorySpace;
};

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
      mlir::spirv::populateMemorySpaceToStorageClassPatterns(converter,
                                                             patterns);

      if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
        return signalPassFailure();
    }

    auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(gpuModule);
    std::unique_ptr<mlir::ConversionTarget> target =
        mlir::SPIRVConversionTarget::get(targetAttr);

    mlir::RewritePatternSet patterns(context);
    mlir::SPIRVConversionOptions options;
    options.use64bitIndex = true;

    mlir::SPIRVTypeConverter typeConverter(targetAttr, options);

    /// Walk gpu.func and collect root ops for these two special patterns
    /// 1. Pattern to convert arith.truncf (f32 -> bf16) followed by
    ///    arith.bitcast (bf16 -> i16)
    ///    into a SPIR-V convert op.
    /// 2. Pattern to convert arith.bitcast (i16 -> bf16) followed by
    ///    arith.extf (bf16 -> f32)
    ///    into a SPIR-V convert op.
    /// And convert the patterns into spirv bf16<->f32 conversion ops.
    mlir::OpBuilder builder(gpuModule);
    llvm::SmallVector<mlir::Operation *, 16> eraseOps;
    gpuModule->walk([&](mlir::gpu::GPUFuncOp fop) {
      fop->walk([&](mlir::arith::BitcastOp bop) {
        if (bop.getType().isInteger(16)) {
          mlir::arith::TruncFOp inputOp = llvm::dyn_cast<mlir::arith::TruncFOp>(
              bop.getOperand().getDefiningOp());
          if (inputOp) {
            if (inputOp.getType().isBF16() &&
                inputOp.getOperand().getType().isF32()) {
              builder.setInsertionPoint(inputOp);
              auto narrow = builder.create<mlir::spirv::INTELConvertFToBF16Op>(
                  inputOp.getLoc(), builder.getI16Type(), inputOp.getOperand());
              bop->getResult(0).replaceAllUsesWith(narrow);
              eraseOps.push_back(bop);
              eraseOps.push_back(inputOp);
            }
          }
        }
      });
      fop->walk([&](mlir::arith::ExtFOp eop) {
        if (eop.getType().isF32()) {
          mlir::arith::BitcastOp inputOp =
              llvm::dyn_cast<mlir::arith::BitcastOp>(
                  eop.getOperand().getDefiningOp());
          if (inputOp) {
            if (inputOp.getType().isBF16() &&
                inputOp.getOperand().getType().isInteger(16)) {
              builder.setInsertionPoint(inputOp);
              auto widen = builder.create<mlir::spirv::INTELConvertBF16ToFOp>(
                  inputOp.getLoc(), builder.getF32Type(), inputOp.getOperand());
              eop->getResult(0).replaceAllUsesWith(widen);
              eraseOps.push_back(eop);
              eraseOps.push_back(inputOp);
            }
          }
        }
      });
    });
    target->addDynamicallyLegalOp<mlir::spirv::INTELConvertBF16ToFOp>(
        [](mlir::spirv::INTELConvertBF16ToFOp) { return true; });
    target->addDynamicallyLegalOp<mlir::spirv::INTELConvertFToBF16Op>(
        [](mlir::spirv::INTELConvertFToBF16Op) { return true; });
    for (auto eraseOp : eraseOps) {
      eraseOp->erase();
    }
    target->addIllegalDialect<imex::xegpu::XeGPUDialect>();
    typeConverter.addConversion([&](xegpu::TileType type) -> ::mlir::Type {
      return ::mlir::IntegerType::get(context, 64);
    });
    typeConverter.addConversion([&](::mlir::VectorType type) -> ::mlir::Type {
      unsigned rank = type.getRank();
      auto elemType = type.getElementType();
      if (rank < 1)
        return type;
      else {
        // load2d/store2d is vnni format with 3 dims
        if (rank == 3 && elemType.getIntOrFloatBitWidth() < 32) {
          elemType = ::mlir::IntegerType::get(context, 32);
          rank--;
        }
        unsigned sum = 1;
        for (unsigned i = 0; i < rank; i++) {
          sum *= type.getShape()[i];
        }
        return ::mlir::VectorType::get(sum, elemType);
      }
    });

    //------- Upstream Conversion------------
    mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
    mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMemRefToSPIRVPatterns(typeConverter, patterns);
    mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
    // ---------------------------------------

    // IMEX GPUToSPIRV extension
    mlir::ScfToSPIRVContext scfToSpirvCtx;
    mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);
    imex::populateXeGPUToVCIntrinsicsPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
      return signalPassFailure();
  }
}

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUXToSPIRVPass>(mapMemorySpace);
}
} // namespace imex
