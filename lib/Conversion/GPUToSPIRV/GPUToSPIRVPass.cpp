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

#include "../PassDetail.h"

#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include <mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

  // Map MemRef memory space to SPIR-V storage class first if requested.
  if (mapMemorySpace) {
    std::unique_ptr<mlir::ConversionTarget> target =
        mlir::spirv::getMemorySpaceToStorageClassTarget(*context);
    mlir::spirv::MemorySpaceToStorageClassMap memorySpaceMap =
        mlir::spirv::mapMemorySpaceToVulkanStorageClass;
    mlir::spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);

    mlir::RewritePatternSet patterns(context);
    mlir::spirv::populateMemorySpaceToStorageClassPatterns(converter, patterns);

    if (failed(applyFullConversion(gpuModules, *target, std::move(patterns))))
      return signalPassFailure();
  }

  auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(module);
  std::unique_ptr<mlir::ConversionTarget> target =
      mlir::SPIRVConversionTarget::get(targetAttr);

  mlir::SPIRVTypeConverter typeConverter(targetAttr);
  mlir::RewritePatternSet patterns(context);

  //------- Upstream Conversion------------
  mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
  mlir::arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
  mlir::populateMemRefToSPIRVPatterns(typeConverter, patterns);
  mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
  // ---------------------------------------

  // IMEX GPUToSPIRV extension
  mlir::ScfToSPIRVContext scfToSpirvCtx;
  mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
  mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
  mlir::populateMathToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyFullConversion(gpuModules, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUXToSPIRVPass>(mapMemorySpace);
}
} // namespace imex
