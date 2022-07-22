//===- imex-runner.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the IMEX runner.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
//#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>

#include <imex/InitIMEXDialects.h>
#include <imex/InitIMEXPasses.h>

static ::mlir::LogicalResult runIMEXPasses(::mlir::ModuleOp module) {
  ::mlir::PassManager passManager(module.getContext());
  ::mlir::applyPassManagerCLOptions(passManager);

  passManager.addPass(::imex::createConvertPTensorToLinalgPass());
  passManager.addPass(::imex::createDistElimPass());
  passManager.addPass(::mlir::createConvertShapeToStandardPass());

  passManager.addPass(::mlir::arith::createConstantBufferizePass());
  passManager.addNestedPass<::mlir::func::FuncOp>(
      ::mlir::createSCFBufferizePass());
  passManager.addNestedPass<::mlir::func::FuncOp>(
      ::mlir::createShapeBufferizePass());
  // passManager.addNestedPass<::mlir::func::FuncOp>(::mlir::createLinalgInitTensorToAllocTensorPass());
  passManager.addNestedPass<::mlir::func::FuncOp>(
      ::mlir::createLinalgBufferizePass());
  // passManager.addNestedPass<::mlir::func::FuncOp>(::mlir::createBufferizationBufferizePass());
  passManager.addNestedPass<::mlir::func::FuncOp>(
      ::mlir::createTensorBufferizePass());
  passManager.addPass(::mlir::func::createFuncBufferizePass());
  passManager.addNestedPass<::mlir::func::FuncOp>(
      ::mlir::bufferization::createFinalizingBufferizePass());
  // passManager.addNestedPass<mlir::func::FuncOp>(
  //     bufferization::createBufferDeallocationPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      ::mlir::createConvertLinalgToParallelLoopsPass());
  //   passManager.addNestedPass<mlir::func::FuncOp>(
  //       gpu_runtime::createParallelLoopGPUMappingPass());
  //   passManager.addNestedPass<mlir::func::FuncOp>(createParallelLoopToGpuPass());

  //  passManager.addNestedPass<mlir::func::FuncOp>(
  //      gpu_runtime::createInsertGPUAllocsPass());
  passManager.addPass(::mlir::createCanonicalizerPass());
  //  passManager.addNestedPass<mlir::func::FuncOp>(
  //      gpu_runtime::createUnstrideMemrefsPass());
  passManager.addNestedPass<::mlir::func::FuncOp>(
      mlir::createLowerAffinePass());

  //  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(::mlir::memref::createFoldSubViewOpsPass());
  //  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
  //      gpu_runtime::createAbiAttrsPass());
  //  passManager.addPass(gpu_runtime::createSetSPIRVCapabilitiesPass());

  //  passManager.addPass(gpu_runtime::createGPUToSpirvPass());
  //   OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  //   modulePM.addPass(spirv::createLowerABIAttributesPass());
  //   modulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
  ::mlir::LowerToLLVMOptions llvmOptions(module.getContext(),
                                         ::mlir::DataLayout(module));
  // passManager.nest<::mlir::func::FuncOp>().addPass(::mlir::LLVM::createRequestCWrappersPass());

  // Gpu -> GpuRuntime
  //   passManager.addPass(gpu_runtime::createSerializeSPIRVPass());
  //   passManager.addNestedPass<mlir::func::FuncOp>(gpu_runtime::createGPUExPass());

  // GpuRuntime -> LLVM

  //   passManager.addPass(gpu_runtime::createEnumerateEventsPass());
  //   passManager.addPass(gpu_runtime::createGPUToLLVMPass());
  passManager.addPass(::mlir::createConvertFuncToLLVMPass(llvmOptions));
  passManager.addPass(::mlir::createMemRefToLLVMPass());
  passManager.addPass(::mlir::createReconcileUnrealizedCastsPass());

  return passManager.run(module);
}

int main(int argc, char **argv) {
  ::llvm::llvm_shutdown_obj x;
  ::mlir::registerPassManagerCLOptions();

  ::llvm::InitLLVM y(argc, argv);
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();

  ::mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runIMEXPasses;

  ::mlir::DialectRegistry registry;
  registry.insert<::mlir::arith::ArithmeticDialect, ::mlir::LLVM::LLVMDialect,
                  //   ::mlir::gpu::GPUDialect, ::mlir::spirv::SPIRVDialect,
                  ::mlir::func::FuncDialect, ::mlir::memref::MemRefDialect,
                  ::mlir::linalg::LinalgDialect, ::mlir::tensor::TensorDialect,
                  ::mlir::shape::ShapeDialect, ::mlir::AffineDialect>();
  ::mlir::registerLLVMDialectTranslation(registry);
  ::imex::registerAllDialects(registry);

  return ::mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
