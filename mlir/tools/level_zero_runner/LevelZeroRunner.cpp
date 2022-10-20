// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/ExecutionEngine/JitRunner.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>

#include "imex/Conversion/GpuRuntimeToLlvm.hpp"
#include "imex/Conversion/GpuToGpuRuntime.hpp"
#include "imex/Conversion/UtilToLlvm.hpp"

using namespace mlir;

static LogicalResult runMLIRPasses(mlir::Operation *op) {
  auto module = mlir::cast<mlir::ModuleOp>(op);
  PassManager passManager(module.getContext());
  applyPassManagerCLOptions(passManager);

  passManager.addPass(arith::createConstantBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(createSCFBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      bufferization::createEmptyTensorToAllocTensorPass());
  passManager.addNestedPass<mlir::func::FuncOp>(createLinalgBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      bufferization::createBufferizationBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(createTensorBufferizePass());
  passManager.addPass(func::createFuncBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      bufferization::createFinalizingBufferizePass());
  // passManager.addNestedPass<mlir::func::FuncOp>(
  //     bufferization::createBufferDeallocationPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createConvertLinalgToParallelLoopsPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createGpuMapParallelLoopsPass());
  passManager.addNestedPass<mlir::func::FuncOp>(createParallelLoopToGpuPass());

  passManager.addNestedPass<mlir::func::FuncOp>(
      gpu_runtime::createInsertGPUAllocsPass());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      gpu_runtime::createUnstrideMemrefsPass());
  passManager.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());

  passManager.addPass(createGpuKernelOutliningPass());
  //  passManager.addPass(memref::createFoldSubViewOpsPass());
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      gpu_runtime::createAbiAttrsPass());
  passManager.addPass(gpu_runtime::createSetSPIRVCapabilitiesPass());

  passManager.addPass(gpu_runtime::createGPUToSpirvPass());
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createLowerABIAttributesPass());
  modulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
  LowerToLLVMOptions llvmOptions(module.getContext(), DataLayout(module));
  passManager.nest<func::FuncOp>().addPass(LLVM::createRequestCWrappersPass());

  // Gpu -> GpuRuntime
  passManager.addPass(gpu_runtime::createSerializeSPIRVPass());
  passManager.addNestedPass<mlir::func::FuncOp>(gpu_runtime::createGPUExPass());

  // GpuRuntime -> LLVM

  passManager.addPass(gpu_runtime::createEnumerateEventsPass());
  passManager.addPass(createConvertFuncToLLVMPass(llvmOptions));
  passManager.addPass(gpu_runtime::createGPUToLLVMPass());
  passManager.addPass(
      imex::createUtilToLLVMPass([&](MLIRContext &) { return llvmOptions; }));
  passManager.addPass(createMemRefToLLVMConversionPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  return passManager.run(module);
}

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  mlir::registerLLVMDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
