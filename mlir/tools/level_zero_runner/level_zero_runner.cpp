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

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>

#include "mlir-extensions/Conversion/gpu_runtime_to_llvm.hpp"
#include "mlir-extensions/Conversion/gpu_to_gpu_runtime.hpp"

using namespace mlir;

static LogicalResult runMLIRPasses(ModuleOp module) {
  PassManager passManager(module.getContext());
  applyPassManagerCLOptions(passManager);

  /*
    passManager.addNestedPass<mlir::FuncOp>(
        arith::createArithmeticBufferizePass());
    passManager.addNestedPass<mlir::FuncOp>(createLinalgBufferizePass());
    passManager.addNestedPass<mlir::FuncOp>(createSCFBufferizePass());
    passManager.addNestedPass<mlir::FuncOp>(createTensorBufferizePass());
    passManager.addPass(createFuncBufferizePass());
    passManager.addNestedPass<mlir::FuncOp>(
        bufferization::createFinalizingBufferizePass());
    passManager.addNestedPass<mlir::FuncOp>(
        bufferization::createBufferDeallocationPass());
    passManager.addNestedPass<mlir::FuncOp>(
        createConvertLinalgToParallelLoopsPass());

    // passManager.addPass(greedilyMapParallelSCFToGPU());
    passManager.addNestedPass<mlir::FuncOp>(createParallelLoopToGpuPass());
    passManager.addNestedPass<mlir::FuncOp>(createLowerAffinePass());
  */
  passManager.addNestedPass<mlir::FuncOp>(
      gpu_runtime::runInsertGPUAllocsPass());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addNestedPass<mlir::FuncOp>(
      gpu_runtime::runUnstrideMemrefsPass());
  passManager.addNestedPass<mlir::FuncOp>(mlir::createLowerAffinePass());
  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      gpu_runtime::runAbiAttrsPass());
  passManager.addPass(gpu_runtime::runSetSPIRVCapabilitiesPass());
  // passManager.addPass(createConvertMemRefToSPIRVPass());
  passManager.addPass(gpu_runtime::runGPUToSpirvPass());
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createLowerABIAttributesPass());
  modulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
  LowerToLLVMOptions llvmOptions(module.getContext(), DataLayout(module));
  llvmOptions.emitCWrappers = true;

  // Gpu -> GpuRuntime
  passManager.addPass(gpu_runtime::runSerializeSPIRVPass());
  passManager.addNestedPass<mlir::FuncOp>(gpu_runtime::runGPUExPass());

  // GpuRuntime -> LLVM
  passManager.addPass(createLowerToLLVMPass(llvmOptions));
  passManager.addPass(createMemRefToLLVMPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());
  passManager.addPass(gpu_runtime::runEnumerateEventsPass());
  passManager.addPass(gpu_runtime::runGPUToLLVMPass());

  return passManager.run(module);
}

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::LLVM::LLVMDialect,
                  mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect,
                  mlir::StandardOpsDialect, mlir::memref::MemRefDialect,
                  mlir::linalg::LinalgDialect>();
  mlir::registerLLVMDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
