// Copyright 2021 Intel Corporation
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

#include "pipelines/lower_to_gpu.hpp"

#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Dialect/GPU/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "base_pipeline.hpp"
#include "pipelines/lower_to_llvm.hpp"

#include "plier/compiler/pipeline_registry.hpp"

namespace {
struct ParallelLoopGPUMappingPass
    : public mlir::PassWrapper<ParallelLoopGPUMappingPass, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnFunction() override {
    mlir::greedilyMapParallelSCFToGPU(getFunction().getBody());
  }
};

struct AbiAttrsPass
    : public mlir::PassWrapper<AbiAttrsPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    auto attrName = mlir::spirv::getEntryPointABIAttrName();
    const int32_t sizes[] = {1, 1, 1};
    auto abi = mlir::spirv::getEntryPointABIAttr(sizes, context);
    for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
      if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
          gpuFunc->getAttr(attrName))
        continue;

      gpuFunc->setAttr(attrName, abi);
    }
  }
};

static void populateLowerToGPUPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::FuncOp>(
      std::make_unique<ParallelLoopGPUMappingPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createParallelLoopToGpuPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(std::make_unique<AbiAttrsPass>());
  pm.addPass(mlir::createConvertGPUToSPIRVPass());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerLowerToGPUPipeline(plier::PipelineRegistry &registry) {
  registry.register_pipeline([](auto sink) {
    auto stage = getLowerLoweringStage();
    sink(lowerToGPUPipelineName(), {stage.begin},
         {stage.end, lowerToLLVMPipelineName()}, {},
         &populateLowerToGPUPipeline);
  });
}

llvm::StringRef lowerToGPUPipelineName() { return "lower_to_gpu"; }
