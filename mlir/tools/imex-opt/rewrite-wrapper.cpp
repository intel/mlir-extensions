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

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "mlir-extensions/Conversion/SCFToAffine/SCFToAffine.h"
#include "mlir-extensions/Conversion/gpu_runtime_to_llvm.hpp"
#include "mlir-extensions/Conversion/gpu_to_gpu_runtime.hpp"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

static mlir::PassPipelineRegistration<>
    ParallelLoopToGpu("parallel-loop-to-gpu", "Maps scf parallel loop to gpu",
                      [](mlir::OpPassManager &pm) {
                        pm.addNestedPass<mlir::func::FuncOp>(
                            gpu_runtime::createParallelLoopGPUMappingPass());
                      });

static mlir::PassPipelineRegistration<>
    InsertGpuAlloc("insert-gpu-alloc", "Converts memref alloc to gpu alloc",
                   [](mlir::OpPassManager &pm) {
                     pm.addNestedPass<mlir::func::FuncOp>(
                         gpu_runtime::createInsertGPUAllocsPass());
                   });

static mlir::PassPipelineRegistration<>
    UnstrideMemrefPass("unstride-memref", "Used to flatten 2D to 1D",
                       [](mlir::OpPassManager &pm) {
                         pm.addNestedPass<mlir::func::FuncOp>(
                             gpu_runtime::createUnstrideMemrefsPass());
                       });

static mlir::PassPipelineRegistration<>
    AbiAttrsPass("abi-attrs", "Create AbiAttrs Pass",
                 [](mlir::OpPassManager &pm) {
                   pm.addNestedPass<mlir::gpu::GPUModuleOp>(
                       gpu_runtime::createAbiAttrsPass());
                 });

static mlir::PassPipelineRegistration<>
    SetSpirvCapabalities("set-spirv-capablilities", "Sets Spirv capabilities",
                         [](mlir::OpPassManager &pm) {
                           pm.addPass(gpu_runtime::createGPUToSpirvPass());
                         });

static mlir::PassPipelineRegistration<>
    GpuToSpirv("gpu-to-spirv", "Converts Gpu to spirv module",
               [](mlir::OpPassManager &pm) {
                 pm.addPass(gpu_runtime::createGPUToSpirvPass());
               });

static mlir::PassPipelineRegistration<>
    SerializeSpirv("serialize-spirv", "Serializes the spir-v binary",
                   [](mlir::OpPassManager &pm) {
                     pm.addPass(gpu_runtime::createSerializeSPIRVPass());
                   });

static mlir::PassPipelineRegistration<> GpuToGpuRuntime(
    "gpu-to-gpu-runtime", "Converts Gpu ops to gpu runteim",
    [](mlir::OpPassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(gpu_runtime::createGPUExPass());
    });

static mlir::PassPipelineRegistration<>
    EnumerateEvents("enumerate-events", "Adds event dependency",
                    [](mlir::OpPassManager &pm) {
                      pm.addPass(gpu_runtime::createEnumerateEventsPass());
                    });

static mlir::PassPipelineRegistration<>
    GpuToLlvm("convert-gpu-to-llvm",
              "Converts Gpu runtime dialect to llvm runtime calls",
              [](mlir::OpPassManager &pm) {
                pm.addPass(gpu_runtime::createGPUToLLVMPass());
              });

static mlir::PassPipelineRegistration<> scfToAffineReg(
    "scf-to-affine", "Converts SCF parallel struct into Affine parallel",
    [](mlir::OpPassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFToAffinePass());
    });
