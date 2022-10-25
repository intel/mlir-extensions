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
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "imex/Conversion/CfgToScf.hpp"
#include "imex/Conversion/GpuRuntimeToLlvm.hpp"
#include "imex/Conversion/GpuToGpuRuntime.hpp"
#include "imex/Conversion/NtensorToLinalg.hpp"
#include "imex/Conversion/NtensorToMemref.hpp"
#include "imex/Conversion/SCFToAffine/SCFToAffine.h"
#include "imex/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.hpp"
#include "imex/Dialect/ntensor/Transforms/ResolveArrayOps.hpp"
#include "imex/Transforms/ExpandTuple.hpp"
#include "imex/Transforms/MakeSignless.hpp"
#include "imex/Transforms/MemoryRewrites.hpp"

// Passes registration.

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
    AbiAttrsPass("set-spirv-abi-attrs", "Create AbiAttrs Pass",
                 [](mlir::OpPassManager &pm) {
                   pm.addNestedPass<mlir::gpu::GPUModuleOp>(
                       gpu_runtime::createAbiAttrsPass());
                 });

static mlir::PassPipelineRegistration<> SetSpirvCapabalities(
    "set-spirv-capablilities", "Sets spirv capablilities",
    [](mlir::OpPassManager &pm) {
      pm.addPass(gpu_runtime::createSetSPIRVCapabilitiesPass());
    });

static mlir::PassPipelineRegistration<>
    GpuToSpirv("gpux-to-spirv", "Converts Gpu to spirv module",
               [](mlir::OpPassManager &pm) {
                 pm.addPass(gpu_runtime::createGPUToSpirvPass());
               });

static mlir::PassPipelineRegistration<>
    SerializeSpirv("serialize-spirv", "Serializes the spir-v binary",
                   [](mlir::OpPassManager &pm) {
                     pm.addPass(gpu_runtime::createSerializeSPIRVPass());
                   });

static mlir::PassPipelineRegistration<> GpuToGpuRuntime(
    "gpu-to-gpux", "Converts Gpu ops to gpux", [](mlir::OpPassManager &pm) {
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

static mlir::PassPipelineRegistration<>
    cfgToScf("cfg-to-scf", "Convert function from CFG form to SCF ops",
             [](mlir::OpPassManager &pm) {
               pm.addNestedPass<mlir::func::FuncOp>(imex::createCFGToSCFPass());
             });

static mlir::PassPipelineRegistration<> expandTuple(
    "expand-tuple", "Expand tuple into individual elements",
    [](mlir::OpPassManager &pm) { pm.addPass(imex::createExpandTuplePass()); });

static mlir::PassPipelineRegistration<> ntensorResolveArrayOps(
    "ntensor-resolve-array-ops", "Resolve ntensor array ops into primitive ops",
    [](mlir::OpPassManager &pm) {
      pm.addPass(imex::ntensor::createResolveArrayOpsPass());
    });

static mlir::PassPipelineRegistration<>
    ntensorAliasAnalysis("ntensor-alias-analysis",
                         "Run alias analysis on ntensor ops",
                         [](mlir::OpPassManager &pm) {
                           pm.addPass(imex::createNtensorAliasAnalysisPass());
                         });

static mlir::PassPipelineRegistration<>
    ntensorToMemref("ntensor-to-memref", "Convert ntensor array ops to memref",
                    [](mlir::OpPassManager &pm) {
                      pm.addPass(imex::createNtensorToMemrefPass());
                    });

static mlir::PassPipelineRegistration<>
    ntensorToLinalg("ntensor-to-linalg", "Convert ntensor array ops to linalg",
                    [](mlir::OpPassManager &pm) {
                      pm.addPass(imex::createNtensorToLinalgPass());
                    });

static mlir::PassPipelineRegistration<> makeSignless(
    "imex-make-signless",
    "Convert types of various signedness to corresponding signless type",
    [](mlir::OpPassManager &pm) {
      pm.addPass(imex::createMakeSignlessPass());
    });

static mlir::PassPipelineRegistration<> makeBarriersUniform(
    "gpux-make-barriers-uniform",
    "Adapt gpu barriers to non-uniform control flow",
    [](mlir::OpPassManager &pm) {
      pm.addPass(gpu_runtime::createMakeBarriersUniformPass());
    });

static mlir::PassPipelineRegistration<> tileParallelLoopsGPU(
    "gpux-tile-parallel-loops", "Naively tile parallel loops for gpu",
    [](mlir::OpPassManager &pm) {
      pm.addPass(gpu_runtime::createTileParallelLoopsForGPUPass());
    });

static mlir::PassPipelineRegistration<> memoryOpts(
    "imex-memory-opts", "Apply memory optimizations",
    [](mlir::OpPassManager &pm) { pm.addPass(imex::createMemoryOptPass()); });
