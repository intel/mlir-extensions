//===-- Passes.td - Transform pass definition file ---------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the base classes of IMEX conversion passes.
///
//===----------------------------------------------------------------------===//

#ifndef _IMEX_TRANSFORMS_PASSES_TD_INCLUDED_
#define _IMEX_TRANSFORMS_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

def SerializeSPIRVPass : Pass<"serialize-spirv", "::mlir::ModuleOp"> {
  let summary = "serialize MLIR SPIR-V module to SPIR-V binary";
  let description = [{
    This pass iterates all the SPIR-V modules in the top module and serializes
    each SPIR-V module to SPIR-V binary and then attachs the binary blob as a
    string attribute to the corresponding gpu module.
  }];
  let constructor = "imex::createSerializeSPIRVPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::spirv::SPIRVDialect"
    ];
}

def InsertGPUAllocs : Pass<"insert-gpu-allocs", "::mlir::func::FuncOp"> {
  let summary = "Converts memref allocs to gpu allocs";
  let constructor = "imex::createInsertGPUAllocsPass()";
  let dependentDialects = ["::mlir::memref::MemRefDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::arith::ArithDialect"];
  let options = [
    Option<"clientAPI", "client-api", "std::string", /*default=*/"\"opencl\"",
           "The client API to use for inserting gpu allocs">,
    Option<"inRegions", "in-regions", "bool", "false",
           "Add gpu allocs only for memref.AllocOps within GPU regions">,
    Option<"hostShared", "host-shared", "bool", "true",
           "If set, allocate shared memory accessible both on both host and device.">
  ];
}

def InsertGPUCopy : Pass<"insert-gpu-copy", "::mlir::func::FuncOp"> {
  let summary = "Converts memref.copy op to gpu.memcpy if within an env region.";
  let constructor = "imex::createInsertGPUCopyPass()";
  let dependentDialects = ["::mlir::memref::MemRefDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::arith::ArithDialect"];
}

def SetSPIRVCapabilities : Pass<"set-spirv-capabilities"> {
  let summary = "Sets Spirv capabilities";
  let constructor = "imex::createSetSPIRVCapabilitiesPass()";
  let dependentDialects = ["::mlir::spirv::SPIRVDialect"];
  let options = [
    Option<"clientAPI", "client-api", "std::string", /*default=*/"\"opencl\"",
           "The client API to use for setting Spirv capabilities">
  ];
}

def SetSPIRVAbiAttribute : Pass<"set-spirv-abi-attrs", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Sets Spirv Abi attribute";
  let constructor = "imex::createSetSPIRVAbiAttributePass()";
  let dependentDialects = ["::mlir::gpu::GPUDialect",
                           "::mlir::spirv::SPIRVDialect"];
  let options = [
    Option<"clientAPI", "client-api", "std::string", /*default=*/"\"opencl\"",
           "The client API to use for setting Spirv Abi attribute">
  ];
}

def AddOuterParallelLoop : Pass<"imex-add-outer-parallel-loop", "::mlir::func::FuncOp"> {
  let summary = "add an outer parallel loop when there is not";
  let description = [{
    When the original func does not have an outer parallel loop, this pass adds
    one so that the immediately followed pass gpu-map-parallel-loops can work.
  }];
  let constructor = "imex::createAddOuterParallelLoopPass()";
  let dependentDialects = [
    "::mlir::scf::SCFDialect"
    ];
}

def LowerMemRefCopy : Pass<"imex-lower-memref-copy", "::mlir::func::FuncOp"> {
  let summary = "lower memref.copy to linalg.generic";
  let description = [{
    This Pass transforms memref.copy to linalg.generic with identity index map and
    parallel iterator. If satisfied, this pass also does memref.copy canonicalization.

    This pass is supposed to work after bufferization and before linalg-lowering.
  }];
  let constructor = "imex::createLowerMemRefCopyPass()";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect"
    ];
}

def BF16ToGPU : Pass<"bf16-to-gpu", "::mlir::ModuleOp"> {
  let summary = "transform GPU dialect with bf16 to a form that can be lowered to spirv with Intel spirv extension ops.";
  let description = [{
    This pass transforms gpu dialect with bf16 dtype to a form that uses i16
    and f32 dtype that can be lowered to spirv dialect with Intel spirv extension ops.
    bf16 is bitcast to a bitwidth equal type i16 as bf16 is not a supported type
    in spirv.
    Computation is replaced by first extending bf16 to f32, do the compute in f32
    and truncate result back to bf16.
  }];
  let constructor = "imex::createBF16ToGPUPass()";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
    ];
}

def CastIndex : Pass<"cast-index", "::mlir::ModuleOp"> {
  let summary = "Place cast to and from i32 for compute intense arith ops in gpu.func using index type and replace index type with i32 for such ops.";
  let description = [{
    This pass collects compute intense arith ops in gpu.func using index type and
    replace with i32 type. Cast for index type to and from i32 is inserted before
     and after.
  }];
  let constructor = "imex::createCastIndexPass()";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::index::IndexDialect"
    ];
}

def EmulateNonNativeBF16 : Pass<"imex-emulate-non-native-bf16", "::mlir::gpu::GPUModuleOp"> {
  let summary = "transform gpu.func with bf16 emulation (upconvert and downconvert) for ops that are not natively supported";
  let description = [{
    This pass transforms a set of operations inside gpu.func
    whose respective lowered SPIR-V ops do not support bf16 data type natively.
    For the unsupported ops, computation is replaced by first extending bf16 to f32,
    do the compute in f32 and truncate result back to bf16 when appropiate.
  }];
  let constructor = "imex::createEmulateNonNativeBF16Pass()";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
    ];
}

def RemoveTemporaries : Pass<"imex-remove-temporaries"> {
  let summary = "Remove redundant memref.alloc and memref.copy operations";
  let description = [{
    This pass removes redundant temporary allocations, i.e. memref.alloc, memref.copy, and memref.dealloc operations when possible. A typical use case is in-place elementwise binary operations which often include a temporary memref allocation, linalg.generic loop, and memref.copy to the destination.

    This pass is intended to run after bufferization and buffer-deallocation.
  }];
  let constructor = "imex::createRemoveTemporariesPass()";
}

def VectorLinearize : Pass<"imex-vector-linearize"> {
  let summary = "Linearizes ND vectors into 1D for N >= 2";
  let constructor = "imex::createVectorLinearizePass()";
  let dependentDialects = [
    "::mlir::vector::VectorDialect"
  ];
}

def RemoveSingleElemVector : Pass<"imex-remove-single-elem-vector"> {
  let summary = "Remove <1xT> vectors, and replace them with scalars.";
  let constructor = "imex::createRemoveSingleElemVectorPass()";
  let dependentDialects = [
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def VnniTransformation : Pass<"imex-xegpu-apply-vnni-transformation"> {
  let summary = "apply vnni transformation for to B operand of dpas instructions if necessary.";
  let constructor = "imex::createVnniTransformationPass()";

  let dependentDialects = [
    "::mlir::vector::VectorDialect"
  ];
}

def OptimizeTranspose : Pass<"imex-xegpu-optimize-transpose"> {
  let summary = "Eliminate in-register vector transpose by fusing with load.";
  let constructor = "imex::createOptimizeTransposePass()";
  let dependentDialects = [
    "::mlir::xegpu::XeGPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::gpu::GPUDialect"
  ];
  let options = [
     Option<"device", "device", "std::string",
            /*default=*/"\"pvc\"",
            "gpu platform architecture where these ops are running">
 ];
}

def HoistTranspose : Pass<"imex-xegpu-hoist-transpose"> {
  let summary = "Move vector transpose ops closer to the load ops to enable load+tranpose fusion by OptimizeTranpose pass later in the pipeline.";
  let constructor = "imex::createHoistTransposePass()";
  let dependentDialects = [
    "::mlir::xegpu::XeGPUDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def TileLoops : Pass<"tile-loops", "::mlir::func::FuncOp"> {
  let summary = "Tile linalg.generic loops for GPU offloading";
  let description = [{
    Tiles loops defined with tensor inputs/outputs using the given tile sizes.
    This pass should be applied after loop fusion and before bufferization.
    Uses `TileUsingSCF` method. To map the loop to GPU blocks and threads this
    pass should be called twice. If `in-regions` is set, only loops within GPU
    regions are tiled.
  }];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"minTileFactor", "min-tile-factor", "int64_t", "2",
           "Minimum factor between dimension size and a tile size">,
    Option<"inRegions", "in-regions", "bool", "false",
           "Convert loops only within GPU regions">
  ];
  let constructor = "imex::createTileLoopsPass()";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def GpuXeVMAttachTarget: Pass<"xevm-attach-target", "mlir::gpu::GPUModuleOp"> {
  let summary = "Attaches a XeVM target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and attaches
    a XeVM target if the module matches the name specified by the `module` argument.

    Example:
    ```
    // File: in.mlir:
    gpu.module @nvvm_module_1 {...}
    gpu.module @rocdl_module_2 {...}
    gpu.module @xevm_module_3 {...}
    // mlir-opt --xevm-attach-target="module=xevm.* chip=pvc" in.mlir
    gpu.module @nvvm_module_1 {...}
    gpu.module @rocdl_module_2 {...}
    gpu.module @xevm_module_3 [#xevm.target<chip = "pvc">] {...}
    ```
  }];
  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"triple", "triple", "std::string",
           /*default=*/ "\"spirv64-unknown-unknown\"",
           "Target triple.">,
    Option<"chip", "chip", "std::string",
           /*default=*/"\"pvc\"",
           "Target chip.">,
    Option<"optLevel", "O", "unsigned",
           /*default=*/"2",
           "Optimization level.">,
    ListOption<"linkLibs", "l", "std::string",
                  "Extra bitcode libraries paths to link to.">,
    Option<"cmdOptions", "ocloc-cmd-options", "std::string",
                  /*default=*/ [{""}],
                  "Command line options passed to downstream compiler">
  ];
}


#endif // _IMEX_TRANSFORMS_PASSES_TD_INCLUDED_
