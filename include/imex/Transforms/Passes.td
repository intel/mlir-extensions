//===-- Passes.td - Transform pass definition file ---------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
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
           "The client API to use for inserting gpu allocs">
  ];
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
    Computation is replace by first extending bf16 to f32, do the compute in f32
    and truncate result back to bf16.
  }];
  let constructor = "imex::createBF16ToGPUPass()";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
    ];
}
#endif // _IMEX_TRANSFORMS_PASSES_TD_INCLUDED_
