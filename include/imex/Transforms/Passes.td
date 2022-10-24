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

def UnstrideMemrefsPass : Pass<"unstride-memrefs", "::mlir::func::FuncOp"> {
  let summary = "convert 2D+ memrefs into 1D memrefs";
  let description = [{
    This pass iterates all memrefs and if the memref's rank is greater than one,
    the pass converts the memref into rank 1 and updates all usage of the previous
    memref with the new one and update access expressions accordingly.
  }];
  let constructor = "imex::createUnstrideMemrefsPass()";
  let dependentDialects = [
    "mlir::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::gpu::GPUDialect",
    "mlir::memref::MemRefDialect"
    ];
}

def InsertGPUAllocs : Pass<"insert-gpu-allocs", "::mlir::func::FuncOp"> {
  let summary = "Converts memref allocs to gpu allocs";
  let constructor = "imex::createInsertGPUAllocsPass()";
  let dependentDialects = ["::mlir::memref::MemRefDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::arith::ArithDialect"];
  let options = [
    Option<"clientAPI", "client-api", "std::string", /*default=*/"\"vulkan\"",
           "The client API to use for inserting gpu allocs">
  ];
}

#endif // _IMEX_TRANSFORMS_PASSES_TD_INCLUDED_
