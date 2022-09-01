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
  let constructor = "imex::createSerializeSPIRVPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::spirv::SPIRVDialect"
    ];
}

#endif // _IMEX_TRANSFORMS_PASSES_TD_INCLUDED_
