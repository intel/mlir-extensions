//===-- Passes.td - DistRuntime pass definition file --------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the DistRuntime dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _DistRuntime_PASSES_TD_INCLUDED_
#define _DistRuntime_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerDistRuntimeToIDTR pass
//===----------------------------------------------------------------------===//

def LowerDistRuntimeToIDTR: Pass<"lower-distruntime-to-idtr"> {
  let summary = "Lower distruntime dialect to calls to IDTR";
  let constructor = "::imex::createDistRuntimeToIDTRPass()";
  let dependentDialects = ["::mlir::func::FuncDialect",
                           "::mlir::memref::MemRefDialect",
                           "::imex::ndarray::NDArrayDialect",
                           "::mlir::bufferization::BufferizationDialect"];
  let options = [];
}

#endif // _DistRuntime_PASSES_TD_INCLUDED_
