//===-- Passes.td - NDArray pass definition file ------------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _NDARRAY_PASSES_TD_INCLUDED_
#define _NDARRAY_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// NDArrayDist
//===----------------------------------------------------------------------===//

def NDArrayDist : Pass<"ndarray-dist"> {
  let summary = "Use Dist-Ops to enable distributed NDArray Ops";
  let description = [{
    Transforms NDArray Ops into a sequence of operations to enable compute-follows-data
    for distributed memory. Using the Dist dialect for disribution operations.

    #### Output IR
    - Dist dialect
    - NDArray dialect
    - Linalg dialect
    - Arith dialect
  }];
  let constructor = "imex::createNDArrayDistPass()";
  let dependentDialects = ["::imex::ndarray::NDArrayDialect",
                           "::imex::dist::DistDialect",
                           "::imex::distruntime::DistRuntimeDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::memref::MemRefDialect"];
  let options = [];
}

def AddGPURegions : Pass<"add-gpu-regions"> {
  let summary = "Add RegionOps around NDArray Ops where applicable.";
  let description = [{
    Whenever a NDArray operation works on NDArrayTypes with GPUEnvAttr
    create a new RegionOp with the GPUEnvAttr and move the NDArray operation
    inside and let the RegionOp yield its result.
  }];
  let constructor = "imex::createAddGPURegionsPass()";
  let dependentDialects = ["::imex::region::RegionDialect"];
  let options = [];
}

#endif // _NDARRAY_PASSES_TD_INCLUDED_
