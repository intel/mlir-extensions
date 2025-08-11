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

//===----------------------------------------------------------------------===//
// CoalesceShardOps
//===----------------------------------------------------------------------===//

def CoalesceShardOps : Pass<"coalesce-shard-ops"> {
  let summary = "Coalesce shard operations from mesh dialect.";
  let description = [{
    Combine shard ops which would lead to resharding of tensors.
    This pass handles coalesing of shard ops which annotate ndarray's
    subview operations.
  }];
  let constructor = "imex::createCoalesceShardOpsPass()";
  let dependentDialects = ["::mlir::shard::ShardDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // _NDARRAY_PASSES_TD_INCLUDED_
