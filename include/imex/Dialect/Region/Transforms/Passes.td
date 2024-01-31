//===-- Passes.td - Region pass definition file -----------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the Region dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Region_PASSES_TD_INCLUDED_
#define _Region_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DistCoalesce
//===----------------------------------------------------------------------===//

def RegionBufferize : Pass<"region-bufferize"> {
  let summary = "Bufferization of region ops";
  let description = [{
    Bufferize EnvironmentRegionOp and EnvironmentRegionYieldOp.
  }];
  let constructor = "imex::createRegionBufferizePass()";
  let dependentDialects = ["::mlir::bufferization::BufferizationDialect",
                           "::mlir::memref::MemRefDialect"];
  let options = [];
}

#endif // _Region_PASSES_TD_INCLUDED_
