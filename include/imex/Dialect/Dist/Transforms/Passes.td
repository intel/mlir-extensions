//===-- Passes.td - Dist pass definition file --------------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the Dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Dist_PASSES_TD_INCLUDED_
#define _Dist_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DistElim
//===----------------------------------------------------------------------===//

def DistElim : Pass<"dist-elim", "::mlir::ModuleOp"> {
  let summary = "Eliminate Dist dialect";
  let description = [{
    Eliminate Dist dialect by replacing ops with node-local equivalents.

    #### Input invariant

    - Dist Operations were converted from PTensor

    #### Output IR

    - Shape dialect
  }];
  let constructor = "imex::createDistElimPass()";
  let dependentDialects = ["::mlir::shape::ShapeDialect"];
  let options = [];
}

#endif // _Dist_PASSES_TD_INCLUDED_
