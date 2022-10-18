//===-- Passes.td - PTensor pass definition file --------------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the PTensor dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _PTENSOR_PASSES_TD_INCLUDED_
#define _PTENSOR_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// PTensorDist
//===----------------------------------------------------------------------===//

def PTensorDist : Pass<"ptensor-dist", "::mlir::func::FuncOp"> {
  let summary = "Use Dist-Ops to enable distributed PTensor Ops";
  let description = [{
    Transforms PTensor Ops into a sequence of operations to enable compute-follows-data
    for distributed memory. Using the Dist dialect for disribution operations.

    #### Output IR
    - Dist dialect
    - PTensor dialect
    - Linalg dialect
    - Shape dialect
    - Arithmetic dialect
  }];
  let constructor = "imex::createPTensorDistPass()";
  let dependentDialects = ["::imex::ptensor::PTensorDialect",
                           "::imex::dist::DistDialect",
                           "::mlir::arith::ArithmeticDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::shape::ShapeDialect"];
  let options = [];
}

#endif // _PTENSOR_PASSES_TD_INCLUDED_
