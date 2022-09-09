//===-- IMEXPasses.td - Conversion pass definition file ----*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the base classes of IMEX conversino passes.
///
//===----------------------------------------------------------------------===//

#ifndef _IMEX_CONVERSION_PASSES_TD_INCLUDED_
#define _IMEX_CONVERSION_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// PTensorToLinalg
//===----------------------------------------------------------------------===//

def ConvertPTensorToLinalg : Pass<"convert-ptensor-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Convert from the PTensor dialect to the Linalg and Dist dialects";
  let description = [{
    Convert PTensor dialect operations into the LLVM IR dialect operations.

    #### Input invariant

    -   all tensors are PTensorType

    #### Output IR

    - a PTensorType will be lowered to a unrealized_conversion_cast to tuple of
      * rtensor: RankedTensor
      * device: device where the tensor lives (AnyType, default=None)
      * team: a group of processes among which the tensor is distributed
        (AnyType, default=None)
      * rthandle: a handle for communication with the dist dialect/runtime
        (AnyType, default=None)
    - On function boundaries (func::callOp, func::returnOp, func::functionOp)
      PTensorTypes get expanded into 4 separate arguments (rtensor, device, team, handle).
    - PTensor operations are converted to Linalg operations, accompaigned by
      * operations of the Dist dialect if the input tensors are distributed
      * FIXME iGPU::deviceRegionOps if the input tensors live on a device
  }];
  let constructor = "imex::createConvertPTensorToLinalgPass()";
  let dependentDialects = ["::imex::dist::DistDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::AffineDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::arith::ArithmeticDialect",
                           "::mlir::shape::ShapeDialect"];
  let options = [];
}

#endif // _IMEX_CONVERSION_PASSES_TD_INCLUDED_
