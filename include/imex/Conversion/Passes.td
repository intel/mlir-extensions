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

    -   all tensors are PTensorType;

    #### Output IR

    - PTensorTypes are converted to mlir::Tensor
    - PTensor operations are converted to Linalg operations, accompaigned by
      operations of the Dist dialect if the input tensors are distributed
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
