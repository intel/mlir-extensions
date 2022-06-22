//===-- IMEXPasses.td - Conversion pass definition file ----*- tablegen -*-===//
//
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

#ifndef _IMEX_CONVERSION_PASSES_INCLUDED_
#define _IMEX_CONVERSION_PASSES_INCLUDED_

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
                           "::mlir::arith::ArithmeticDialect"];
  let options = [];
}

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

#endif // _IMEX_CONVERSION_PASSES_INCLUDED_
