//===-- Passes.td - XeTile pass definition file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the XeTile dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XeTile_PASSES_TD_INCLUDED_
#define _XeTile_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// XeTileToXeGPU pass
//===----------------------------------------------------------------------===//

// def XeTileToXeGPU: Pass<"xetile-xegpu", "::mlir::func::FuncOp"> {
//   let summary = "Add a pass for lowering XeTile dialect ops to XeGPU";
//   let description = [{
//
//     #### Input invariant
//
//
//     #### Output IR
//
//   }];
//   let constructor = "::imex::createXeTileToXeGPUPass()";
//   let dependentDialects = ["::imex::xetile::XeTileDialect"];
//   let options = [];
// }

#endif // _XeTile_PASSES_TD_INCLUDED_
