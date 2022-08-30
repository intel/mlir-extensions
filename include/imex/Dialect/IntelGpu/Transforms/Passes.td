//===-- Passes.td - IntelGpu pass definition file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the IntelGpu dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _IntelGpu_PASSES_TD_INCLUDED_
#define _IntelGpu_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FIXME pass
//===----------------------------------------------------------------------===//

// def FIXME-passname: Pass<"FIXME-command-line", ""> {
//   let summary = "FIXME";
//   let description = [{
//     FIXME
//
//     #### Input invariant
//
//     - FIXME
//
//     #### Output IR
//
//     - FIXME
//   }];
//   let constructor = "::imex::createFIXMEPass()";
//   let dependentDialects = ["::imex::intel_gpu::IntelGpuDialect"];
//   let options = [];
// }

#endif // _IntelGpu_PASSES_TD_INCLUDED_
