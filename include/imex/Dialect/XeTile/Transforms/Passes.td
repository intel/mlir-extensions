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

def XeTileTiling : Pass<"xetile-tiling", "::mlir::ModuleOp">{
  let summary = "transform XeTile large tiles(input) into register region block layout";

  let description = [{
    This pass transforms XeTile large tiles smaller tiles with blocked layout to map to register region.
    This blocked layout is represented by high dimension vectors, inner dimension matches to DPAS size
    config, This lowers 2D vector to 4D vector.
  }];

  let constructor = "imex::createXeTileTilingPass()";
  let dependentDialects = ["::imex::xetile::XeTileDialect"];
  let options = [];
}

#endif // _XeTile_PASSES_TD_INCLUDED_
