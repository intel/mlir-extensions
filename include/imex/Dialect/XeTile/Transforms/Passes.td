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

def XeTileInitDuplicate : Pass<"xetile-init-duplicate", "::mlir::gpu::GPUModuleOp">{
  let summary = "It duplicates an initTileOp if it is used by load and prefetch, or "
                "load and store. ";

  let description = [{
    In some cases, a tile is created for many different uses for convinience. However,
    these uses may conflicts in repect to optimizaitons. For example, if a tile is created
    for both load and store, it may limits us to select an optimal parameter for either
    block load or block store due to the differences of hardware capabilities for these
    two instructions. To this end, it would duplicate the tile, such that one for load
    and one for store.
  }];

  let constructor = "imex::createXeTileInitDuplicatePass()";
  let dependentDialects = ["imex::xetile::XeTileDialect"];
}

def XeTileWgToSg : Pass<"xetile-wg-to-sg", "::mlir::gpu::GPUModuleOp">{
  let summary = "Transform WG level XeTile code to SG XeTile";

  let description = [{
    This transform pass transforms WG level XeTile code to SG XeTile.
  }];

  let constructor = "imex::createXeTileWgToSgPass()";
  let dependentDialects = ["imex::xetile::XeTileDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::index::IndexDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::vector::VectorDialect"];
}

def XeTileCanonicalization : Pass<"xetile-canonicalization", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Collection of canonicalization patterns for XeTile dialect.";

  let description = [{
    This pass performs a set of canonicalization steps on XeTile ops that are expected by the
    downstream passes. First, this will convert certain vector ops (transpose, broadcast,
     multi_reduction) to equivalent XeTile ops. Next, it will convert all XeTile ops
     consuming or producing col-major tiles to one with row-major tiles. Finally, it will
     perform cleanup to remove redundant ops that maybe produced by the previous steps.
  }];
  let constructor = "imex::createXeTileCanonicalizationPass()";
  let dependentDialects = [
    "::imex::xetile::XeTileDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

def XeTileBlocking : Pass<"xetile-blocking", "::mlir::gpu::GPUModuleOp">{
  let summary = "transform XeTile large tiles(input) into arrays of smaller "
                "blocks with appropriate size, such that the operator on each "
                "of the blocks can be mapped into one hardware instruction.";

  let description = [{
    This transform pass preprocesses the xetile program by decomposing large XeTile tiles
    into smaller ones that can be handled by a hardware instruction. It is going to replace
    the xetile-blocking pass.
  }];

  let constructor = "imex::createXeTileBlockingPass()";
  let dependentDialects = ["imex::xetile::XeTileDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::vector::VectorDialect"];

  let options = [
     Option<"device", "device", "std::string",
            /*default=*/"\"pvc\"",
            "gpu platform architecture where these ops are running">
 ];
}


#endif // _XeTile_PASSES_TD_INCLUDED_
