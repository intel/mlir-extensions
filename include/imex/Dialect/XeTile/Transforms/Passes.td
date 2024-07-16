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

def XeTileBlocking : Pass<"xetile-blocking", "::mlir::gpu::GPUModuleOp">{
  let summary = "transform XeTile large tiles(input) into arrays of smaller "
                "blocks with appropriate size, such that the operator on each "
                "of the blocks can be mapped into one hardware instruction.";

  let description = [{
    This transform pass preprocesses the xetile program by decomposing large XeTile tiles
    into smaller ones that can be handled by a hardware instruction. This blocked layout
    is represented by high dimension vectors, inner dimension matches to DPAS size config.
    This lowers 2D vector to 4D vector.
  }];

  let constructor = "imex::createXeTileBlockingPass()";
  let dependentDialects = ["imex::xetile::XeTileDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::vector::VectorDialect"];

  let options = [
     Option<"device", "device", "std::string",
            /*default=*/"\"pvc\"",
            "gpu platform architecture where these ops are running">
 ];
}


// TODO: [block-aligning] remove the following code when upstreaming the pass.
// The pass is not supposed to be exposed to users. Temporary keep in case we
// need debug it for down stream development.
def XeTileBlockAligning: Pass <"xetile-block-aligning", "::mlir::gpu::GPUModuleOp"> {
  let summary = "optimize the performance for mma.";

  let description = [{
    This transform is to optimize performance by aligning the block size among operators
    to reduce in-register data movements. Currently, it mainly focues on the alignment
    between load and MMA operators.
  }];

  let constructor = "imex::createXeTileBlockAligningPass()";
  let dependentDialects = ["imex::xetile::XeTileDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::vector::VectorDialect"];
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

def XeTileOptimizeTranspose : Pass<"xetile-optimize-transpose"> {
  let summary = "Fuse tile loads and transpose operations.";

  let description = [{
    This pass fuses tile loads that are consumed by transpose operations into a single
    load operation. Transformation uses the `order` attribute in the XeTile type to
    perform the load and transpose together.
  }];
  let constructor = "imex::createXeTileOptimizeTransposePass()";
  let dependentDialects = [
    "::imex::xetile::XeTileDialect"
  ];
}


#endif // _XeTile_PASSES_TD_INCLUDED_
