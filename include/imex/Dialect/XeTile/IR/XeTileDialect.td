//===--------------- XeTileOps.td - XeTile dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the XeTile dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XETILE_BASE_TD_INCLUDED_
#define _XETILE_BASE_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"

// Provide a definition of the 'XeTile' dialect in the ODS framework so that we
// can define our operations.
def XeTile_Dialect : Dialect {
    // The namespace of our dialect
    let name = "xetile";

    // A short one-line summary
    let summary = "A dialect for enabling tile-base programming at subgroup level";

    // A longer description
    let description = [{
        XeTile provides an abstraction supporting tile-based computation to simplify the
        lowering of DNN operation like matrix multiplication. XeTile dialect works at tile sizes
        that are larger than the tile sizes supported by the hardware. XeTile dilaect also hides
        the auto-padding requirements for out-of-bound memory accesses and, supports arbitrary
        input matrix sizes.
    }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::xetile";

    let dependentDialects = [
          "::mlir::memref::MemRefDialect",
          "::mlir::vector::VectorDialect"];

    // TODO: temporary disable it.
    let useDefaultTypePrinterParser = true;
    let useDefaultAttributePrinterParser = true;
}


#endif // _XETILE_BASE_TD_INCLUDED_
