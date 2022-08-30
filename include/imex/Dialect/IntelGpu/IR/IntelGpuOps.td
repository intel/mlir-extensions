//===- IntelGpuOps.td - IntelGpu dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the basic operations for the IntelGpu dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _IntelGpu_OPS_TD_INCLUDED_
#define _IntelGpu_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// Provide a definition of the 'IntelGpu' dialect in the ODS framework so that we
// can define our operations.
def IntelGpu_Dialect : Dialect {
    // The namespace of our dialect
    let name = "intel_gpu";

    // A short one-line summary of our dialect.
    let summary = "IntelGpu Dialect for representing Intel GPU specific ops";

    // A longer description of our dialect.
    let description = [{
            FIXME insert discription
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::intel_gpu";
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class IntelGpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntelGpu_Dialect, mnemonic, traits>;

def IntelGpu_OpaqueType
    : DialectType<IntelGpu_Dialect, CPred<"$_self.isa<::imex::intel_gpu::OpaqueType>()">,
                  "opaque_type">,
      BuildableType<"$_builder.getType<::imex::intel_gpu::OpaqueType>()"> {}

def CreateStreamOp : IntelGpu_Op<"create_stream"> {
  let results = (outs IntelGpu_OpaqueType : $stream);
  let hasCanonicalizer = 1;
  let builders = [OpBuilder<(ins)>];
}

def DestroyStreamOp : IntelGpu_Op<"destroy_stream"> {
  let arguments = (ins IntelGpu_OpaqueType : $stream);
}

#endif // _IntelGpu_OPS_TD_INCLUDED_
