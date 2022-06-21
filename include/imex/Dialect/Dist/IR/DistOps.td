//===- DistOps.td - Dist dialect  --------------------------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic operations of the Dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Dist_OPS_TD_INCLUDED_
#define _Dist_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'Dist' dialect in the ODS framework so that we
// can define our operations.
def Dist_Dialect : Dialect {
    // The namespace of our dialect
    let name = "dist";

    // A short one-line summary of our dialect.
    let summary = "A high-level dialect for distributing PTensor operations";

    // A longer description of our dialect.
    let description = [{
            The dist dialect describes interfaces for interacting with
	    a runtime which handles distributed aspects of PTensor operations.
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::dist";
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dist_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dist_Dialect, mnemonic, traits>;

// Register a ptensor of given shape with a (potentially distributed) runtime.
// Returns an id to uniquely identify the tensor instance in future interactino with the runtime.
// The runtime does not own or manage any PTensor memory. When needed by an operation,
// (local) data needs to be provided.
def RegisterPTensorOp : Dist_Op<"register_ptensor", [NoSideEffect]> {
    // Global shape needed for initial registration. Views are handled by a separate op.
    let arguments = (ins AnyType: $shape);

    // result is an Integer Id
    let results = (outs I64);
}

// Get the offsets (one for each dimension) of the local partition of a distributed PTensor in number of elements.
// Partitionings can be N-dimensional but must cut only the first N dimensions.
def LocalOffsetsOp : Dist_Op<"local_offsets", [NoSideEffect]> {
    // Id of tensor as returned by RegisterPTensorOp
    let arguments = (ins I64: $ptensor);

    // result is a 1d memref
    let results = (outs AnyType);
}

// Get the shape (one size for each dimension) of the local partition of a distributed PTensor in number of elements.
// Partitionings can be N-dimensional but must cut only the first N dimensions.
def LocalShapeOp : Dist_Op<"local_shape", [NoSideEffect]> {
    // Id of tensor as returned by RegisterPTensorOp
    let arguments = (ins I64: $ptensor);

    // result is a 1d memref
    let results = (outs AnyType);
}

// Inplace allreduce
def AllReduceOp : Dist_Op<"allreduce", [NoSideEffect]> {
    // reduction operation and and local tensor
    let arguments = (ins AnyAttr: $op, AnyTensor: $tensor);

    // result is allreduced input tensor
    let results = (outs AnyTensor: $result);
}

#endif // _Dist_OPS_TD_INCLUDED_
