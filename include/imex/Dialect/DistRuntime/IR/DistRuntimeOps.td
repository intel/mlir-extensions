//===- DistRuntimeOps.td - DistRuntime dialect  -------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the basic operations for the DistRuntime dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _DistRuntime_OPS_TD_INCLUDED_
#define _DistRuntime_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'DistRuntime' dialect in the ODS framework so that we
// can define our operations.
def DistRuntime_Dialect : Dialect {
    // The namespace of our dialect
  let name = "distruntime";

  // A short one-line summary of our dialect.
  let summary = "This dialect provides distributed runtime calls.";

  // A longer description of our dialect.
  let description = [{
    The DistRuntime dialect supports communication in SPMD environments. The
    general assumption is that teams (e.g., groups) of processes/threads execute the
    same set of operations but operate on different data. Most operations in this
    dialect hence require the team argument. It is up to the lowering/runtime to
    interpret the value of a given team argument.

    Two kinds of operations are provided:

    1. A set of basic operations providing information about the state of the
      runtime, like the id of the caller within a team or the number of members
      within a team. These operations do not require the participation of other
      team members.
    2. A set of high-level primitives that typically require communication and,
      therefore, involve - implicitly or explicitly - other team members.

    Some operations are defined to enable asynchronous communication which allows
    for overlapping communication and computation. Such operations return a value of
    the opaque type `AsyncHandle`. It is up to the runtime and related lowering
    passes to give life to the type. The results of such an asynchronous
    operation, whether they are written in-place or returned as values, must be
    preceded with a ca
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::imex::distruntime";

  //  We use the default parser/printer which handles registered types
  let useDefaultTypePrinterParser = true;
}

// common base class for types in DistRuntime dialect.
class DistRuntime_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<DistRuntime_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DistRuntime_AsyncHandle : DistRuntime_Type<"AsyncHandle", "asynchandle">
{
  let summary = "Custom Type for representing a handle for asynchronous operation.";
  let description = [{
    Custom Type for representing a handle for asynchronous operation/communication.

    Asynchronous runtime operations return handles. Uses of other outputs
    of such operations require a prior call to wait.
  }];
}

def AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface definition for asynchronous operations of the DistRuntime dialect.
  }];

  let methods = [
    InterfaceMethod<
      "Return all results of the asynchronous operation and arguments it mutates",
      "::mlir::SmallVector<::mlir::Value>", "getDependent"
    >,
  ];
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class DistRuntime_Op<string mnemonic, list<Trait> traits = []> :
    Op<DistRuntime_Dialect, mnemonic, traits>;

def CopyReshapeOp : DistRuntime_Op<"copy_reshape",
    [Pure, DeclareOpInterfaceMethods<AsyncOpInterface>, AttrSizedOperandSegments]> {
  let summary = "Copy adequate data from input to a new reshaped output";
  let description = [{
    Copy the necessary data from the distributed input array to the locally owned part of the output array.
    The shape of the output array is assumed to be a reshaped version of the input's shape.

    The local data is not modified.

    Arguments:

    - `team`: the distributed team owning the distributed array
    - `gShape`: the global shape of the distributed array
    - `lOffsets`: the offset of the local data within the global array
    - `lArray`: the locally owned data
    - `ngShape`: the global shape of the distributed output array
    - `nlOffsets`: the offsets of the locally owned output array

    `gShape`, `lOffsets` are variadic arguments with same size `ri` where `ri` is the rank of the global input array (e.g., one number for each dimension of the global input array).
    `ngShape`, `nlOffsets` are variadic arguments with same size `ro` where `ro` is the rank of the global output array (e.g., one number for each dimension of the global output array).
  }];
  let arguments = (ins AnyAttr:$team,
                       AnyType:$lArray, Variadic<Index>:$gShape, Variadic<Index>:$lOffsets,
                       Variadic<Index>:$ngShape, Variadic<Index>:$nlOffsets, Variadic<Index>:$nlShape);
  let results = (outs DistRuntime_AsyncHandle:$handle, AnyType:$nlArray);
  let assemblyFormat = [{
    $lArray `g_shape` $gShape `l_offs` $lOffsets `to` `n_g_shape` $ngShape `n_offs` $nlOffsets `n_shape` $nlShape attr-dict `:` `(` type(operands) `)` `->` `(` qualified(type(results)) `)`
  }];
  let hasCanonicalizer = 1;
}

def WaitOp : DistRuntime_Op<"wait", []> {
  let summary = "Wait for asynchronous operation to finish.";
  let description = [{
    Wait for asynchronous operation to finish.
    Accepts an `AsyncHandle`.
  }];
  let arguments = (ins DistRuntime_AsyncHandle:$handle);
}

def CopyPermuteOp : DistRuntime_Op<"copy_permute",
    [Pure, DeclareOpInterfaceMethods<AsyncOpInterface>, AttrSizedOperandSegments]> {
  let summary = "Copy adequate data from input to a new permuted output";
  let description = [{
    Copy the necessary data from the distributed input array to the locally owned part of the new output array.
    The shape of the output array is assumed to be a permuted version of the input's shape.

    The local data is not modified.

    Arguments:

    - `team`: the distributed team owning the distributed array
    - `gShape`: the global shape of the distributed array
    - `lOffsets`: the offset of the local data within the global array
    - `lArray`: the locally owned data
    - `nlShape`: the local shape of the distributed output array
    - `nlOffsets`: the offsets of the locally owned output array

    `gShape`, `lOffsets` are variadic arguments with same size `ri` where `ri` is the rank of the global input array (e.g., one number for each dimension of the global input array).
    `ngShape`, `nlOffsets` are variadic arguments with same size `ro` where `ro` is the rank of the global output array (e.g., one number for each dimension of the global output array).
  }];
  let arguments = (ins AnyAttr:$team,
                       AnyType:$lArray,
                       Variadic<Index>:$gShape,
                       Variadic<Index>:$lOffsets,
                       Variadic<Index>:$nlOffsets,
                       Variadic<Index>:$nlShape,
                       DenseI64ArrayAttr:$axes);
  let results = (outs DistRuntime_AsyncHandle:$handle, AnyType:$nlArray);
  let assemblyFormat = [{
    $lArray `g_shape` $gShape `l_offs` $lOffsets `to` `n_offs` $nlOffsets `n_shape` $nlShape `axes` $axes attr-dict `:` `(` type(operands) `)` `->` `(` qualified(type(results)) `)`
  }];
  let hasCanonicalizer = 1;
}

#endif // _DistRuntime_OPS_TD_INCLUDED_
