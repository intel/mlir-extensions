//===- GPUXOps.td - GPUX dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic operations of the GPUX dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _GPUX_OPS_TD_INCLUDED_
#define _GPUX_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"


// Provide a definition of the 'GPUX' dialect in the ODS framework so that we
// can define our operations.
def GPUX_Dialect : Dialect {
    // The namespace of our dialect
    let name = "gpux";

    // A short one-line summary of our dialect.
    let summary = "GPUX Dialect for extending the upstream GPU dialect ops";

    // A longer description of our dialect.
    let description = [{
            GPUX Dialect provides operations for creating/destroying and launching operations
            on a stream.
            GPUX dialect extends the alloc, dealloc, launch_func and wait operations from
            upstream GPU dialect and adds an additional argument for stream in those ops.
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::gpux";
    let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class GPUX_Op<string mnemonic, list<Trait> traits = []> :
    Op<GPUX_Dialect, mnemonic, traits>;

// Provide a definition for the GPUX StreamType for use in ODS. This allows for
// using StreamType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StreamType as belonging to the GPUX dialect.
def GPUX_StreamType
    : DialectType<GPUX_Dialect, CPred<"$_self.isa<::imex::gpux::StreamType>()">,
                  "Stream Type">;

def GPUX_CreateStreamOp : GPUX_Op<"create_stream"> {

  // Operation for creating a temporary stream on which gpu kernels
  // will be launched
  let results = (outs GPUX_StreamType : $gpux_stream);
}

def GPUX_DestroyStreamOp : GPUX_Op<"destroy_stream"> {

  // Operation for destroying the above created stream
  let arguments = (ins GPUX_StreamType : $gpux_stream);
}

// The following ops are same as defined in upstream GPU dialect
// https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/mlir/include/mlir/Dialect/GPU/GPUOps.td
// with an additional argument for stream.

def GPUX_LaunchFuncOp
    : GPUX_Op<"launch_func",
               [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {

  // Operation for launching a kernel on gpu. Operation is same as upstream
  // GPU dialect launch_func op but with an additional stream argument

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                     GPUX_StreamType:$gpux_stream,
                     SymbolRefAttr:$kernel,
                     Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
                     Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
                     Optional<I32>:$dynamicSharedMemorySize,
                     Variadic<AnyType>:$operands);
  let results = (outs Optional<GPU_AsyncToken> : $asyncToken);
}

def GPUX_AllocOp
    : GPUX_Op<"alloc", [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {

  // Operation for allocating memory on gpu. Operation is same as upstream
  // GPU dialect alloc op but with an additional stream argument

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPUX_StreamType:$gpux_stream,
                   Variadic<Index>:$dynamicSizes, Variadic<Index>:$symbolOperands);
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref,
                 Optional<GPU_AsyncToken>:$asyncToken);
}

def GPUX_DeallocOp : GPUX_Op<"dealloc", [GPU_AsyncOpInterface]> {

  // Operation for freeing allocated memory on gpu. Operation is same as upstream
  // GPU dialect dealloc op but with an additional stream argument

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPUX_StreamType:$gpux_stream,
                   Arg<AnyMemRef, "", [MemFree]>:$memref);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
}

def GPUX_WaitOp : GPUX_Op<"wait", [GPU_AsyncOpInterface]> {

   // Operation for waiting on a stream. Operation is same as upstream
   // GPU dialect wait op but with an additional stream argument

   let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                    GPUX_StreamType:$gpux_stream);
   let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
}

#endif // _GPUX_OPS_TD_INCLUDED_
