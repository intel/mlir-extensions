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
            GPUX Dialect provides operations for creating/destroying and
            launching operations on a stream. GPUX dialect extends the alloc,
            dealloc, memcpy, memset, launch_func and wait operations from
            upstream GPU dialect and adds an additional argument for stream
            in those ops.
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::gpux";

    //  We use the default parser/printer which handles registered types
    let useDefaultTypePrinterParser = true;
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class GPUX_Op<string mnemonic, list<Trait> traits = []> :
    Op<GPUX_Dialect, mnemonic, traits>;

// common base class for types in GPUX dialect.
class GPUX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<GPUX_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def GPUX_OpaqueType : GPUX_Type<"Opaque", "OpaqueType">
{
  let summary = "Custom Type for OpaqueType";
  let description = [{
    "GPUX_OpaqueType is a Custom Type for Opaque Pointer"
  }];
}

def GPUX_StreamType : GPUX_Type<"Stream", "StreamType">
{
  let summary = "Custom Type for GPUXStream";
  let description = [{
    "GPUX_StreamType is a Custom Type for Stream."
  }];
}

def GPUX_DeviceType : GPUX_Type<"Device", "DeviceType">
{
  let summary = "Custom Type for GPUXDevice";
  let description = [{
    "GPUX_DeviceType is a Custom Type for Device."
  }];
}

def GPUX_ContextType : GPUX_Type<"Context", "ContextType">
{
  let summary = "Custom Type for GPUXContext";
  let description = [{
    "GPUX_ContextType is a Custom Type for Context."
  }];
}

def GPUX_CreateDeviceOp : GPUX_Op<"create_device"> {

  // Operation to create and return a device.
  let arguments = (ins GPUX_DeviceType : $device);
  let results = (outs GPUX_DeviceType : $gpu_device);
}

def GPUX_CreateContextOp : GPUX_Op<"create_context"> {

  // Operation to create and return a context.
  let arguments = (ins GPUX_DeviceType : $device);
  let results = (outs GPUX_ContextType : $gpu_context);
}

def GPUX_CreateStreamOp : GPUX_Op<"create_stream", [SameVariadicOperandSize]> {

  // Operation to create and return a stream, based on the context
  // and device. This stream is used for launching/queuing kernels
  // on the GPU. If no device and context are provided, a default
  // device and context will be created.
  let arguments = (ins Optional<GPUX_DeviceType> : $device,
                       Optional<GPUX_ContextType> : $context);
  let results = (outs GPUX_StreamType : $gpux_stream);
  let builders = [OpBuilder<(ins "::llvm::Optional<::mlir::Value>" : $device,
                                 "::llvm::Optional<::mlir::Value>" : $context)>];
}

def GPUX_DestroyDeviceOp : GPUX_Op<"destroy_device"> {

  // Operation to deallocate the passed in device pointer.
  let arguments = (ins GPUX_DeviceType : $device);
}

def GPUX_DestroyContextOp : GPUX_Op<"destroy_context"> {

  // Operation to deallocate the passed in context pointer.
  let arguments = (ins GPUX_ContextType : $context);
}

def GPUX_DestroyStreamOp : GPUX_Op<"destroy_stream"> {

  // Operation to deallocate the passed in stream pointer.
  let arguments = (ins GPUX_StreamType : $gpux_stream);
}

// The following ops are same as defined in upstream GPU dialect
// https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/mlir/include/mlir/Dialect/GPU/GPUOps.td
// with an additional argument for stream.

def GPUX_LaunchFuncOp
    : GPUX_Op<"launch_func",
               [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {

  // Operation for launching a kernel on gpu.

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                     GPUX_StreamType:$gpux_stream,
                     SymbolRefAttr:$kernel,
                     Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
                     Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
                     Optional<I32>:$dynamicSharedMemorySize,
                     Variadic<AnyType>:$kernelOperands);
  let results = (outs Optional<GPU_AsyncToken> : $asyncToken);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value" : $gpux_stream,
                   "::mlir::gpu::GPUFuncOp":$kernelFunc,
                   "::mlir::gpu::KernelDim3":$gridSize,
                   "::mlir::gpu::KernelDim3":$blockSize,
                   "::mlir::Value":$dynamicSharedMemorySize,
                   "::mlir::ValueRange":$kernelOperands,
                   CArg<"::mlir::Type", "nullptr">:$asyncTokenType,
                   CArg<"::mlir::ValueRange", "{}">:$asyncDependencies)>
  ];

  let extraClassDeclaration = [{
    /// The name of the kernel's containing module.
    mlir::StringAttr getKernelModuleName();

    /// The name of the kernel.
    mlir::StringAttr getKernelName(); }];

}

def GPUX_AllocOp
    : GPUX_Op<"alloc", [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {

  // Operation for allocating memory on gpu.

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPUX_StreamType:$gpux_stream,
                   Variadic<Index>:$dynamicSizes,
                   Variadic<Index>:$symbolOperands,
                   UnitAttr:$hostShared);
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref,
                 Optional<GPU_AsyncToken>:$asyncToken);
}

def GPUX_DeallocOp : GPUX_Op<"dealloc", [GPU_AsyncOpInterface]> {

  // Operation for freeing allocated memory on gpu.

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPUX_StreamType:$gpux_stream,
                   Arg<AnyMemRef, "", [MemFree]>:$memref);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
}

def GPUX_WaitOp : GPUX_Op<"wait", [GPU_AsyncOpInterface]> {

   // Operation for waiting on a stream.

   let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                    GPUX_StreamType:$gpux_stream);
   let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
}

def GPUX_MemcpyOp : GPUX_Op<"memcpy", [GPU_AsyncOpInterface]> {

  // Operation for copying the content of one memref to another.
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPUX_StreamType:$gpux_stream,
                   Arg<AnyMemRef, "", [MemWrite]>:$dst,
                   Arg<AnyMemRef, "", [MemRead]>:$src);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
}

def GPUX_MemsetOp : GPUX_Op<"memset", [GPU_AsyncOpInterface]> {

 // Operation for setting the content of memref to a scalar value.

 let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPUX_StreamType:$gpux_stream,
                   Arg<AnyMemRef, "", [MemWrite]>:$dst,
                   Arg<AnyType, "">:$value);
 let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
}

#endif // _GPUX_OPS_TD_INCLUDED_
