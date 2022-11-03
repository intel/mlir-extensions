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

    let dependentDialects = [
        "::imex::ptensor::PTensorDialect"
    ];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::dist";
    let useDefaultTypePrinterParser = 1;
}

// common base classes for types in Dist dialect
class Dist_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Dist_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Dist_Tensor : Dist_Type<"DistTensor", "dtensor">
{
  let summary = "A type used to bind distributed information to a PTensor";
  let description = [{
    A distributed PTensor needs information like offset and shape of local partition.
    The DistTensor type is used to define operations to carry and extract such information.
  }];
  let parameters = (ins "::imex::ptensor::PTensorType":$p_tensor_type);
  let assemblyFormat = "`<` $p_tensor_type `>`";
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dist_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dist_Dialect, mnemonic, traits>;

def RuntimePrototypesOp : Dist_Op<"runtime_prototypes"> {
    let summary = "Add function prototypes used for calling into distributed runtime";
}

def NProcsOp : Dist_Op<"nprocs", [Pure]> {
    let summary = "Number of processes for given team";
    let arguments = (ins AnyType:$team);
    let results = (outs I64);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$team), [{
            build($_builder, $_state, $_builder.getI64Type(), team);
        }]>,
    ];
}

def PRankOp : Dist_Op<"prank", [Pure]> {
    let summary = "Process rank in team";
    let arguments = (ins AnyType:$team);
    let results = (outs I64);
    let builders = [
        OpBuilder<(ins "::mlir::Value":$team), [{
            build($_builder, $_state, $_builder.getI64Type(), team);
        }]>,
    ];
}

def InitDistTensorOp : Dist_Op<"init_dist_tensor", [Pure]> {
    let summary = "Bind a PTensor to distributed meta information";
    let description = [{
        The attached PTensor is the local partiton of the distributed PTensor.
        The distributed meta information about a new PTensor provides
          - the global shape
          - the process-local offsets
          - the distributed team
    }];
    let arguments = (ins AnyType:$g_shape, AnyType:$p_tensor, AnyType:$l_offsets, AnyType:$team);
    let results = (outs Dist_Tensor);
    let builders = [
      OpBuilder<(ins "::mlir::Value":$g_shape, "::mlir::Value":$p_tensor, "::mlir::Value":$l_offsets, "::mlir::Value":$team), [{
            build($_builder, $_state,
                  ::imex::dist::DistTensorType::get($_builder.getContext(), p_tensor.getType().dyn_cast<::imex::ptensor::PTensorType>()),
                  g_shape, p_tensor, l_offsets, team);
      }]>,
    ];
}

def ExtractFromDistOp : Dist_Op<"extract_from_dist", []> {
    let summary = "Get one element from the distributed meta information.";
    let arguments = (ins I32Attr:$what, AnyType:$d_tensor);
    let results = (outs AnyType);
    let builders = [
      OpBuilder<(ins "::imex::dist::INFO":$what, "::mlir::Value":$d_tensor), [{
        // auto-deduce return type from from operands
        auto i64Type = $_builder.getI64Type();
        auto idx = getIntAttr<32>($_builder, what);
        $_state.addAttribute(getWhatAttrName(odsState.name), idx);
        $_state.addOperands(d_tensor);
        int64_t rank = 0;
        auto ttype = d_tensor.getType().dyn_cast<::imex::dist::DistTensorType>();
        assert(ttype);
        rank = ttype.getPTensorType().getRtensor().getRank();
        switch(idx.getInt()) {
            case GSHAPE:
            case LOFFSETS:
                $_state.addTypes(::mlir::RankedTensorType::get({rank}, i64Type));
                break;
            case LTENSOR:
                $_state.addTypes(ttype.getPTensorType());
                break;
            case TEAM:
                $_state.addTypes(i64Type);
                break;
            default:
                assert(!"Unknown distributed meta information requested");
        };
      }]>,
    ];
}

def LocalOffsetsOp : Dist_Op<"local_offsets", []> {
    let summary = "Compute the offsets (one for each dimension) of the local partition in number of elements.";
    let description = [{
        Result is a 1d memref.
    }];
    let arguments = (ins I64Attr:$rank, I64:$num_procs, I64:$prank, AnyType:$gshape);
    let results = (outs AnyType);
    let builders = [
      OpBuilder<(ins "int64_t":$rank, "::mlir::Value":$num_procs, "::mlir::Value":$prank, "::mlir::Value":$gshape), [{
            build($_builder, $_state, ::mlir::RankedTensorType::get({rank}, $_builder.getI64Type()), getIntAttr<64>($_builder, rank), num_procs, prank, gshape);
      }]>,
    ];
}

def LocalShapeOp : Dist_Op<"local_shape", []> {
    let summary = "Compute the shape (one for each dimension) of the local partition in number of elements.";
    let description = [{
        Result is a 1d memref.
    }];
    let arguments = (ins I64Attr:$rank, I64:$num_procs, I64:$prank, AnyType:$gshape);
    let results = (outs AnyType);
    let builders = [
      OpBuilder<(ins "int32_t":$rank, "::mlir::Value":$num_procs, "::mlir::Value":$prank, "::mlir::Value":$gshape), [{
            build($_builder, $_state, ::mlir::RankedTensorType::get({rank}, $_builder.getI64Type()), getIntAttr<64>($_builder, rank), num_procs, prank, gshape);
      }]>,
    ];
}

def AllReduceOp : Dist_Op<"allreduce", []> {
    let summary = "Inplace allreduce";
    let description = [{
        Result is the allreduced input tensor.
    }];
    // reduction operation and local tensor
    let arguments = (ins AnyAttr:$op, AnyTensor:$tensor);
    let results = (outs AnyType);
}

#endif // _Dist_OPS_TD_INCLUDED_
