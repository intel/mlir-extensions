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
    let results = (outs Index);
    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$team), [{
            build($_builder, $_state, $_builder.getIndexType(), team);
        }]>,
    ];
}

def PRankOp : Dist_Op<"prank", [Pure]> {
    let summary = "Process rank in team";
    let arguments = (ins AnyType:$team);
    let results = (outs Index);
    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$team), [{
            build($_builder, $_state, $_builder.getIndexType(), team);
        }]>,
    ];
}

def InitDistTensorOp : Dist_Op<"init_dist_tensor", [SameVariadicOperandSize, Pure]> {
    let summary = "Bind a PTensor to distributed meta information";
    let description = [{
        The attached PTensor is the local partiton of the distributed PTensor.
        The distributed meta information about a new PTensor provides
          - the global shape
          - the process-local offsets
          - the distributed team
    }];
    let arguments = (ins Variadic<Index>:$g_shape, AnyType:$p_tensor, Variadic<Index>:$l_offsets, AnyType:$team);
    let results = (outs Dist_Tensor);
    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::ValueRange":$g_shape, "::mlir::Value":$p_tensor, "::mlir::ValueRange":$l_offsets, "::mlir::Value":$team), [{
            build($_builder, $_state,
                  ::imex::dist::DistTensorType::get($_builder.getContext(), p_tensor.getType().dyn_cast<::imex::ptensor::PTensorType>()),
                  g_shape, p_tensor, l_offsets, team);
        }]>,
    ];
}

def GlobalShapeOfOp : Dist_Op<"global_shape_of", []> {
    let summary = "Get global shape of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs Variadic<Index>:$g_shape);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        build($_builder, $_state, $_builder.getIndexType(), d_tensor);
      }]>,
    ];
}

def LocalOffsetsOfOp : Dist_Op<"local_offsets_of", []> {
    let summary = "Get local offsets of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs Variadic<Index>:$l_offsets);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        build($_builder, $_state, $_builder.getIndexType(), d_tensor);
      }]>,
    ];
}

def LocalTensorOfOp : Dist_Op<"local_tensor_of", []> {
    let summary = "Get local tensor of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs AnyType:$l_tensor);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        auto ttype = d_tensor.getType().dyn_cast<::imex::dist::DistTensorType>();
        build($_builder, $_state, ttype.getPTensorType(), d_tensor);
      }]>,
    ];
}

def TeamOfOp : Dist_Op<"team_of", []> {
    let summary = "Get team of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs AnyType:$team);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        build($_builder, $_state, $_builder.getIndexType(), d_tensor);
      }]>,
    ];
}

def LocalPartitionOp : Dist_Op<"local_partition", [SameVariadicResultSize, Pure]> {
    let summary = "Compute the shape and offsets of the local partition in number of elements (one for each dimension).";
    let arguments = (ins I64Attr:$rank, Index:$num_procs, Index:$p_rank, Variadic<Index>:$g_shape);
    let results = (outs Variadic<Index>:$l_offsets, Variadic<Index>:$l_shape);
    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "int64_t":$rank, "::mlir::Value":$num_procs, "::mlir::Value":$prank, "::mlir::ValueRange":$gshape), [{
            auto IndexType = $_builder.getIndexType();
            build($_builder,
                  $_state,
                  ::mlir::TypeRange{IndexType, IndexType},
                  $_builder.getI64IntegerAttr(rank),
                  num_procs,
                  prank,
                  gshape);
        }]>,
    ];
}

def LocalOfSliceOp : Dist_Op<"local_of_slice",
    [SameVariadicOperandSize, SameVariadicResultSize, Pure]> {
    let summary = "Compute local overlap of a distributed tensor and slice";
    let description = [{
        Slice and tensor operate on the global index space. This operation computes the
        local part of the slice as owned by the local partition of the tensor. The operation
        returns local offsets and sizes (e.g. relative to the local memref). Additionally,
        it computes and returns the offsets of the resulting local slice relative to the global input slice.
    }];

    let arguments = (ins
        AnyType:$d_tensor,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,
        Variadic<Index>:$strides
    );
    let results = (outs Variadic<Index>:$l_offsets, Variadic<Index>:$l_sizes, Variadic<Index>:$g_offsets);

    let assemblyFormat = [{
        $d_tensor `[` $offsets `]``[` $sizes `]``[` $strides `]` attr-dict `:` qualified(type($d_tensor)) `to` `(`qualified(type(results))`)`
    }];

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$d_tensor, "::mlir::ValueRange":$offsets, "::mlir::ValueRange":$sizes, "::mlir::ValueRange":$strides), [{
            auto IndexType = $_builder.getIndexType();
            build($_builder, $_state, ::mlir::TypeRange{IndexType, IndexType, IndexType}, d_tensor, offsets, sizes, strides);
        }]>,
    ];
}

def LocalToGlobalOp : Dist_Op<"local_to_global", [Pure]> {
  let summary = "Translate local inidices into global indices";
  let description = [{
    Input inidices are interprete as relative to the local part of the given DTensor.
  }];

  let arguments = (ins AnyType:$d_tensor, Variadic<Index>:$l_indices);
  let results = (outs Variadic<Index>:$g_indices);

//   let assemblyFormat = [{
//     $d_tensor attr-dict `:` qualified(type($source)) `to` `(`qualified(type(results))`)`
//   }];
}

def AllReduceOp : Dist_Op<"allreduce", []> {
    let summary = "Inplace allreduce";
    let description = [{
        Result is the allreduced input tensor.
    }];
    // reduction operation and local tensor
    let arguments = (ins AnyAttr:$op, AnyMemRef:$data);
    let results = (outs AnyType);
}

#endif // _Dist_OPS_TD_INCLUDED_
