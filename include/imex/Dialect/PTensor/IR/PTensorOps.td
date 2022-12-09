//===- PTensorOps.h - PTensor dialect  --------------------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic operations of the PTensor dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _PTENSOR_OPS_TD_INCLUDED_
#define _PTENSOR_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Provide a definition of the 'PTensor' dialect in the ODS framework so that we
// can define our operations.
def PTensor_Dialect : Dialect {
    // The namespace of our dialect
    let name = "ptensor";

    // A short one-line summary of our dialect.
    let summary = "A high-level dialect for parallel tensor operations";

    // A longer description of our dialect.
    let description = [{
            The ptensor dialect describes parallel operations on tensors.
            Generic parallel patterns are provided, such as element-wise-unary,
            element-wise-binary or reduce.

            Generally the PTensor dialect is intended to provide high-level abstractions
            to allow compute-follows-data semantics. For this the PTensorType annotates
            RankedTensors with information about the location (device) of
            the tensor-data when PTensors are created.

            Initially the functional scope of the dialect is the
            [array-API](https://data-apis.org/array-api/latest/index.html).
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::ptensor";

    //  We use the default parser/printer which handles registered types
    let useDefaultTypePrinterParser = true;
}

// common base classes for types in PTensor dialect
class PTensor_Type<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<PTensor_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def PTensor_PTensor : PTensor_Type<"PTensor", "ptensor">
{
  let summary = "RankedTensor, optionally annoted with device info";
  let description = [{
    The ptensor type represents a tensor to applied to and created by operations in
    the PTensor dialect. Every PTensor is expected to be created by MkPTensorOp,
    either explicitly or implicitly by creation operations in the PTensor dialect.
  }];
  // Here we define the underlying Tensor
  let parameters = (ins "int64_t":$rank,
                        "::mlir::Type":$element_type,
                        DefaultValuedParameter<"bool", "false">:$onDevice);
  let assemblyFormat = "`<` $rank `x` $element_type (`,` $onDevice^)? `>`";
  let extraClassDeclaration = [{
    ::mlir::MemRefType getMemRefType();
    ::mlir::RankedTensorType getTensorType();
  }];
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class PTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<PTensor_Dialect, mnemonic, traits>;


def MkPTensorOp : PTensor_Op<"init_ptensor", []> {
    let summary = "Create a new PTensor from tensor, device info";
    let description = [{
        High-level operation to create a new PTensor.
        Every PTensor is expected to be created by this operation.
        FIXME The resulting PTensor will be marked as living on a device if $device is not none.
    }];

    let arguments = (ins AnyMemRef:$tensor,
                         AnyType:$device);
    let results = (outs PTensor_PTensor:$res);
    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<(ins "bool":$onDevice,
                     "::mlir::Value":$mem_ref,
                     "::mlir::Value":$device), [{
        $_state.addOperands({mem_ref, device});
        auto mr = mem_ref.getType().dyn_cast<::mlir::MemRefType>();
        $_state.addTypes(::imex::ptensor::PTensorType::get(
            $_state.getContext(),
            mr.getShape().size(),
            mr.getElementType(),
            onDevice));
      }]>,
      // Default: no device
      OpBuilder<(ins "::mlir::Value":$mem_ref), [{
        auto dmy = createInt<1>($_builder.getUnknownLoc(), $_builder, 0);
        $_state.addOperands({mem_ref, dmy});
        auto mr = mem_ref.getType().dyn_cast<::mlir::MemRefType>();
        $_state.addTypes(::imex::ptensor::PTensorType::get(
            $_state.getContext(),
            mr.getShape().size(),
            mr.getElementType(),
            false)
        );
      }]>,
    ];
}

def ExtractMemRefOp : PTensor_Op<"extract_tensor", []> {
    let summary = "Extract MemRef from a PTensor";
    let description = [{
        High-level operation to extract the MemRef from the given PTensor.
        Assumes that the given PTensor was created by MkPTensorOp.
    }];

    let arguments = (ins AnyType:$input);
    let results = (outs AnyMemRef);
}

def ExtractSliceOp : PTensor_Op<"extract_slice", [SameVariadicOperandSize, Pure]> {
  let summary = "extract slice operation";
  let description = [{
    The "extract_slice" operation extract a tensor from another tensor as
    specified by the operation's offsets, sizes and strides arguments.

    The extract_slice operation is a shallow wrapper around tensor.extract_slice.
  }];

  let arguments = (ins
    AnyType:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );
  let results = (outs PTensor_PTensor:$result);

  let assemblyFormat = [{
    $source `[` $offsets `]``[` $sizes `]``[` $strides `]` attr-dict `:` qualified(type($source)) `to` qualified(type($result))
  }];
}

def InsertSliceOp : PTensor_Op<"insert_slice", [SameVariadicOperandSize]> {
  let summary = "Copy values from a tensor into a slice of another.";
  let description = [{
    Copy values from a tensor into a slice of another.
  }];

  let arguments = (ins
    AnyType:$destination,
    AnyType:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );

  let assemblyFormat = [{
    $source `into` $destination `[` $offsets `]``[` $sizes `]``[` $strides `]` attr-dict `:` qualified(type($destination)) `into` qualified(type($source))
  }];
}

def ARangeOp : PTensor_Op<"arange", [SameVariadicOperandSize]> {
    let summary = "Create an arange (see array-API)";
    let description = [{
        Returns evenly `step`-spaced values within the half-open interval [`start`, `stop`) as a ptensor of rank 1.

        Optionally assigns it to device `device`, and team `team`.
    }];

    let arguments = (ins AnyType:$start, AnyType:$stop, AnyType:$step,
                         Optional<AnyType>:$device, Optional<AnyType>:$team);
    // result is a ptensor
    let results = (outs PTensor_PTensor);
}

def CreateOp : PTensor_Op<"create", [AttrSizedOperandSegments]> {
    let summary = "Returns a new PTensor having a specified shape and type and optionally filled with a value.";
    let description = [{
        Returns a new PTensor having a specified shape and type and optionally filled with a value.

        Optionally assigns it to device `device`, and team `team`.
    }];

    let arguments = (ins Variadic<Index>:$shape, I8Attr:$dtype, Optional<AnyType>:$value,
                         Optional<AnyType>:$device, Optional<AnyType>:$team);
    // result is a ptensor
    let results = (outs PTensor_PTensor);

    let assemblyFormat = [{
      $shape oilist(`value` $value | `device` $device | `team` $team) attr-dict `:` `(` type(operands) `)` `->` qualified(type(results))
    }];

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::ValueRange":$shape, "::imex::ptensor::DType":$dtype, "::mlir::Value":$value, "::mlir::Value":$device, "::mlir::Value":$team), [{
            auto dt = toMLIR($_builder, dtype);
            build($_builder, $_state, ::imex::ptensor::PTensorType::get(
                $_state.getContext(), shape.size(), dt, device ? true : false),
                shape, $_builder.getI8IntegerAttr(dtype), value, device, team);
        }]>,
    ];

    let extraClassDeclaration = [{
      ::imex::ptensor::DType getDType() {
        return static_cast<::imex::ptensor::DType>(getDtype());
      }
    }];
}

def EWBinOp : PTensor_Op<"ewbin", []> {
    let summary = "Apply elementwise binary operation";
    let description = [{
        Apply the `op(lhs[i], rhs[i])` on all elements `i` and return a new ptensor.
        The broadcasting rules of the array-API are applied to operator and result types.
    }];

    // arange takes 2 PTensorType operands: lhs and rhs
    let arguments = (ins AnyAttr:$op, AnyType:$lhs, AnyType:$rhs);
    // result is a ptensor
    let results = (outs PTensor_PTensor);
}

def ReductionOp : PTensor_Op<"reduction", []> {
    let summary = "Apply reduction operation";
    let description = [{
        Apply the reduction operation `op` over all elements of `input`.
        The produced result is a 0-dim tensor with the same dtype as `input`.
    }];

    // reduction takes 1 operand (PTensorType) and one attribute (reduction operation)
    let arguments = (ins AnyAttr:$op, AnyType:$input);
    // result is a ptensor
    let results = (outs PTensor_PTensor);
}

#endif // _PTENSOR_OPS_TD_INCLUDED_
