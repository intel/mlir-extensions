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
            RankedTensors with information about the location (device, distrbution) of
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
  let summary = "RankedTensor, optionally annoted with device and distribution info";
  let description = [{
    The ptensor type represents a tensor to applied to and created by operations in
    the PTensor dialect. Every PTensor is expected to be created by MkPTensorOp,
    either explicitly or implicitly by creation operations in the PTensor dialect.
  }];
  // Here we define the underlying RankedTensor
  let parameters = (ins "::mlir::RankedTensorType":$rtensor,
                        DefaultValuedParameter<"bool", "false">:$onDevice,
                        DefaultValuedParameter<"bool", "false">:$dist);

  let assemblyFormat = "`<` $rtensor (`,` struct($onDevice, $dist)^)? `>`";
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class PTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<PTensor_Dialect, mnemonic, traits>;


def MkPTensorOp : PTensor_Op<"init_ptensor", [NoSideEffect]> {
    let summary = "Create a new PTensor from RankedTensor, device, team and guid";
    let description = [{
        High-level operation to create a new PTensor.
        Every PTensor is expected to be created by this operation.
        FIXME The resulting PTensor will be marked as living on a device if $device is not none.
        FIXME The resulting PTensor will be marked as distributed if $team is not none.
    }];

    let arguments = (ins AnyRankedTensor:$rtensor,
                         AnyType:$device,
                         AnyType:$team,
                         AnyType:$handle);
    let results = (outs PTensor_PTensor);
    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<(ins "bool":$onDevice,
                     "bool":$dist,
                     "::mlir::Value":$rtensor,
                     "::mlir::Value":$device,
                     "::mlir::Value":$team,
                     "::mlir::Value":$handle), [{
        $_state.addOperands({rtensor, device, team, handle});
        $_state.addTypes(::imex::ptensor::PTensorType::get(
            $_state.getContext(),
            rtensor.getType().dyn_cast<::mlir::RankedTensorType>(),
            onDevice,
            dist));
      }]>,
      // Default: no device, non-distributed
      OpBuilder<(ins "::mlir::Value":$rtensor), [{
        auto null = createSignlessInt<1>($_builder, $_builder.getUnknownLoc(), 0);
        build($_builder, $_state, false, false, rtensor, null, null, null);
      }]>,
    ];
}

def ExtractRTensorOp : PTensor_Op<"extract_rtensor", [NoSideEffect]> {
    let summary = "Extract RankedTensor from a PTensor";
    let description = [{
        High-level operation to extract the RankedTensor from the given PTensor.
        Assumes that the given PTensor was created by MkPTensorOp.
    }];

    let arguments = (ins PTensor_PTensor:$input);
    let results = (outs AnyRankedTensor);
}

def ARangeOp : PTensor_Op<"arange", [NoSideEffect, SameVariadicOperandSize]> {
    let summary = "Create an arange (see array-API)";
    let description = [{
        Returns evenly `step`-spaced values within the half-open interval [`start`, `stop`) as a ptensor of rank 1.

        Optionally assigns it to device `device`, team `team` and uses pre-defined runtime-handle `rth`.
    }];

    let arguments = (ins AnyType : $start, AnyType : $stop, AnyType : $step,
                         Optional<AnyType>:$device, Optional<AnyType>:$team, Optional<AnyType>:$rth);
    // result is a ptensor
    let results = (outs PTensor_PTensor);
}

def EWBinOp : PTensor_Op<"ewbin", [NoSideEffect]> {
    let summary = "Apply elementwise binary operation";
    let description = [{
        Apply the `op(lhs[i], rhs[i])` on all elements `i` and return a new ptensor.
        The broadcasting rules of the array-API are applied to operator and result types.
    }];

    // arange takes 2 PTensorType operands: lhs and rhs
    let arguments = (ins AnyAttr: $op, PTensor_PTensor : $lhs, PTensor_PTensor : $rhs);
    // result is a ptensor
    let results = (outs PTensor_PTensor);
}

def ReductionOp : PTensor_Op<"reduction", [NoSideEffect]> {
    let summary = "Apply reduction operation";
    let description = [{
        Apply the reduction operation `op` over all elements of `input`.
        The produced result is a 0-dim tensor with the same dtype as `input`.
    }];

    // reduction takes 1 operand (PTensorType) and one attribute (reduction operation)
    let arguments = (ins AnyAttr: $op, PTensor_PTensor : $input);
    // result is a ptensor
    let results = (outs PTensor_PTensor);
}

#endif // _PTENSOR_OPS_TD_INCLUDED_
