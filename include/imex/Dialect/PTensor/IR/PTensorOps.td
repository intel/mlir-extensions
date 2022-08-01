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
            The ptensor dialect describe parallel operations on tensors.
            Generic parallel patterns are provided, such as element-wise-unary,
            element-wise-binary or reduce.
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::ptensor";

    //  We use the default parser/printer which handles registered types
    let useDefaultTypePrinterParser = true;
}

class PTensor_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PTensor_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def PTensor_PTensor : PTensor_Type<"PTensor", "ptensor">
{
  let summary = "RankedTensor, optionally distributed";
  // Here we defined the underlying RankedTensor and whether the tensor is distributed
  let parameters = (ins "::mlir::RankedTensorType":$rtensor, "bool":$dist);

  let assemblyFormat = "`<` $rtensor`,` $dist `>`"; // FIXME
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class PTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<PTensor_Dialect, mnemonic, traits>;


def ARangeOp : PTensor_Op<"arange", [NoSideEffect]> {
    // arange takes 3 integer operands (for now): start, stop and step
    let arguments = (ins AnyType : $start, AnyType : $stop, AnyType : $step, BoolAttr : $dist);

    // result is a tensor
    let results = (outs PTensor_PTensor);
}

def EWBinOp : PTensor_Op<"ewbin", [NoSideEffect]> {
    // arange takes 2 PTensorType operands: lhs and rhs
    let arguments = (ins AnyAttr: $op, PTensor_PTensor : $lhs, PTensor_PTensor : $rhs);

    // result is a tensor
    let results = (outs PTensor_PTensor);
}

def ReductionOp : PTensor_Op<"reduction", [NoSideEffect]> {
    // reduction takes 1 operand (PTensorType) and one attribute (reduction operation)
    let arguments = (ins AnyAttr: $op, PTensor_PTensor : $input);

    // result is a tensor
    let results = (outs PTensor_PTensor);
}

#endif // _PTENSOR_OPS_TD_INCLUDED_
