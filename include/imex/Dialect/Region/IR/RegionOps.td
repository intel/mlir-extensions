//===- RegionOps.td - Region dialect  -------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the basic operations for the Region dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Region_OPS_TD_INCLUDED_
#define _Region_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Region_Dialect : Dialect {
    // The namespace of our dialect
    let name = "region";

    // A short one-line summary of our dialect.
    let summary = "Region operation to group operations.";

    // A longer description of our dialect.
    let description = [{
        A "region" executes operations inside it with a specific
        enviroment. Operations are executed exactly once. All SSA values that
        dominate the op can be accessed inside the op.
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::region";

    //  We use the default parser/printer which handles registered attrs
    let useDefaultAttributePrinterParser = true;
}


class Region_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Region_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def GPUEnvAttr
    : Region_Attr<"GPUEnv", "gpu_env"> {
  let summary = "Environment for tensors on GPU/s/devices";
  let parameters = (ins
    "::mlir::StringAttr":$device
    // "int16_t":$spirv_major_version,
    // "int16_t":$spirv_minor_version,
    // "bool":$has_fp16,
    // "bool":$has_fp64
  );
  let assemblyFormat = "`<` struct(params) `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$device
                                        // "int16_t":$spirv_major_version,
                                        // "int16_t":$spirv_minor_version,
                                        // "bool":$has_fp16,
                                        // "bool":$has_fp64
                                        ), [{
      return get(device.getContext(), device); //, spirv_major_version, spirv_minor_version, has_fp16, has_fp64);
    }]>
  ];
}


// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Region_Op<string mnemonic, list<Trait> traits = []> :
    Op<Region_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// EnvironmentRegionOp
//===----------------------------------------------------------------------===//

def EnvironmentRegionOp : Region_Op<"env_region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"::imex::region::EnvironmentRegionYieldOp">,
    RecursiveMemoryEffects]> {
  let summary = "Operation that executes its region with a specific environment";
  let description = [{
    "env_region" executes operations inside its region within a specific
    environment. Operations are executed exactly once. All SSA values that
    dominate the op can be accessed inside the op.

    "env_region" takes "environment" attribute and zero or more SSA arguments.

    Actual intepretation of the "environment" attribute and arguments is not
    specified here and is left to the lowering passes.

    Values can be yielded from "env_region" region using "env_region_yield" op.
    Values yielded from this op's region define the op's results.

    Regions can be arbitrarily nested into each other, it is up to specific
    passes how to interpret nested env regions.
  }];

  let arguments = (ins AnyAttr:$environment, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat =
      "attr-dict $environment ($args^ `:` type($args))? (`->` type($results)^)? $region";

  let extraClassDeclaration = [{
    /// Inline op body into parent region and erase the op.
    static void inlineIntoParent(::mlir::PatternRewriter &builder, EnvironmentRegionOp op);
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Attribute":$environment,
      CArg<"::mlir::ValueRange", "{}">:$args,
      CArg<"::mlir::TypeRange", "{}">:$results,
      CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>", "nullptr">)>
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// EnvironmentRegionYieldOp
//===----------------------------------------------------------------------===//

def EnvironmentRegionYieldOp : Region_Op<"env_region_yield", [
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"::imex::region::EnvironmentRegionOp">
  ]> {

  let summary = "Environment region yield and termination operation";
  let description = [{
    "env_region_yield" yields an SSA value from the "env_region" op region and
    terminates it.

    If "env_region_yield" has any operands, the operands must match the parent
    operation's results.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

#endif // _Region_OPS_TD_INCLUDED_
