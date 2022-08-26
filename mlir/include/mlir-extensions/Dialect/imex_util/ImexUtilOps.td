// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PLIER_UTIL_OPS
#define PLIER_UTIL_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def ImexUtil_Dialect : Dialect {
  let name = "imex_util";
  let cppNamespace = "::imex::util";

  let dependentDialects = ["::mlir::memref::MemRefDialect"];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

def ImexUtil_OpaqueType
    : DialectType<ImexUtil_Dialect, CPred<"$_self.isa<::imex::util::OpaqueType>()">,
                  "opaque_type">,
      BuildableType<"$_builder.getType<::imex::util::OpaqueType>()"> {}

class ImexUtil_Op<string mnemonic, list<Trait> traits = []>
    : Op<ImexUtil_Dialect, mnemonic, traits>;

def EnforceShapeOp : ImexUtil_Op<"enforce_shape"> {
  let arguments = (ins AnyRankedTensor : $value, Variadic<Index> : $sizes);

  let results = (outs AnyRankedTensor : $result);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::ValueRange"
                             : $shape)>];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ParallelOp : ImexUtil_Op<"parallel", [
  AttrSizedOperandSegments, DeclareOpInterfaceMethods<LoopLikeOpInterface>,
  SingleBlockImplicitTerminator<"::imex::util::YieldOp">, RecursiveSideEffects
]> {

  let arguments = (ins Variadic<Index>:$lowerBounds,
                         Variadic<Index>:$upperBounds,
                         Variadic<Index>:$steps);
  let regions = (region SizedRegion<1> : $region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "::mlir::ValueRange"
      : $lowerBounds, "::mlir::ValueRange"
      : $upperBounds, "::mlir::ValueRange"
      : $steps,
        CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location, "
             "::mlir::ValueRange, ::mlir::ValueRange, ::mlir::Value)>",
             "nullptr">)>];

    let extraClassDeclaration = [{
        unsigned getNumLoops() { return steps().size(); }
    }];
}

def UndefOp : ImexUtil_Op<"undef", [NoSideEffect]> {
  let results = (outs AnyType);
}

def YieldOp : ImexUtil_Op<"yield", [
  NoSideEffect, ReturnLike, Terminator, ParentOneOf<["::imex::util::ParallelOp"]>
]> {
  let arguments = (ins Variadic<AnyType> : $results);
  let builders = [OpBuilder<(ins), [{/* nothing to do */}]>];
  // Override default verifier (defined in SCF_Op), no custom verification
  // needed.
  let hasVerifier = 0;
}

def RetainOp : ImexUtil_Op<"retain", [ViewLikeOpInterface]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def ChangeLayoutOp : ImexUtil_Op<"change_layout", [ViewLikeOpInterface, NoSideEffect]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef : $dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def SignCastOp : ImexUtil_Op<"sign_cast", [NoSideEffect]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ExtractMemrefMetadataOp
    : ImexUtil_Op<"extract_memref_metadata", [NoSideEffect]> {
  let arguments = (ins AnyMemRef : $source, IndexAttr : $dimIndex);

  let results = (outs Index : $result);
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value" : $src,
                 "int64_t" : $dim)>,
    OpBuilder<(ins "::mlir::Value" : $src)>
  ];
}

def PseudoCopyOp : ImexUtil_Op<"pseudo_copy", [ViewLikeOpInterface]> {
  let arguments = (ins AnyType : $source);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def ForceCopyOp : ImexUtil_Op<"force_copy", [ViewLikeOpInterface]> {
  let arguments = (ins AnyRankedTensor : $source);

  let results = (outs AnyRankedTensor);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def TakeContextOp : ImexUtil_Op<"take_context"> {
  let arguments = (ins OptionalAttr<SymbolRefAttr>:$initFunc,
                       OptionalAttr<SymbolRefAttr>:$releaseFunc);

  let results = (outs ImexUtil_OpaqueType:$context,
                      Variadic<AnyType>:$results);

    let builders = [
      OpBuilder<(ins "::mlir::SymbolRefAttr":$initFunc,
                     "::mlir::SymbolRefAttr":$releaseFunc,
                     "::mlir::TypeRange":$resultTypes)>
   ];
}

def ReleaseContextOp : ImexUtil_Op<"release_context"> {
  let arguments = (ins ImexUtil_OpaqueType:$context);
}

def BuildTupleOp : ImexUtil_Op<"build_tuple", [NoSideEffect]> {
  let summary = "Constructs tuple from provided values";
  let description = [{
    "build_tuple" takes multiple SSA values and constructs tuple from them.

    Zero arguments are allowed and will result in an empty tuple.
  }];

  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs AnyTuple:$result);

  let assemblyFormat = "attr-dict ($args^ `:` type($args) `->`)? type($result)";
}

def TupleExtractOp : ImexUtil_Op<"tuple_extract", [NoSideEffect]> {
  let summary = "Extracts value from tuple";
  let description = [{
    "tuple_extract" extracts element with specific index from tuple.

    If index is out of bounds behaviour is unspecified and is left to the
    lowering passes.
  }];

  let arguments = (ins AnyTuple:$source,
                       Index:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $source `:` type($source) `,` $index `->` type($result)";

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    ::llvm::Optional<int64_t> getConstantIndex();
  }];

  let hasFolder = 1;
}

def EnvironmentRegionOp : PlierUtil_Op<"env_region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"plier::EnvironmentRegionYieldOp">,
    RecursiveSideEffects]> {
  let summary = "Operation that executes its region with a specific environment";
  let description = [{
    "env_region" executes operations inside its region within a specific
    enviroment. Operations are executed exactly once. All SSA values that
    dominate the op can be accessed inside the op.

    "env_region" takes "environment" attribute and zero or more SSA arguments.

    Actual intepretation of the "environment" attribute and arguments is not
    specified here and is left to the lowering passes.

    Values can be yielded from "env_region" region using "env_region_yield" op.
    Values yielded from this op's region define the op's results.
  }];


  let arguments = (ins AnyAttr:$environment, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat =
      "attr-dict $environment ($args^ `:` type($args))? (`->` type($results)^)? $region";
}

def EnvironmentRegionYieldOp : PlierUtil_Op<"env_region_yield", [
  NoSideEffect,
  ReturnLike,
  Terminator,
  HasParent<"EnvironmentRegionOp">
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

#endif // PLIER_UTIL_OPS
