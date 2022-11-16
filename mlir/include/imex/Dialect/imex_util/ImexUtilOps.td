// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class ImexUtil_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<ImexUtil_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class ImexUtil_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ImexUtil_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class ImexUtil_Op<string mnemonic, list<Trait> traits = []>
    : Op<ImexUtil_Dialect, mnemonic, traits>;

def ImexUtil_OpaqueType : ImexUtil_Type<"Opaque", "opaque", [], "::mlir::Type">;

def ImexUtil_TypeVar : ImexUtil_Type<"TypeVar", "typevar", [], "::mlir::Type"> {
  let parameters = (ins
    "::mlir::Type":$type
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let assemblyFormat = "`<` $type `>`";
}

def IndexRangeAttr
    : ImexUtil_Attr<"IndexRange", "index_range"> {
  let parameters = (ins
    "int64_t":$min,
    "int64_t":$max
    );
  let assemblyFormat = "`<` `[` $min `,` $max `]` `>`";
}

def EnforceShapeOp : ImexUtil_Op<"enforce_shape"> {

  let arguments = (ins AnyShaped:$value, Variadic<Index>:$sizes);
  let results = (outs AnyShaped:$result);

  let builders = [OpBuilder<(ins
    "::mlir::Value":$value,
    "::mlir::ValueRange":$shape)
  >];

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict $value `:` type($value) `(` $sizes `)` `->` type($result)";
}

def ParallelOp : ImexUtil_Op<"parallel", [
  AttrSizedOperandSegments, DeclareOpInterfaceMethods<LoopLikeOpInterface>,
  SingleBlockImplicitTerminator<"::imex::util::YieldOp">, RecursiveMemoryEffects
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
        unsigned getNumLoops() { return getSteps().size(); }
    }];
}

def UndefOp : ImexUtil_Op<"undef", [Pure]> {
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def YieldOp : ImexUtil_Op<"yield", [
  Pure, ReturnLike, Terminator, ParentOneOf<["::imex::util::ParallelOp"]>
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
      ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def ChangeLayoutOp : ImexUtil_Op<"change_layout", [ViewLikeOpInterface, Pure]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef : $dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def SignCastOp : ImexUtil_Op<"sign_cast", [Pure]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType:$dest);
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($dest)";
}

def ExtractMemrefMetadataOp
    : ImexUtil_Op<"extract_memref_metadata", [Pure]> {
  let arguments = (ins AnyMemRef : $source, IndexAttr : $dimIndex);

  let results = (outs Index : $result);
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value" : $src,
                 "int64_t" : $dim)>,
    OpBuilder<(ins "::mlir::Value" : $src)>
  ];
}

def ForceCopyOp : ImexUtil_Op<"force_copy", [ViewLikeOpInterface]> {
  let arguments = (ins AnyRankedTensor : $source);

  let results = (outs AnyRankedTensor);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
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

def BuildTupleOp : ImexUtil_Op<"build_tuple", [Pure]> {
  let summary = "Constructs tuple from provided values";
  let description = [{
    "build_tuple" takes multiple SSA values and constructs tuple from them.

    Zero arguments are allowed and will result in an empty tuple.
  }];

  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs AnyTuple:$result);

  let assemblyFormat = "attr-dict ($args^ `:` type($args) `->`)? type($result)";
}

def TupleExtractOp : ImexUtil_Op<"tuple_extract", [Pure]> {
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

def EnvironmentRegionOp : ImexUtil_Op<"env_region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"::imex::util::EnvironmentRegionYieldOp">,
    RecursiveMemoryEffects]> {
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

  let extraClassDeclaration = [{
    /// Inline op body into parent region and erase the op.
    static void inlineIntoParent(::mlir::PatternRewriter &builder, EnvironmentRegionOp op);
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Attribute":$environment,
      CArg<"::mlir::ValueRange", "llvm::None">:$args,
      CArg<"::mlir::TypeRange", "llvm::None">:$results,
      CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>", "nullptr">)>
  ];

  let hasCanonicalizer = 1;
}

def EnvironmentRegionYieldOp : ImexUtil_Op<"env_region_yield", [
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"::imex::util::EnvironmentRegionOp">
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

def MemrefApplyOffsetOp : ImexUtil_Op<"memref_apply_offset", [
    Pure, ViewLikeOpInterface
  ]> {

  let summary = "Applies memref offset to the underlying data pointer";
  let description = [{
    "memref_apply_offset" applies memref offset to the underlying data pointer.
    Returned memref will always have dynamic offset of 0.
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

// TODO: Upstream arith.bitcast doesnt suuport casting between vectors and
// non-vectors
def BitcastOp : ImexUtil_Op<"bitcast", [Pure]> {

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

// TODO: Upstream
def MemrefBitcastOp : ImexUtil_Op<"memref_bitcast", [
    Pure, ViewLikeOpInterface, SameOperandsElementType, SameOperandsShape
  ]> {

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return getSource(); }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

#endif // PLIER_UTIL_OPS
