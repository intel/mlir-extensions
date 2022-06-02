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

def PlierUtil_Dialect : Dialect {
  let name = "plier_util";
  let cppNamespace = "plier";

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
}

def PlierUtil_OpaqueType
    : DialectType<PlierUtil_Dialect, CPred<"$_self.isa<::plier::OpaqueType>()">,
                  "opaque_type">,
      BuildableType<"$_builder.getType<::plier::OpaqueType>()"> {}

class PlierUtil_Op<string mnemonic, list<Trait> traits = []>
    : Op<PlierUtil_Dialect, mnemonic, traits>;

def EnforceShapeOp : PlierUtil_Op<"enforce_shape"> {
  let arguments = (ins AnyRankedTensor : $value, Variadic<Index> : $sizes);

  let results = (outs AnyRankedTensor : $result);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::ValueRange"
                             : $shape)>];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ParallelOp : PlierUtil_Op<"parallel", [
  AttrSizedOperandSegments, DeclareOpInterfaceMethods<LoopLikeOpInterface>,
  SingleBlockImplicitTerminator<"plier::YieldOp">, RecursiveSideEffects
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

def UndefOp : PlierUtil_Op<"undef", [NoSideEffect]> {
  let results = (outs AnyType);
}

def YieldOp : PlierUtil_Op<"yield", [
  NoSideEffect, ReturnLike, Terminator, ParentOneOf<["ParallelOp"]>
]> {
  let arguments = (ins Variadic<AnyType> : $results);
  let builders = [OpBuilder<(ins), [{/* nothing to do */}]>];
  // Override default verifier (defined in SCF_Op), no custom verification
  // needed.
  let hasVerifier = 0;
}

def RetainOp : PlierUtil_Op<"retain", [ViewLikeOpInterface]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def ChangeLayoutOp : PlierUtil_Op<"change_layout", [ViewLikeOpInterface, NoSideEffect]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef);
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def SignCastOp : PlierUtil_Op<"sign_cast", [NoSideEffect]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ExtractMemrefMetadataOp
    : PlierUtil_Op<"extract_memref_metadata", [NoSideEffect]> {
  let arguments = (ins AnyMemRef : $source, IndexAttr : $dimIndex);

  let results = (outs Index : $result);
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value" : $src,
                 "int64_t" : $dim)>,
    OpBuilder<(ins "::mlir::Value" : $src)>
  ];
}

def PseudoCopyOp : PlierUtil_Op<"pseudo_copy", [ViewLikeOpInterface]> {
  let arguments = (ins AnyType : $source);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def ForceCopyOp : PlierUtil_Op<"force_copy", [ViewLikeOpInterface]> {
  let arguments = (ins AnyRankedTensor : $source);

  let results = (outs AnyRankedTensor);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def TakeContextOp : PlierUtil_Op<"take_context"> {
  let arguments = (ins OptionalAttr<SymbolRefAttr>:$initFunc,
                       OptionalAttr<SymbolRefAttr>:$releaseFunc);

  let results = (outs PlierUtil_OpaqueType:$context,
                      Variadic<AnyType>:$results);

    let builders = [
      OpBuilder<(ins "::mlir::SymbolRefAttr":$initFunc,
                     "::mlir::SymbolRefAttr":$releaseFunc,
                     "::mlir::TypeRange":$resultTypes)>
   ];
}

def ReleaseContextOp : PlierUtil_Op<"release_context"> {
  let arguments = (ins PlierUtil_OpaqueType:$context);
}

#endif // PLIER_UTIL_OPS
