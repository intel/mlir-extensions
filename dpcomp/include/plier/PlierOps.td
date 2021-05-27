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

#ifndef PLIER_OPS
#define PLIER_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def Plier_Dialect : Dialect {
  let name = "plier";
  let cppNamespace = "plier";
}

def Plier_PyType : DialectType<Plier_Dialect,
                 CPred<"$_self.isa<::plier::PyType>()">, "pytype">,
                 BuildableType<"$_builder.getType<::plier::PyType>()"> {
}

class Plier_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Plier_Dialect, mnemonic, traits>;

def ArgOp : Plier_Op<"arg", [NoSideEffect]> {
    let arguments = (ins
                     UI32Attr:$index,
                     StrAttr:$name);

    let results = (outs AnyType);
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "unsigned":$index, "::mlir::StringRef":$name)>
    ];
}

def ConstOp : Plier_Op<"const", [NoSideEffect]> {
    let arguments = (ins
                     AnyAttr:$val);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Attribute":$val)>
    ];
}

def GlobalOp : Plier_Op<"global", [NoSideEffect]> {
    let arguments = (ins
                     StrAttr:$name);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::StringRef":$name)>
    ];
}

def BinOp : Plier_Op<"binop", []> {
    let arguments = (ins
                     AnyType:$lhs,
                     AnyType:$rhs,
                     StrAttr:$op);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs, "::mlir::StringRef ":$op)>
    ];
}

def UnaryOp : Plier_Op<"unary", []> {
    let arguments = (ins
                     AnyType:$value,
                     StrAttr:$op);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value, "::mlir::StringRef ":$op)>
    ];
}

def CastOp : Plier_Op<"cast", []> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);
    let hasFolder = 1;
}

def PyCallOp : Plier_Op<"call", []> {
    let arguments = (ins
                     AnyType:$func,
                     Variadic<AnyType>:$args,
                     StrAttr:$func_name,
                     UI32Attr:$kw_start,
                     ArrayAttr:$kw_names);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$func, "::mlir::StringRef":$func_name,
                      "::mlir::ValueRange":$args,
                      "::mlir::ArrayRef<std::pair<std::string, ::mlir::Value>>":$kwargs)>
    ];
}

def BuildTupleOp : Plier_Op<"build_tuple", [NoSideEffect]> {
    let arguments = (ins
                     Variadic<AnyType>:$args);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$args)>
    ];
}

def GetItemOp : Plier_Op<"getitem", []> {
    let arguments = (ins
                     AnyType:$value,
                     AnyType:$index);

    let results = (outs AnyType);
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value, "::mlir::Value":$index)>
    ];
}

def StaticGetItemOp : Plier_Op<"static_getitem", []> {
    let arguments = (ins
                     AnyType:$value,
                     AnyType:$index_var,
                     UI32Attr:$index);

    let results = (outs AnyType);
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value, "::mlir::Value":$index_var, "unsigned":$index)>
    ];
}

def SetItemOp : Plier_Op<"setitem", []> {
    let arguments = (ins
                     AnyType:$target,
                     AnyType:$index,
                     AnyType:$value);

    let builders = [];
}

def GetiterOp : Plier_Op<"getiter", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value)>
    ];
}

def IternextOp : Plier_Op<"iternext", []> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value)>
    ];
}

def PairfirstOp : Plier_Op<"pair_first", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value)>
    ];
}

def PairsecondOp : Plier_Op<"pair_second", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value)>
    ];
}

def DelOp : Plier_Op<"del", []> {
    let arguments = (ins
                     AnyType:$value);
}

def GetattrOp : Plier_Op<"getattr", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value,
                     StrAttr:$name);

    let results = (outs AnyType);

    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value, "::mlir::StringRef":$name)>
    ];
}

def EnforceShapeOp : Plier_Op<"enforce_shape"> {
    let arguments = (ins AnyRankedTensor:$value,
                         Variadic<Index>:$sizes);

    let results = (outs AnyRankedTensor:$result);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value, "::mlir::ValueRange":$shape)>
    ];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def ParallelOp : Plier_Op<"parallel",
      [AttrSizedOperandSegments,
       DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"plier::YieldOp">,
       RecursiveSideEffects]> {

    let arguments = (ins Variadic<Index>:$lowerBounds,
                         Variadic<Index>:$upperBounds,
                         Variadic<Index>:$steps);
    let regions = (region SizedRegion<1>:$region);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$lowerBounds, "::mlir::ValueRange":$upperBounds, "::mlir::ValueRange":$steps,
            CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location, ::mlir::ValueRange, ::mlir::ValueRange, ::mlir::Value)>",
                 "nullptr">)>
    ];

    let extraClassDeclaration = [{
        unsigned getNumLoops() { return steps().size(); }
    }];
}

def YieldOp : Plier_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                                 ParentOneOf<["ParallelOp"]>]> {
    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    // Override default verifier (defined in SCF_Op), no custom verification
    // needed.
    let verifier = ?;
}

def UndefOp : Plier_Op<"undef", [NoSideEffect]> {
  let results = (outs AnyType);
}

def BuildSliceOp : Plier_Op<"build_slice", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$low,
                     AnyType:$high,
                     AnyType:$step);

    let results = (outs AnyType);
}

def RetainOp : Plier_Op<"retain", [ViewLikeOpInterface]> {
    let arguments = (ins AnyMemRef:$source);

    let results = (outs AnyMemRef);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$value)>
    ];

    let extraClassDeclaration = [{
        ::mlir::Value getViewSource() { return source(); }
    }];
}

#endif // PLIER_OPS
