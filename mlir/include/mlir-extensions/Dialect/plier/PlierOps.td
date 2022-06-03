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
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/GPU/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def Plier_Dialect : Dialect {
  let name = "plier";
  let cppNamespace = "plier";

  let hasConstantMaterializer = 1;
}

def Plier_PyType
    : DialectType<Plier_Dialect, CPred<"$_self.isa<::plier::PyType>()">,
                  "pytype">,
      BuildableType<"$_builder.getType<::plier::PyType>()"> {}

def Plier_OpaqueType
    : DialectType<Plier_Dialect, CPred<"$_self.isa<::plier::OpaqueType>()">,
                  "opaque_type">,
      BuildableType<"$_builder.getType<::plier::OpaqueType>()"> {}

class Plier_Op<string mnemonic, list<Trait> traits = []>
    : Op<Plier_Dialect, mnemonic, traits>;

def ArgOp : Plier_Op<"arg", [NoSideEffect]> {
  let arguments = (ins UI32Attr : $index, StrAttr : $name);

  let results = (outs AnyType);
  let hasFolder = 1;

  let builders = [OpBuilder<(ins "unsigned"
                             : $index, "::mlir::StringRef"
                             : $name)>];
}

def ConstOp : Plier_Op<"const", [NoSideEffect]> {
  let arguments = (ins AnyAttr : $val);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Attribute" : $val)>];
}

def GlobalOp : Plier_Op<"global", [NoSideEffect]> {
  let arguments = (ins StrAttr : $name);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::StringRef" : $name)>];
}

def BinOp : Plier_Op<"binop", []> {
  let arguments = (ins AnyType : $lhs, AnyType : $rhs, StrAttr : $op);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $lhs, "::mlir::Value"
                             : $rhs, "::mlir::StringRef "
                             : $op)>];
}

def UnaryOp : Plier_Op<"unary", []> {
  let arguments = (ins AnyType : $value, StrAttr : $op);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::StringRef "
                             : $op)>];
}

def CastOp : Plier_Op<"cast", [NoSideEffect]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PyCallOp : Plier_Op<"call", [AttrSizedOperandSegments]> {
  let arguments = (ins
                     Optional<AnyType>:$func,
                     Variadic<AnyType>:$args,
                     Optional<AnyType>:$varargs,
                     Variadic<AnyType>:$kwargs,
                     StrAttr:$func_name,
                     ArrayAttr:$kw_names);

  let results = (outs AnyType);

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$func, "::mlir::StringRef":$func_name,
      "::mlir::ValueRange":$args, "::mlir::Value":$varargs,
      "::mlir::ArrayRef<std::pair<std::string, ::mlir::Value>>":$kwargs)>,
     OpBuilder<(ins
      "::mlir::Type":$type, "::mlir::Value":$func, "::mlir::StringRef":$func_name,
      "::mlir::ValueRange":$args, "::mlir::Value":$varargs,
      "::mlir::ArrayRef<std::pair<std::string, ::mlir::Value>>":$kwargs)>
   ];
}

def BuildTupleOp : Plier_Op<"build_tuple", [NoSideEffect]> {
  let arguments = (ins Variadic<AnyType> : $args);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::ValueRange" : $args)>];
}

def GetItemOp : Plier_Op<"getitem", [NoSideEffect]> {
  let arguments = (ins AnyType : $value, AnyType : $index);

  let results = (outs AnyType);
  let hasFolder = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::Value"
                             : $index)>];
}

def SetItemOp : Plier_Op<"setitem", []> {
  let arguments = (ins AnyType : $target, AnyType : $index, AnyType : $value);

  let builders = [];
}

def GetiterOp : Plier_Op<"getiter", [NoSideEffect]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def IternextOp : Plier_Op<"iternext", []> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def PairfirstOp : Plier_Op<"pair_first", [NoSideEffect]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def PairsecondOp : Plier_Op<"pair_second", [NoSideEffect]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def DelOp : Plier_Op<"del", []> { let arguments = (ins AnyType : $value); }

def GetattrOp : Plier_Op<"getattr", [NoSideEffect]> {
  let arguments = (ins AnyType : $value, StrAttr : $name);

  let results = (outs AnyType);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::StringRef"
                             : $name)>];
}

def ExhaustIterOp : Plier_Op<"exhaust_iter", [NoSideEffect]> {
  let arguments = (ins AnyType : $value, I64Attr : $count);

  let results = (outs AnyType);

  let hasFolder = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "int64_t"
                             : $count)>];
}

def BuildSliceOp : Plier_Op<"build_slice", [NoSideEffect]> {
  let arguments = (ins AnyType : $begin, AnyType : $end, AnyType : $step);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $begin, "::mlir::Value"
                             : $end, "::mlir::Value"
                             : $stride)>];
}

def SliceGetItemOp : Plier_Op<"slice_getitem", [NoSideEffect]> {
  let arguments = (ins AnyType : $slice, AnyType : $array, AnyType : $index,
                   I64Attr : $dim);

  let results = (outs AnyType);

  let hasCanonicalizer = 1;
}

#endif // PLIER_OPS
