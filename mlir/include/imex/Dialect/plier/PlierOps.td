// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PLIER_OPS
#define PLIER_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def Plier_Dialect : Dialect {
  let name = "plier";
  let cppNamespace = "plier";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

class Plier_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Plier_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Plier_SliceType : Plier_Type<"Slice", "slice", [], "::mlir::Type">;

def Plier_Py : Plier_Type<"Py", "pytype", [], "::mlir::Type"> {

  let parameters = (ins "::mlir::StringAttr":$name);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::MLIRContext *":$context,
      "::llvm::StringRef":$name
    ), [{
      return $_get(context, ::mlir::StringAttr::get(context, name));
    }]>
  ];

  let extraClassDeclaration = [{
    static PyType getUndefined(::mlir::MLIRContext *context) {
      return PyType::get(context, "");
    }
  }];

  let assemblyFormat = "`<` $name `>`";
}

def Plier_OmittedType : Plier_Type<"Omitted", "omitted", [], "::mlir::Type"> {

  let parameters = (ins
    "::mlir::Type":$type,
    OptionalParameter<"::mlir::Attribute">:$value
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$type,
      CArg<"::mlir::Attribute", "{}">:$value
    ), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $type (`,` $value^)? `>`";
}

class Plier_Op<string mnemonic, list<Trait> traits = []>
    : Op<Plier_Dialect, mnemonic, traits>;

def ArgOp : Plier_Op<"arg", [Pure]> {
  let arguments = (ins UI32Attr : $index, StrAttr : $name);

  let results = (outs AnyType);
  let hasFolder = 1;

  let builders = [OpBuilder<(ins "unsigned"
                             : $index, "::mlir::StringRef"
                             : $name)>];
}

def ConstOp : Plier_Op<"const", [Pure]> {
  let arguments = (ins AnyAttr : $val);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Attribute" : $val)>];
}

def GlobalOp : Plier_Op<"global", [Pure]> {
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

def CastOp : Plier_Op<"cast", [Pure]> {
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

def BuildTupleOp : Plier_Op<"build_tuple", [Pure]> {
  let arguments = (ins Variadic<AnyType> : $args);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::ValueRange" : $args)>];
}

def GetItemOp : Plier_Op<"getitem", [Pure]> {
  let arguments = (ins AnyType : $value, AnyType : $index);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::Value"
                             : $index)>];
}

def SetItemOp : Plier_Op<"setitem", []> {
  let arguments = (ins AnyType : $target, AnyType : $index, AnyType : $value);

  let builders = [];
}

def GetiterOp : Plier_Op<"getiter", [Pure]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def IternextOp : Plier_Op<"iternext", []> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def PairfirstOp : Plier_Op<"pair_first", [Pure]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def PairsecondOp : Plier_Op<"pair_second", [Pure]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];
}

def DelOp : Plier_Op<"del", []> { let arguments = (ins AnyType : $value); }

def GetattrOp : Plier_Op<"getattr", [Pure]> {
  let arguments = (ins AnyType : $value, StrAttr : $name);

  let results = (outs AnyType);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::StringRef"
                             : $name)>];
}

def ExhaustIterOp : Plier_Op<"exhaust_iter", [Pure]> {
  let arguments = (ins AnyType : $value, I64Attr : $count);

  let results = (outs AnyType);

  let hasFolder = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "int64_t"
                             : $count)>];
}

def BuildSliceOp : Plier_Op<"build_slice", [Pure]> {
  let arguments = (ins AnyType : $begin, AnyType : $end, AnyType : $step);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $begin, "::mlir::Value"
                             : $end, "::mlir::Value"
                             : $stride)>];
}

def SliceGetItemOp : Plier_Op<"slice_getitem", [Pure]> {
  let arguments = (ins AnyType : $slice, AnyType : $array, AnyType : $index,
                   I64Attr : $dim);

  let results = (outs AnyType);
}

#endif // PLIER_OPS
