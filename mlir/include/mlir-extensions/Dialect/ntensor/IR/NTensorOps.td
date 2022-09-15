// Copyright 2022 Intel Corporation
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

#ifndef NTENSOR_OPS
#define NTENSOR_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SubElementInterfaces.td"

def NTensor_Dialect : Dialect {
  let name = "ntensor";
  let cppNamespace = "::imex::ntensor";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

class NTensor_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<NTensor_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class NTensor_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<NTensor_Dialect, mnemonic, traits>;

def NTensor_Tensor : NTensor_Type<"NTensor", "ntensor", [
  DeclareTypeInterfaceMethods<SubElementTypeInterface>, ShapedTypeInterface],
  "::imex::ntensor::NTensorBase"> {
  let summary = "Multi-dimensional array with a fixed number of dimensions";
  let description = [{
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType,
    DefaultValuedParameter<"::mlir::Attribute", "">:$environment
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::llvm::ArrayRef<int64_t>":$shape,
      "::mlir::Type":$elementType,
      CArg<"::mlir::Attribute", "{}">:$environment
    ), [{
      return $_get(elementType.getContext(), shape, elementType, environment);
    }]>
  ];

  let extraClassDeclaration = [{
    using ::mlir::ShapedType::Trait<NTensorType>::clone;
    using ::mlir::ShapedType::Trait<NTensorType>::getElementTypeBitWidth;
    using ::mlir::ShapedType::Trait<NTensorType>::getRank;
    using ::mlir::ShapedType::Trait<NTensorType>::getNumElements;
    using ::mlir::ShapedType::Trait<NTensorType>::isDynamicDim;
    using ::mlir::ShapedType::Trait<NTensorType>::hasStaticShape;
    using ::mlir::ShapedType::Trait<NTensorType>::getNumDynamicDims;
    using ::mlir::ShapedType::Trait<NTensorType>::getDimSize;
    using ::mlir::ShapedType::Trait<NTensorType>::getDynamicDimIndex;
  }];

  let assemblyFormat = "`<` custom<Shape>($shape, $elementType) (`,` $environment^)? `>`";
}

def UnaryOp : NTensor_OpBase<"unary", []> {
  let summary = "Unary operation.";
  let description = [{

  }];

  let arguments = (ins AnyType:$value, StrAttr:$op);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$op ` ` attr-dict `(` $value `:` type($value) `)` `->` type($result)";
}

def BinaryOp : NTensor_OpBase<"binary", []> {
  let summary = "Binary operation.";
  let description = [{

  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$op);

  let results = (outs AnyType:$result);
}

def CallOp : NTensor_OpBase<"call", []> {
  let summary = "Call operation.";
  let description = [{

  }];

  let arguments = (ins Variadic<AnyType>:$args, StrAttr:$op);

  let results = (outs Variadic<AnyType>:$results);
}

def SetitemOp : NTensor_OpBase<"setitem", []> {
  let summary = "Array setitem operation.";
  let description = [{

  }];

  let arguments = (ins AnyType:$source, AnyType:$index, AnyType:$value);
}

def GetitemOp : NTensor_OpBase<"getitem", [NoSideEffect]> {
  let summary = "Array setitem operation.";
  let description = [{

  }];

  let arguments = (ins AnyType:$source, AnyType:$index);

  let results = (outs AnyType:$result);
}

def PrimitiveOp : NTensor_OpBase<"primitive", []> {
  let summary = "Primitive operation.";
  let description = [{

  }];

  let arguments = (ins Variadic<AnyType>:$args, StrAttr:$op);

  let results = (outs Variadic<AnyType>:$results);
}

#endif // NTENSOR_OPS
