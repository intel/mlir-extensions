//===---------------- XeTileTypes.td - XeTile dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the custom types used by XeTile dialect.
///
//===----------------------------------------------------------------------===//
#ifndef _XETILE_TYPES_TD_INCLUDED_
#define _XETILE_TYPES_TD_INCLUDED_


include "imex/Dialect/XeTile/IR/XeTileDialect.td"

// common base class for types in XeTile dialect
class XeTile_Type<string name, string typeMnemonic, list<Trait> traits = [],
                 string baseCppClass = "mlir::Type">
    : TypeDef<XeTile_Dialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
}

def XeTile : XeTile_Type<"Tile", "tile", [ShapedTypeInterface],
                            "mlir::TensorType">
{
  let summary = "A type representing a N-D tile";
  let description = [{
      Tile data type in XeTile dialect is used to represent an N-D memory region. This captures the
      2d shape and type of the memory region it points to. Optional encoding attribute can be
      attached to the tile type to carry extra information such as data layout information.

      Syntax:

      ```
      tile-type ::= `tile` `<` tile-dim-list element-type (`,` encoding )? `>`
      element-type ::= float-type | integer-type
      tile-dim-list := (decimal-literal `x`)*
      encoding ::= attribute-value
      ```

      Examples:

      ```mlir
      // 2D tile with i32 elements
      tile<3x42xi32>

      // 4D tile with f32 elements
      tile<4x5x6x7xf32>

      // 2D tile with i16 elements and encoding
      tile<64x64xi16, #encoding>

      ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        "mlir::Type":$elementType,
                        OptionalParameter<"mlir::Attribute">:$encoding);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>":$shape,
      "mlir::Type":$elementType,
      CArg<"mlir::Attribute", "{}">:$encoding
    ), [{
      return $_get(elementType.getContext(), shape, elementType, encoding);
    }]>
  ];

  let extraClassDeclaration = [{
    using mlir::ShapedType::Trait<TileType>::getElementTypeBitWidth;
    using mlir::ShapedType::Trait<TileType>::getRank;
    using mlir::ShapedType::Trait<TileType>::getNumElements;
    using mlir::ShapedType::Trait<TileType>::isDynamicDim;
    using mlir::ShapedType::Trait<TileType>::hasStaticShape;
    using mlir::ShapedType::Trait<TileType>::getNumDynamicDims;
    using mlir::ShapedType::Trait<TileType>::getDimSize;
    using mlir::ShapedType::Trait<TileType>::getDynamicDimIndex;

    TileType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, mlir::Type elementType) {
      return TileType::get(shape.value_or(getShape()), elementType, getEncoding());
    }

    TileType clone(mlir::Type elementType) {
      return llvm::cast<TileType>(cloneWith(getShape(), elementType));
    }

    TileType clone(llvm::ArrayRef<int64_t> shape) {
      return llvm::cast<TileType>(cloneWith(shape, getElementType()));
    }

    xetile::SubGroupMapAttr getSgMap() {
      auto encoding = llvm::dyn_cast_if_present<xetile::XeTileAttr>(getEncoding());
      if (encoding)
        return encoding.getSgMap();
      return xetile::SubGroupMapAttr();
    }

    xetile::WorkGroupMapAttr getWgMap() {
      auto wgmap = llvm::dyn_cast_if_present<xetile::WorkGroupMapAttr>(getEncoding());
      if (wgmap)
            return wgmap;
      auto encoding = llvm::dyn_cast_if_present<xetile::XeTileAttr>(getEncoding());
      if (encoding)
        return encoding.getWgMap();
      return xetile::WorkGroupMapAttr();
    }

    mlir::DenseI32ArrayAttr getOrder() {
      auto encoding = llvm::dyn_cast_if_present<xetile::XeTileAttr>(getEncoding());
      if (encoding)
        return encoding.getOrder();
      return mlir::DenseI32ArrayAttr::get(getContext(), {1, 0});
    }

    mlir::Attribute getMemorySpace() {
      auto encoding = llvm::dyn_cast_if_present<xetile::XeTileAttr>(getEncoding());
      if (encoding)
        return encoding.getMemorySpace();
      return mlir::Attribute();
    }

    int getMemorySpaceAsInt() {
      auto encoding = llvm::dyn_cast_if_present<xetile::XeTileAttr>(getEncoding());
      if (encoding && encoding.getMemorySpace()) {
        auto MemorySpace = encoding.getMemorySpace();
        assert(mlir::isa<mlir::IntegerAttr>(MemorySpace) &&
              "Using `getMemorySpaceAsInt` with non-Integer attribute");
        return mlir::cast<mlir::IntegerAttr>(MemorySpace).getInt();
      }
      // return default value 0 indicating Global memory
      return 0;
    }

    mlir::BoolAttr getScatterAttr() {
      auto encoding = llvm::dyn_cast_if_present<xetile::XeTileAttr>(getEncoding());
      if (encoding)
        return encoding.getScattered();
      return mlir::BoolAttr();
    }

  }];

  let assemblyFormat = "`<` custom<XeTileType>($shape, $elementType, $encoding) `>`";
  let genVerifyDecl = true;
}

// Integer types allowd in XeTile
def XeTile_IntType : AnyTypeOf<[I1, I8, I16, I32, I64, SI1, SI8, SI16, SI32, SI64, UI1, UI8, UI16, UI32, UI64]>;

// Float types allowed in XeTile
def XeTile_FloatType : AnyTypeOf<[F16, F32, F64, BF16, TF32]>;

// Define the scalar type for XeTile
def XeTile_ScalarType : AnyTypeOf<[XeTile_IntType, XeTile_FloatType, Index]>;

// define the source type for XeTile init_tile
def XeTile_BaseAddrType : AnyTypeOf<[MemRefOf<[XeTile_ScalarType]>, UI64, UI32, I64, I32]>;

// define the value type for XeTile load_tile and store_tile op
def XeTile_2DVector : VectorOfRankAndType<[2], [XeTile_ScalarType]>;

// input and output types needed for pack and unpack ops
def XeTile_4DVector : VectorOfRankAndType<[4], [XeTile_ScalarType]>;

// define the value type for XeTile load_gather and store_scatter op
def XeTile_1DOr2DVector: VectorOfRankAndType<[1, 2], [XeTile_ScalarType]>;

def XeTile_MaskType: VectorOfRankAndType<[1, 2], [I1]>;

// define the attribute type allowed for padding values for load op
def XeTile_PaddingValueAttr : AnyAttrOf<[I32Attr, F32Attr]>;


#endif // _XETILE_TYPES_TD_INCLUDED_
