//===- XeTileOps.td - XeTile dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the XeTile dialect and its types.
///
//===----------------------------------------------------------------------===//

#ifndef _XeTile_BASE_TD_INCLUDED_
#define _XeTile_BASE_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"

// Provide a definition of the 'XeTile' dialect in the ODS framework so that we
// can define our operations.
def XeTile_Dialect : Dialect {
    // The namespace of our dialect
    let name = "xetile";

    // A short one-line summary
    let summary = "A dialect for enabling tile-base programming at subgroup level";

    // A longer description
    let description = [{
        XeTile provides an abstraction supporting tile-based computation to simplify the
        lowering of DNN operation like matrix multiplication. XeTile dialect works at tile sizes
        that are larger than the tile sizes supported by the hardware. XeTile dilaect also hides
        the auto-padding requirements for out-of-bound memory accesses and, supports arbitrary
        input matrix sizes.
    }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::xetile";

    let dependentDialects = [
          "::mlir::memref::MemRefDialect"];

    // TODO: temporary disable it.
    let useDefaultTypePrinterParser = true;
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class XeTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<XeTile_Dialect, mnemonic, traits>;

// common base class for types in XeTile dialect
class XeTile_Type<string name, string typeMnemonic, list<Trait> traits = [],
                 string baseCppClass = "::mlir::Type">
    : TypeDef<XeTile_Dialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
}

def XeTile : XeTile_Type<"Tile", "tile", [ShapedTypeInterface],
                            "::imex::xetile::TileBase">
{
    let summary = "A type representing a 2D tile";
    let description = [{
        Tile data type in XeTile dialect is used to represent a 2D memory region.
        This captures the 2d shape and type of the memory region it points to.

        Syntax:

        ```
        tile-type ::= `vector` `<` vector-dim-list vector-element-type `>`
        tile-element-type ::= float-type | integer-type | index-type
        tile-dim-list := (static-dim-list `x`)?
        static-dim-list ::= decimal-literal `x` decimal-literal
        ```

        Examples:

        ```mlir
        // A tile with i32 elements
        tile<3x42xi32>

        // A tile with f32 elements
        tile<4x5xf32>
        ```
    }];

    let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                          "::mlir::Type":$elementType);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "::llvm::ArrayRef<int64_t>":$shape, "::mlir::Type":$elementType), [{
                assert(shape.size()==2);
                return $_get(elementType.getContext(), shape, elementType);
            }]>,
        TypeBuilderWithInferredContext<(ins
            "int64_t":$dim0, "int64_t":$dim1, "::mlir::Type":$elementType), [{
                llvm::SmallVector<int64_t, 2> shape{dim0, dim1};
                assert(shape.size()==2);
                return $_get(elementType.getContext(), shape, elementType);
            }]>
    ];

    let extraClassDeclaration = [{
        using ::mlir::ShapedType::Trait<TileType>::clone;
        using ::mlir::ShapedType::Trait<TileType>::getElementTypeBitWidth;
        using ::mlir::ShapedType::Trait<TileType>::getRank;
        using ::mlir::ShapedType::Trait<TileType>::getNumElements;
        using ::mlir::ShapedType::Trait<TileType>::isDynamicDim;
        using ::mlir::ShapedType::Trait<TileType>::hasStaticShape;
        using ::mlir::ShapedType::Trait<TileType>::getNumDynamicDims;
        using ::mlir::ShapedType::Trait<TileType>::getDimSize;
        using ::mlir::ShapedType::Trait<TileType>::getDynamicDimIndex;
    }];

    let assemblyFormat = "`<` custom<Shape>($shape, $elementType) `>`";
}

// Integer types allowd in XeTile
def XeTile_IntType : AnyTypeOf<[I1, I8, I16, I32, I64, SI1, SI8, SI16, SI32, SI64, UI1, UI8, UI16, UI32, UI64]>;

// Float types allowed in XeTile
def XeTile_FloatType : AnyTypeOf<[F16, F32, F64, BF16, F8E4M3FN, F8E5M2, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2FNUZ]>;

// Define the scalar type for XeTile
def XeTile_ScalarType : AnyTypeOf<[XeTile_IntType, XeTile_FloatType]>;

// define a 2D memref of XeTile scalar type
def XeTile_2DMemRef : MemRefRankOf<[XeTile_ScalarType], [2]>;

// define the source type for XeTile init_tile
def XeTile_BaseAddrType : AnyTypeOf<[XeTile_2DMemRef, I64]>;

// define the attribute type allowed for padding values for load op
def XeTile_PaddingValueAttr : AnyAttrOf<[I32Attr, F32Attr]>;

#endif // _XeTile_BASE_TD_INCLUDED_
