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
/// This file defines the basic operations for the XeTile dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XeTile_OPS_TD_INCLUDED_
#define _XeTile_OPS_TD_INCLUDED_

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
        XeTile is the XeTile dialect's representation of a 2D tile.
        XeTile is a 2 dimensional block of data.

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

def XeTile_InitTileOp : XeTile_Op<"init_tile", [Pure]> {
    let summary = "Describes an XeTile with reference to a base memref";
    let description = [{
        The "init_tile" operation is used to describe a reduced-size view of a 2D base
        memref. This operation takes in a memref and returns an xetile.

        The operation takes in the following arguments:
        * source:  a 2D "base" memref represent a memory region.
        * offsets: memref-rank number of offsets into the "base" memref at which to
               create the tile. offsets can be operands (e.g., [%c0, %c]), attributes
               (e.g., [2, 4]), or mix of operand and attributs (e.g., [%c0, 4] and [2, %c0]).


        Example 1 (suppose the tile shape used by the compiler is 32x64):

        ```mlir
        %0 = memref.alloc() : memref<1024x1024xf32>
        %1 = xetile.init_tile %0[256, 512] : memref<1024x1024xf32> -> !xetile.tile<32x64xf32>
        ```

        Example 2 (suppose the tile shape used by the compiler is 32x64):

        ```mlir
        %0 = memref.alloc() : memref<1024x1024xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %1 = xetile.init_tile %0[%c128, %c256] : memref<1024x1024xf32> -> !xetile.tile<32x64xf32>
        ```

    }];

    let arguments = (ins AnyMemRef:$base,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$static_offsets);

    let results = (outs XeTile: $tile);

    let assemblyFormat = [{
        $base ``
        custom<DynamicIndexList>($offsets, $static_offsets)
        attr-dict `:` qualified(type($base)) `->` qualified(type($tile))
    }];
    let extraClassDeclaration = [{
        /// get the type of the base memref
        ::mlir::MemRefType getBaseType() { return getBase().getType().cast<::mlir::MemRefType>(); }

        /// Return the element type of the base memref
        ::mlir::Type getBaseElementType() {
            return getBaseType().getElementType();
        }

        /// Return the shape of the base memref
        ::llvm::ArrayRef<int64_t> getStaticBaseShape() {
            return getBaseType().getShape();
        }

        /// The result of an init_tile is always a Tile of TileType.
        TileType getType() {
            return getTile().getType().cast<TileType>();
        }

        /// Return the element type of the tile
        ::mlir::Type getElementType() {
            return getType().getElementType();
        }

        /// Return the shape of the tile
        ::llvm::ArrayRef<int64_t> getShape() {
            return getType().getShape();
        }

        /// Whether the given dimension size indicates a dynamic dimension.
        static constexpr bool isDynamic(int64_t dValue) {
            return dValue == ::mlir::ShapedType::kDynamic;
        }

        /// Whether the given shape has any size that indicates a dynamic dimension.
        static bool isDynamicShape(::llvm::ArrayRef<int64_t> dSizes) {
            return ::llvm::any_of(dSizes, [](int64_t dSize) { return isDynamic(dSize); });
        }

        bool hasStaticOffsets() {
            return !isDynamicShape(getStaticOffsets());
        }

        int getNumOfStaticOffsets() {
            return std::accumulate(getStaticOffsets().begin(), getStaticOffsets().end(), 0,
                                   [](int64_t a, int64_t b) { return isDynamic(b)? a: a+1;});
        }
    }];

    let hasVerifier = 1;

}

def XeTile_InitCoopTileOp : XeTile_Op<"init_coop_tile", [Pure]> {
    let summary = "splits XeTile into multiple smaller tiles for cooprative work among subgroups";
    let description = [{
        The "init_coop_tile" operation splits an XeTile among multiple subgroups within the same
        workgroup. This enables cooperative work among the subgroups such as cooperative prefetch,
        load A or B matrices. Subgroups within the same workgroup share the memory accesses.

        This operation has the following arguments:
        * source : this is the XeTile that will be split among subgroups
        * coop_id : ID assigned to each subgroup
        * coop_size : size assigned to each subgroup

        Example 1:

        ```mlir
        %1 = xetile.init_coop_tile %src, 0, 64
            : tile<64x64xf32>, index, index -> tile<8x8xf32>
        ```
    }];

    let arguments = (ins
        XeTile: $source,
        Index: $coop_id,
        Index: $coop_size);
    let results = (outs XeTile: $result);

    let assemblyFormat = [{
        $source `,` ` ` $coop_id `,` ` ` $coop_size attr-dict
        `:` qualified(type($source)) `,` qualified(type($coop_id))   `,` qualified(type($coop_size))
        `->` qualified(type($result))
    }];

}

def XeTile_LoadTileOp : XeTile_Op<"load_tile", []> {
    let summary = "Loads a tile into a register region";
    let description = [{
        "load_tile" operation loads the values of a tile into a register region with similar layout.
        Optionally the load operation can be performed in blocked layout as well. This is done by
        specifying a block factor which describes the size (rows and cols) of the block. Blocking
        does not change the order of the outer dimension. For exmaple, if a tile [m, n] is loaded
        with block factor [MB, NB] the resulting register region has the layout [m/MB, n/NB, MB, NB]

        "load_tile" also supports transpose.

        This operatio has following arguments:
        * source : source tile that is loaded from
        * block factor : optional 2-element array arrtibute to specify the size of the inner blocks
                            when loaded in the blocked layout
        * transpose : optional boolean attibute to specify if the output of the load will be
                        trasnposed or not

        Example 1:
        ```mlir
            %4 = xetile.load_tile %src {inner_blocks = [8, 16], TRANSPOSE = true}
                : tile<64x32xf32> -> vector<2x8x16x8xf32>
        ```
    }];

    let arguments = (ins
        XeTile: $source,
        OptionalAttr<ArrayAttr>: $inner_blocks,
        OptionalAttr<BoolAttr>: $transpose
    );
    let results = (outs Builtin_Vector: $result);

    let assemblyFormat = [{
        $source (`inner_blocks` `=` $inner_blocks^)? ` ` (`TRANSPOSE` `=` $transpose^)?
        attr-dict `:` qualified(type($source)) `->` qualified(type($result))
    }];

    let hasVerifier = 1;
}

def XeTile_StoreTileOp : XeTile_Op<"store_tile", []> {
    let summary = "stores a register region into memory";
    let description = [{
        "store_tile" operation can be used to store a register region into memory in plain layout.
        If a block factor is specified, the blocked vector is stored into memory in plan layout.

        This operation takes the following arguments:
        * tile : tile to store into
        * block : vector specifying the valur to store
        * block factor : optional 2-element array arrtibute to specify the size of the inner blocks
                            when stored in the blocked layout

        Example 1:
        ```mlir
            xetile.store_tile %dst, %value {inner_blocks = [8,16]}
                : (!tile<64x32xf32>,  vector<8x2x8x16xf32>)
        ```
    }];

    let arguments = (ins
        XeTile: $tile,
        Builtin_Vector: $block,
        OptionalAttr<ArrayAttr>: $inner_blocks
    );

    let assemblyFormat = [{
        $tile`,`` `$block (`inner_blocks` `=` $inner_blocks^)? attr-dict
            `:` `(` qualified(type($tile)) `,` qualified(type($block)) `)`
    }];
}

def XeTile_PrefetchTileOp : XeTile_Op<"prefetch_tile", []> {
    let summary = "preftech tiles into cache cooperatively";
    let description = [{
        "preftech_tile" operation can be used to prefetch smaller tiles into the cache.
        A larger tile can be split into smaller tiles using the init_coop_tile operation and
        then the resulting smaller tiles can be passed into this operation for prefetching.

        This operation takes following arguments:
        * tile : tile to prefetch into the cache

        Example 1:
        ```mlir
            %1 = xetile.init_coop_tile %src, %c0, %c1
                : tile<64x64xf32>, index, index -> tile<8x8xf32>

            xetile.prefetch_tile %1 : (tile<8x8xf32>)

        ```

    }];

    let arguments = (ins XeTile:$tile);

    let assemblyFormat = [{
        $tile attr-dict `:` `(` qualified(type($tile)) `)`
    }];
}

def XeTile_TileMMAOp : XeTile_Op<"tile_mma", [Pure]> {
    let summary = "matrix multiplication in blocked layout";
    let description = [{
        "tile_mma" operation represents matrix multiplication on tiles. This operation
        takes two input matrices (matrix A, matrix B) and an accumulator matrix (matrix C) to
        perform a general matrix multiplication.
            C_new = A * B + C
        Optionally inputs A, B, C can be in blocked layout where the block factor is specificed by
        an optional inner_blocks attribute.

        Arguments:
        * a : vector representing input matrix A
        * b : vector representing input matrix B
        * c : vector representing accumulator matrix C
        * a_inner_blocks : options block factor for matrix A if it is in blocked layout
        * b_inner_blocks : options block factor for matrix B if it is in blocked layout

        Example 1:
        ```mlir
            %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec
                : (vector<64x32xf32>, vector<32x128xf32>, vector<64x128xf32>) -> vector<64x128xf32>
        ```

        Example 2:
        ```mlir
            %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec a_inner_blocks=[8,8] b_inner_blocks=[8,16]
                : (vector<8x4x8x8xf32>, vector<4x8x8x16xf32>, vector<8x8x8x16xf32>) -> vector<8x8x8x16xf32>
        ```


    }];

    let arguments = (ins
        Builtin_Vector: $a,
        Builtin_Vector: $b,
        Builtin_Vector: $c,
        OptionalAttr<ArrayAttr>: $a_inner_blocks,
        OptionalAttr<ArrayAttr>: $b_inner_blocks
    );

    let results = (outs Builtin_Vector: $output);

    let assemblyFormat = [{
        $a`,` ` `$b`,` ` `$c
        (`a_inner_blocks` `=` $a_inner_blocks^)?
        (`b_inner_blocks` `=` $b_inner_blocks^)? attr-dict
            `:` `(`qualified(type($a))`,` ` `qualified(type($b))`,` ` `qualified(type($c))`)` `->` qualified(type($output))
    }];

    let extraClassDeclaration = [{
        mlir::VectorType getAType() { return llvm::cast<mlir::VectorType>(getA().getType()); }

        mlir::VectorType getBType() { return llvm::cast<mlir::VectorType>(getB().getType()); }

        mlir::VectorType getCType() { return llvm::cast<mlir::VectorType>(getC().getType()); }
    }];

    let hasVerifier = 1;
}

def XeTile_UpdateTileOffsetOp : XeTile_Op<"update_tile_offset", []> {
    let summary = "update the offsets of a tile";
    let description = [{
        "update_tile_offset" operation is used for iterating over the tiles. It takes in a
        tile and offsets x and y relative to the current tile to reach the new tile.

        Arguments:
        * tile : original tile
        * offset_x : offset along x dimension relative to the current tile to reach the next tile
        * offset_y : offset along y dimension relative to the current tile to reach the next tile

        Example 1:
        ```mlir
            xetile.update_tile_offset %tile, %offset_x, %offset_y
                : (tile<32x32xf32>, index, index)
        ```
    }];

    let arguments = (ins
        XeTile: $tile,
        Index: $offset_x,
        Index: $offset_y
    );

    let assemblyFormat = [{
        $tile `,` ` ` $offset_x `,` ` ` $offset_y attr-dict `:`
            `(` qualified(type($tile)) `,` ` ` qualified(type($offset_x)) `,` ` ` qualified(type($offset_y)) `)`
    }];
}


#endif // _XeTile_OPS_TD_INCLUDED_
