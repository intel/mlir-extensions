//===---------------- XeTileOps.td - XeTile dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the operations for the XeTile dialect.
///
//===----------------------------------------------------------------------===//
#ifndef _XETILE_OPS_TD_INCLUDED_
#define _XETILE_OPS_TD_INCLUDED_

include "imex/Dialect/XeTile/IR/XeTileDialect.td"
include "imex/Dialect/XeTile/IR/XeTileTypes.td"
include "imex/Dialect/XeTile/IR/XeTileAttrs.td"

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class XeTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<XeTile_Dialect, mnemonic, traits>;

def XeTile_InitTileOp : XeTile_Op<"init_tile", [Pure, AttrSizedOperandSegments]> {
    let summary = "Describes an XeTile with reference to a base memref";
    let description = [{
        The "init_tile" operation is used to describe a 2D region (i.e. tile) in gloabl memory.
        This operation takes in a 2D memref or an address and return an xetile. If dynamic-shaped
        memref or an address is used as the base, it is required to specify the shape and strides
        of the memory region described by the tile.

        Optionally, the tile can be described in blocked layout as well. This is done by specifying
        an "inner_blocks" attribute which describes the size (rows and cols) of the block. This attribute
        is used by later lowering passes to detremine the 2D block load/store sizes.

        The operation takes in the following arguments:
        * source:  Source can be static/dynamic shaped memref or an address (i64)
        * offsets: 2 offsets into the "source" memref or address at which to
               create the tile. offsets can be operands (e.g., [%c0, %c]), attributes
               (e.g., [2, 4]), or mix of operand and attributes (e.g., [%c0, 4] and [2, %c0]).
        * dynamic_offsets : This is a subset of "offsets". offsets can contain both static and dynamic
                values. "dynamic_offsets" captures the dynamic subset of the offsets.
        * dynamic_shape : 2 shape arguments specifying the size of 2 dimensions of the "source".
                This is only required if a dynmaic shaped memref or an address is used as "source".
                dynamic_shapes needs to be operands i.e. dynamic SSA values (e.g., [%c128, %c128]).
        * dynamic_strides : 2 stride arguments specifying the strides of the 2D "source" memory region.
                This is only required if a dynmaic shaped memref or an address is used as "source".
                dynamic_strides needs to be operands i.e. dynamic SSA values (e.g., [%c128, %c1]).
        * inner_blocks : Optional 2 element integer array describing [rows, cols] of the blocked layout.

        For the follwing examples, suppose the tile shape used by the compiler is 32x64.

        Example 1:
        Creating an xetile using a static shaped 2D memref.

        ```mlir
        %0 = memref.alloc() : memref<1024x1024xf32>
        %c128 = arith.constant 128 : index
        %2 = xetile.init_tile %0[%c128, 512] : memref<1024x1024xf32> -> !xetile.tile<32x64xf32>
        ```

        Example 2:
        Creating an xetile using a static shaped 2D memref and inner blocks attribute.

        ```mlir
        %0 = memref.alloc() : memref<1024x1024xf32>
        %c128 = arith.constant 128 : index
        %2 = xetile.init_tile %0[%c128, 512] { inner_blocks = [8, 16]} : memref<1024x1024xf32> -> !xetile.tile<32x64xf32>
        ```

        Example 2:
        Creating an xetile using a dynamic shaped 2D memref.

        ```mlir
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        %src = memref.alloc(%c1024, %c512) : memref<?x?xf32>
        %1 = xetile.init_tile %src[256, %c64], [%c1024, %c1024], [%c1024, %c1] : memref<?x?xf32> -> !xetile.tile<32x64xf32>
        ```

        Example 3:
        Creating an xetile using an address

        ```mlir
        %src = .... : i64
        ...
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %1 = xetile.init_tile %src[%c128, %c256], [%c1024, %c1024], [%c1024, %c1] : i64 -> !xetile.tile<32x64xf32>
        ```

    }];

    let arguments = (ins XeTile_BaseAddrType:$source,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$static_offsets,
                       Variadic<Index>:$dynamic_shape,
                       Variadic<Index>:$dynamic_strides
                       );

    let results = (outs XeTile: $tile);

    let builders = [
        // creating init_tile op with static memref
        OpBuilder<(ins "xetile::TileType":$resultType,
            "mlir::Value":$source,
            "llvm::ArrayRef<mlir::OpFoldResult>":$offsets)>,
        // creating init_tile op with dynamic memref or an address
        OpBuilder<(ins "xetile::TileType":$resultType,
            "mlir::Value":$source,
            "llvm::ArrayRef<mlir::OpFoldResult>":$offsets,
            "llvm::ArrayRef<mlir::Value>":$dynamic_shape,
            "llvm::ArrayRef<mlir::Value>":$dynamic_strides)>
    ];

    // let assemblyFormat = [{
    //     $source ``
    //     custom<DynamicIndexList>($offsets, $static_offsets)
    //     (`,` `[` $dynamic_shape^ `]`)?
    //     (`,` `[` $dynamic_strides^ `]`)?
    //     attr-dict `:` qualified(type($source)) `->` qualified(type($tile))
    // }];
    let hasCustomAssemblyFormat = true;

    let extraClassDeclaration = [{
        /// get source type, could be a memref or an integer
        mlir::Type getSourceType() {return getSource().getType();}

        /// check if the source is a memref
        bool isSourceMemRef() {
            return llvm::isa<mlir::MemRefType>(getSourceType());
        }

        /// check if the source is an i64 (i.e. pointer)
        bool isSourceInteger() {
            return llvm::isa<mlir::IntegerType>(getSourceType());
        }

        /// get the element type of the source if it is a memref
        /// this method will fail if the source is  not a memeref
        mlir::Type getSourceMemrefElemType() {
            assert(isSourceMemRef() && "The source is not a memref.");
            return getSourceType().cast<mlir::MemRefType>().getElementType();
        }

        /// The result of an init_tile is always a Tile of TileType.
        TileType getType() {
            return getTile().getType().cast<TileType>();
        }

        /// Return the element type of the tile
        mlir::Type getElementType() {
            return getType().getElementType();
        }

        /// Return the shape of the tile
        llvm::ArrayRef<int64_t> getShape() {
            return getType().getShape();
        }

        /// check if the offsets are static
        bool hasStaticOffsets() {
            return !mlir::ShapedType::isDynamicShape(getStaticOffsets());
        }

        /// check if a given dim in static offsets has a static value
        bool hasStaticOffsetAtDim(int dim) {
            return !mlir::ShapedType::isDynamic(getStaticOffsets()[dim]);
        }

        /// check if the source memref has static shape info
        /// this method will fail if the source is not a memref
        bool sourceMemRefHasStaticShape() {
            assert(isSourceMemRef() && "source is not a memref.");
            return getSourceType().cast<mlir::MemRefType>().hasStaticShape();
        }

        /// get the static shape of the source memref
        /// this method will fail if the source is not a memref or has static shape
        llvm::ArrayRef<int64_t> getSourceMemrefStaticShape() {
            assert(sourceMemRefHasStaticShape() && "The source memref does not have static shape.");
            return getSourceType().cast<mlir::MemRefType>().getShape();
        }

        /// check if dynamic shape arguments are present
        bool hasDynamicShape() {
            return getDynamicShape().size();
        }

        /// check if dynamic stride arguments are present
        bool hasDynamicStrides() {
            return getDynamicStrides().size();
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
        "load_tile" operation loads the values of a tile into a register region with 2D or 4D layout.
        4D layout is used when the tile is in blocked layout.

        If optional "transpose" 2-element array attribute is specified, the loaded tile will be
        transposed along the specified non-zero dimension.

        If optional "padding" value is specified, out-of-bounds memory accesses will be padded with the
        specified padding values. This value defaults to "0.0f".

        This operatio has following arguments:
        * source : source tile that is loaded from
        * transpose : optional 2-element array attibute to specify along which axis the transpose
                        operation must be applied to the input tile
        * padding : optional string attribute to specify the padding value if out-of-bounds
                        memory accesses occurs

        Example 1: loading into a 2D regsiter region
        ```mlir
            %4 = xetile.load_tile %src : !xetile.tile<64x32xf32> -> vector<64x32xf32>
        ```

        Example 2: loading with tranpose and padding attributes enabled.
        ```mlir
            %4 = xetile.load_tile %src { transpose = [1, 0], padding = 1.0 : f32}
                : !xetile.tile<64x32xf32> -> vector<32x64xf32>
        ```

        Example 3: loading into a 4D register region.
        ```mlir
            %4 = xetile.load_tile %src : !xetile.tile<64x32xf32> -> vector<8x2x8x16xf32>
        ```
    }];

    let arguments = (ins
        XeTile: $source,
        OptionalAttr<DenseI64ArrayAttr>: $transpose,
        OptionalAttr<XeTile_PaddingValueAttr>: $padding
    );
    let results = (outs XeTile_2DOr4DVector: $value);

    // let assemblyFormat = [{
    //     $source attr-dict `:` qualified(type($source)) `->` qualified(type($value))
    // }];
    let hasCustomAssemblyFormat = true;

    let extraClassDeclaration = [{
        // padding value defaults to zero in the appropriate type if its not specified
        mlir::Attribute getPaddingValueOrDefault() {
            if (llvm::isa<mlir::IntegerType>(getSource().getType().getElementType())) {
                auto int32Zero = mlir::IntegerAttr::get(mlir::IntegerType::get((*this).getContext(), 32), 0);
                return getPadding().value_or(int32Zero);
            }
            auto float32Zero = mlir::FloatAttr::get(mlir::FloatType::getF32((*this).getContext()), 0.0);
            return getPadding().value_or(float32Zero);
        }
    }];

    let hasVerifier = 1;
}

def XeTile_StoreTileOp : XeTile_Op<"store_tile", []> {
    let summary = "stores a register region into memory";
    let description = [{
        "store_tile" operation can be used to store a register region into a 2D memory region
        decribed by a tile. The register region can be in 2D or 4D. 4D register region is used
        when the stored value is in blocked layout.

        This operation takes the following arguments:
        * value : vector specifying the values to store
        * tile : tile representing the 2D memory region to store into

        Example 1: storing a 2D register region
        ```mlir
            xetile.store_tile %value, %dst : !tile<64x32xf32>,  vector<64x32xf32>
        ```

        Example 1: storing a 4D register region
        ```mlir
            xetile.store_tile %value, %dst : !tile<64x32xf32>,  vector<8x2x8x16xf32>
        ```
    }];

    let arguments = (ins
        XeTile_2DOr4DVector: $value,
        XeTile: $tile
    );

    let assemblyFormat = [{
        $value`,`` `$tile attr-dict `:` qualified(type($value)) `,` qualified(type($tile))
    }];
    // let hasCustomAssemblyFormat = true;
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
        "tile_mma" operation represents matrix multiplication on 2D or 4D vectors. This operation
        takes two input vectors (matrix A, matrix B) and an optional accumulator vector (matrix C) to
        perform a general matrix multiplication.
            C_new = A * B + C
        When vectors A, B and, C are specified in 4D if they are in blocked layout i.e. loaded from
        memory in blocked layout.

        Arguments:
        * a : vector representing input matrix A
        * b : vector representing input matrix B
        * c : optional vector representing accumulator matrix C

        Example 1: tile_mma on 2D vectors of A and B
        ```mlir
            %c_new = xetile.tile_mma %a_vec, %b_vec
                : vector<64x32xf32>, vector<32x128xf32> -> vector<64x128xf32>
        ```

        Example 2: tile_mma on 2D vectors of A, B and, C
        ```mlir
            %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec
                : vector<64x32xf32>, vector<32x128xf32>, vector<64x128xf32> -> vector<64x128xf32>
        ```

        Example 3: tile_mma on 4D vectors of A, B and, C
        ```mlir
            %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec
                : vector<8x4x8x8xf32>, vector<4x8x8x16xf32>, vector<8x8x8x16xf32> -> vector<8x8x8x16xf32>
        ```


    }];

    let arguments = (ins
        XeTile_2DOr4DVector: $a,
        XeTile_2DOr4DVector: $b,
        Optional<XeTile_2DOr4DVector>: $c
    );

    let results = (outs XeTile_2DOr4DVector: $output);

    // let assemblyFormat = [{
    //     $a`,` ` `$b (`,` ` `$c^)? attr-dict `:` `(`qualified(type($a))`,` ` `qualified(type($b))
    //         (`,` ` `qualified(type($c))^)?`)` `->` qualified(type($output))
    // }];

    let hasCustomAssemblyFormat = true;

    let extraClassDeclaration = [{
        mlir::VectorType getAType() { return llvm::cast<mlir::VectorType>(getA().getType()); }

        mlir::VectorType getBType() { return llvm::cast<mlir::VectorType>(getB().getType()); }
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
            xetile.update_tile_offset %tile, [%offset_x, %offset_y]
                : tile<32x32xf32>, index, index
        ```
    }];

    let arguments = (ins
        XeTile: $tile,
        Index: $offset_x,
        Index: $offset_y
    );

    let results = (outs
        XeTile: $result
    );

    let assemblyFormat = [{
        $tile `,` ` ` `[` $offset_x `,` ` ` $offset_y `]` ` ` attr-dict `:`
             qualified(type($tile)) `,`
             qualified(type($offset_x)) `,`
             qualified(type($offset_y)) ` `
            `->` qualified(type($result))
    }];
}


#endif // _XETILE_OPS_TD_INCLUDED_
