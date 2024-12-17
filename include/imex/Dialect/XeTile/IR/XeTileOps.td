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

include "mlir/Dialect/Vector/IR/VectorAttributes.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class XeTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<XeTile_Dialect, mnemonic, traits>;

def XeTile_InitTileOp : XeTile_Op<"init_tile", [Pure, AttrSizedOperandSegments,
                                                OffsetSizeAndStrideOpInterface]> {
    let summary = "Describes an XeTile with reference to a base memref";
    let description = [{
        The "init_tile" operation is used to describe a 2D region (i.e. tile) in gloabl memory.
        This operation takes in a memref or an address and return an xetile. If dynamic-shaped
        memref or an address is used as the base, it is required to specify the shape and strides
        of the memory region described by the tile.

        The operation takes in the following arguments:
        * source:  Source can be static/dynamic shaped memref or an address (i64)
        * offsets: offsets into the "source" memref or address at which to
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

        For the follwing examples, suppose the tile shape used by the compiler is 32x64.

        Example 1:
        Creating a tile using a static shaped 2D memref.

        ```mlir
        %0 = memref.alloc() : memref<1024x1024xf32>
        %c128 = arith.constant 128 : index
        %2 = xetile.init_tile %0[%c128, 512] : memref<1024x1024xf32> -> !xetile.tile<32x64xf32>
        ```

        Example 2:
        Creating a tile using a dynamic shaped 2D memref.

        ```mlir
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        %src = memref.alloc(%c1024, %c512) : memref<?x?xf32>
        %1 = xetile.init_tile %src[256, %c64], [%c1024, %c1024], [%c1024, %c1] : memref<?x?xf32> -> !xetile.tile<32x64xf32>
        ```

        Example 3:
        Creating a tile using an address

        ```mlir
        %src = .... : i64
        ...
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %1 = xetile.init_tile %src[%c128, %c256], [%c1024, %c1024], [%c1024, %c1] : i64 -> !xetile.tile<32x64xf32>
        ```

        Example 4:
        Creating a tile using a static shaped 2D memref with additional attributes asscociated.
        with the tile.

        ```mlir
        %0 = memref.alloc() : memref<1024x1024xf32>
        %c128 = arith.constant 128 : index
        %2 = xetile.init_tile %0[%c128, 512] : memref<1024x1024xf32>
            -> !xetile.tile<32x64xf32, #xetile.tile_attr<order=[0, 1]>>
        ```

    }];

    let arguments = (ins XeTile_BaseAddrType: $source,
                       Variadic<Index>: $offsets,
                       Variadic<Index>: $sizes,
                       Variadic<Index>: $strides,
                       OptionalAttr<DenseI64ArrayAttr>: $const_offsets,
                       OptionalAttr<DenseI64ArrayAttr>: $const_sizes,
                       OptionalAttr<DenseI64ArrayAttr>: $const_strides,
                       Optional<VectorOfRankAndType<[1,2], [Index]>>: $indices);

    let results = (outs XeTile: $tile);


    let assemblyFormat = [{
      $source (`,` $indices^):(``custom<DynamicIndexList>($offsets, $const_offsets))?
      (`,` custom<DynamicIndexList>($sizes, $const_sizes)^
       `,` custom<DynamicIndexList>($strides, $const_strides))?
      attr-dict `:` type($source) (`,` type($indices)^)? `->` qualified(type($tile))
    }];

    let builders = [
        // creating init_tile op with static memref
        OpBuilder<(ins "xetile::TileType":$resultType,
            "mlir::Value":$source,
            "llvm::ArrayRef<mlir::OpFoldResult>":$offsets)>,
        // creating init_tile op with dynamic memref or an address
        OpBuilder<(ins "xetile::TileType":$resultType,
            "mlir::Value":$source,
            "llvm::ArrayRef<mlir::OpFoldResult>":$offsets,
            "llvm::ArrayRef<mlir::OpFoldResult>":$sizes,
            "llvm::ArrayRef<mlir::OpFoldResult>":$strides)>,
        // creating init_tile op for scattered operation
        OpBuilder<(ins "xetile::TileType":$resultType,
            "mlir::TypedValue<mlir::MemRefType>": $source,
            "mlir::TypedValue<mlir::VectorType>": $indices)>
    ];

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
            return mlir::cast<mlir::MemRefType>(getSourceType()).getElementType();
        }

        /// The result of an init_tile is always a Tile of TileType.
        TileType getType() {
            return mlir::cast<TileType>(getTile().getType());
        }

        /// Return the element type of the tile
        mlir::Type getElementType() {
            return getType().getElementType();
        }

        /// Return the shape of the tile
        llvm::ArrayRef<int64_t> getShape() {
            return getType().getShape();
        }

        /// check if the source memref has static shape info
        /// this method will fail if the source is not a memref
        bool sourceMemRefHasStaticShape() {
            assert(isSourceMemRef() && "source is not a memref.");
            return mlir::cast<mlir::MemRefType>(getSourceType()).hasStaticShape();
        }

        /// get the static shape of the source memref
        /// this method will fail if the source is not a memref or has static shape
        llvm::ArrayRef<int64_t> getSourceMemrefStaticShape() {
            assert(sourceMemRefHasStaticShape() && "The source memref does not have static shape.");
            return mlir::cast<mlir::MemRefType>(getSourceType()).getShape();
        }

        /// check if dynamic size arguments are present
        bool hasSizeArgs() {
            auto sizes = getConstSizes().value_or(llvm::ArrayRef<int64_t>({}));
            return sizes.size();
        }

        /// check if dynamic stride arguments are present
        bool hasStrideArgs() {
            auto strides = getConstStrides().value_or(llvm::ArrayRef<int64_t>({}));
            return strides.size();
        }

        /// Get static offsets.
        llvm::ArrayRef<int64_t> getStaticOffsets() {
            if (getConstOffsets().has_value())
                return getConstOffsets().value();
            return llvm::ArrayRef<int64_t>();
        }

        /// Get the static sizes.
        llvm::ArrayRef<int64_t> getStaticSizes() {
            if (getIndices())
                return llvm::ArrayRef<int64_t>();
            if (getConstSizes().has_value())
                return getConstSizes().value();
            // At this point, the source must be a memref with static shape.
            assert(sourceMemRefHasStaticShape() && "The source memref does not have static shape.");
            return getSourceMemrefStaticShape();
        }

        /// Get the static strides.
        llvm::ArrayRef<int64_t> getStaticStrides() {
            if (getIndices())
                return llvm::ArrayRef<int64_t>();
            if (getConstStrides().has_value())
                return getConstStrides().value();
            // At this point, the source must be a memref with static shape.
            assert(sourceMemRefHasStaticShape() &&
                   "The source memref does not have static shape.");
            llvm::SmallVector<int64_t> strides;
            int64_t offset;
            auto memrefType = mlir::dyn_cast<mlir::MemRefType>(getSourceType());
            assert(mlir::succeeded(
                       mlir::getStridesAndOffset(memrefType, strides, offset)) &&
                   "Failed to get strides and offset. Invalid source memref.");
            // Reuse the op storage.
            setConstStrides(strides);
            return getConstStrides().value();
        }

        mlir::Attribute getSourceMemorySpace() {
          if (isSourceMemRef())
            return mlir::cast<mlir::MemRefType>(getSourceType()).getMemorySpace();
          return mlir::Attribute();
        }

        unsigned getSourceMemorySpaceAsInt() {
          auto attr = getSourceMemorySpace();
          if (attr) {
            if (mlir::isa<mlir::IntegerAttr>(attr))
              return static_cast<unsigned>(mlir::cast<mlir::IntegerAttr>(attr).getInt());
          }
          return 0;
        }

        /// Return the expected rank of each of the`static_offsets`,
        /// `static_shape` and `static_strides` attributes.
        std::array<unsigned, 3> getArrayAttrMaxRanks() {
          // for scattered tile, the static_offsets, static_shape and
          // static_strides are not used. Their ranks are expected to be 0.
          if (getIndices())
            return {0, 0, 0};

          unsigned rank;
          if (auto ty = llvm::dyn_cast<mlir::MemRefType>(getSourceType())) {
            rank = ty.getRank();
          } else {
            rank = (unsigned)getMixedOffsets().size();
          }
          return {rank, rank, rank};
        }

        /// Return the number of leading operands before the `offsets`,
        /// `shape` and `strides` operands.
        static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

        mlir::Value getViewSource() { return getSource(); }

    }];

    let hasVerifier = 1;

}

def XeTile_LoadTileOp : XeTile_Op<"load_tile", [
        AllElementTypesMatch<["source", "value"]>,
        AllShapesMatch<["source", "value"]>]> {
    let summary = "Loads a tile into a register region";
    let description = [{
        "load_tile" operation loads the values of a tile into a register region with 2D or 4D layout.
        4D layout is used when the tile is in blocked layout (specified by tile's attributes).

        If optional "padding" value is specified, out-of-bounds memory accesses will be padded with the
        specified padding values. This value defaults to "0.0f".

        This operation has following arguments:
        * source : source tile that is loaded from
        * padding : optional string attribute to specify the padding value if out-of-bounds
                        memory accesses occurs. Padding value defaults to zero.

        Example 1: loading into a 2D regsiter region
        ```mlir
            %4 = xetile.load_tile %src : !xetile.tile<64x32xf32> -> vector<64x32xf32>
        ```

        Example 2: loading with padding attributes enabled.
        ```mlir
            %4 = xetile.load_tile %src { padding = 1.0 : f32}
                : !xetile.tile<64x32xf32> -> vector<32x64xf32>
        ```
    }];

    let arguments = (ins XeTile: $source,
                         OptionalAttr<XeTile_PaddingValueAttr>: $padding,
                         OptionalAttr<XeTile_CacheHintAttr>: $l1_hint,
                         OptionalAttr<XeTile_CacheHintAttr>: $l2_hint,
                         OptionalAttr<XeTile_CacheHintAttr>: $l3_hint);

    let results = (outs XeTile_2DVector: $value);

    let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` type($value)";

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

}

def XeTile_StoreTileOp : XeTile_Op<"store_tile", [
        AllElementTypesMatch<["value", "tile"]>,
        AllShapesMatch<["value", "tile"]>]> {
    let summary = "stores a register region into memory";
    let description = [{
        "store_tile" operation can be used to store a register region into a 2D memory region
        decribed by a tile. The register region can be in 2D or 4D. 4D register region is used
        when the stored value is in blocked layout (specified by tile's attributes).

        This operation takes the following arguments:
        * value : vector specifying the values to store
        * tile : tile representing the 2D memory region to store into

        Example 1: storing a 2D register region
        ```mlir
            xetile.store_tile %value, %dst : vector<64x32xf32>, !tile<64x32xf32>
        ```
    }];

    let arguments = (ins
        XeTile_2DVector: $value,
        XeTile: $tile,
        OptionalAttr<XeTile_CacheHintAttr>: $l1_hint,
        OptionalAttr<XeTile_CacheHintAttr>: $l2_hint,
        OptionalAttr<XeTile_CacheHintAttr>: $l3_hint);

    let assemblyFormat = [{
        $value`,`` `$tile attr-dict `:` qualified(type($value)) `,` qualified(type($tile))
    }];
}

def XeTile_PrefetchTileOp : XeTile_Op<"prefetch_tile", []> {
    let summary = "preftech tiles into cache cooperatively";
    let description = [{
        "preftech_tile" operation can be used to  prefetch contents of a memory region (i.e. tile)
        in to the cache.

        This operation takes following arguments:
        * tile : tile to prefetch into the cache

        Example 1:
        ```mlir
            %1 = memref.alloc() : memref<1024x1024xf16>
            %2 = xetile.init_tile %src[0, 0] : !xetile.tile<128x128xf16>

            xetile.prefetch_tile %2 : !xetile.tile<128x128xf16>
        ```

    }];

    let arguments = (ins XeTile:$tile,
                         OptionalAttr<XeTile_CacheHintAttr>: $l1_hint,
                         OptionalAttr<XeTile_CacheHintAttr>: $l2_hint,
                         OptionalAttr<XeTile_CacheHintAttr>: $l3_hint);

    let assemblyFormat = [{
        $tile attr-dict `:` qualified(type($tile))
    }];
}

def XeTile_TileMMAOp : XeTile_Op<"tile_mma", []> {
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
    }];

    let arguments = (ins
        XeTile_2DVector: $a,
        XeTile_2DVector: $b,
        Optional<XeTile_2DVector>: $c,
        OptionalAttr<XeTile_WorkGroupMapAttr>: $wg_map_a,
        OptionalAttr<XeTile_WorkGroupMapAttr>: $wg_map_b,
        OptionalAttr<XeTile_WorkGroupMapAttr>: $wg_map_c
    );

    let results = (outs XeTile_2DVector: $output);
     let assemblyFormat = [{
        $a `,` $b (`,` $c^)? attr-dict `:` type($a)`,` type($b) (`,` type($c)^)?  `->` type($output)
     }];

    let extraClassDeclaration = [{
        mlir::VectorType getAType() {
            return llvm::cast<mlir::VectorType>(getA().getType());
        }
        mlir::VectorType getBType() {
            return llvm::cast<mlir::VectorType>(getB().getType());
        }
        mlir::Type getElementType() {
            return getA().getType().getElementType();
        }
        mlir::VectorType getOutputType() {
            return getOutput().getType();
        }
    }];

    let hasVerifier = 1;
}

def XeTile_UpdateTileOffsetOp : XeTile_Op<"update_tile_offset", [AttrSizedOperandSegments,
                                                                AllTypesMatch<["tile", "result"]>]> {
    let summary = "update the offsets of a tile";
    let description = [{
        "update_tile_offset" operation is used for iterating over the tiles. It takes in a
        `tile` and offsets `x` and `y`. `x` and `y` are specified relative to the current tile.

        Arguments:
        * tile : original tile
        * offset_x : offset along x dimension relative to the current tile to reach the next tile
        * offset_y : offset along y dimension relative to the current tile to reach the next tile

        Example 1:
        ```mlir
            xetile.update_tile_offset %tile, [%offset_x, %offset_y] : tile<32x32xf32>
        ```
    }];

    let arguments = (ins
        XeTile: $tile,
        Optional<Index>: $offset_x,
        Optional<Index>: $offset_y,
        Optional<FixedVectorOfRankAndType<[1,2], [Index]>>:$indices);

    let results = (outs
        XeTile: $result
    );

    let assemblyFormat = [{
        $tile`,`` `(``$indices^):(`[` $offset_x `,` $offset_y `]` )? attr-dict `:` qualified(type($tile)) (`,` type($indices)^)?
    }];
}

def XeTile_AtomicRMWOp : XeTile_Op<"atomic_rmw", []> {
    let summary = "performs  a read modify write operation that is free from data races.";
    let description = [{
        This operation performs a atomic read-modify-write on the memroy location
        specified by the tile. Operation reuses the arith dialect attribute
         mlir::arith::AtomicRMWKindAttr to specify the atomic operation kind.

        Example 1:
        ```mlir
            %ret_value = xetile.atomic_rmw “addf” %value, %tile:
                vector<8x16xbf16>, tile<8x16xbf16> -> vector<8x16xbf16>
        ```
    }];
    let arguments = (ins XeTile_AtomicRMWKindAttr:$kind,
                        XeTile_2DVector:$value,
                        XeTile:$tile);
    let results = (outs XeTile_2DVector:$result);
    let assemblyFormat = [{
        $kind $value `,` $tile  attr-dict `:` qualified(type($value)) `,` qualified(type($tile))
                `->` qualified(type($result))
    }];
}

def XeTile_TransposeOp: XeTile_Op<"transpose", []> {
    let summary = "transpose a 2D vector.";
    let description = [{
        It has the same semantic with `vector.transpose`, but limits the vector to be 2D.
    }];

    let arguments = (ins XeTile_2DVector: $vector,
                         DenseI64ArrayAttr:$permutation);
    let results = (outs XeTile_2DVector: $result);
    let assemblyFormat = [{
        $vector `,` $permutation attr-dict `:` type($vector) `->` type($result)
    }];
    let hasVerifier = 1;
}

def XeTile_ReductionOp: XeTile_Op<"reduction", []> {
    let summary = "performs a reduction operation over a 2D vector.";
    let description = [{
        It has the same semantics as the `vector.multi_reduction`,
        but restricts the vector dimension to 2D, and also the result
        is 2D too, with the reduced axis being 1.
    }];

    let arguments = (ins Vector_CombiningKindAttr: $kind,
                         XeTile_2DVector: $source,
                         DenseI64ArrayAttr: $reduction_dims);
    let results = (outs XeTile_2DVector: $result);
    let assemblyFormat = [{
        $kind `,` $source $reduction_dims attr-dict `:` type($source) `->` type($result)
    }];

    let hasVerifier = 1;
}

def XeTile_BroadcastOp: XeTile_Op<"broadcast", []> {
    let summary = "broadcast a vector from 1D to 2D.";

    let arguments = (ins XeTile_2DVector: $source,
                         DenseI64ArrayAttr: $broadcast_dim);
    let results = (outs XeTile_2DVector: $result);
    let assemblyFormat = [{
        $source $broadcast_dim attr-dict `:` type($source) `->` type($result)
    }];
    let hasVerifier = 1;
}

def XeTile_ConvertLayoutOp: XeTile_Op<"convert_layout", [AllTypesMatch<["source", "result"]>]> {
    let summary = "Convert the sg layout of the input operand";
    let description = [{
        convert_layout with wg_map attributes remaps the SG layout
        into a new layout which shuffles the data between subgroups with a workgroup
    }];
    let arguments = (ins XeTile_2DVector: $source,
                         XeTile_WorkGroupMapAttr: $wg_map_result,
                         OptionalAttr<XeTile_WorkGroupMapAttr>: $wg_map_source
                         );
    let results = (outs XeTile_2DVector: $result);
    let assemblyFormat = [{
        $source attr-dict `:` type($source)
    }];
}

def XeTile_LoadGatherOp: XeTile_Op<"load", [AllElementTypesMatch<["tile", "value"]>,
                                            AllShapesMatch<["value", "mask"]>]> {
   let summary = "load a set of scattered data points from memory.";
   let description = [{
        The `load` operation is used to load data with scattered tile (each element in the tile
        is interpreted as location of the data). the `mask` operand masks out memory access so
        that it is safe to pass out-of-boundary addresses/offsets as long as they are masked.
        In this case, the value specified in the padding attribute will be returned. The default
        padding value is zero.}];

   let arguments = (ins XeTile: $tile,
                        XeTile_MaskType: $mask,
                        OptionalAttr<XeTile_PaddingValueAttr>: $padding,
                        OptionalAttr<XeTile_CacheHintAttr>: $l1_hint,
                        OptionalAttr<XeTile_CacheHintAttr>: $l2_hint,
                        OptionalAttr<XeTile_CacheHintAttr>: $l3_hint);
   let results = (outs XeTile_1DOr2DVector: $value);
   let assemblyFormat = [{
        $tile `` `,` $mask attr-dict `:` qualified(type($tile)) `` `,` type($mask) `->` type($value)
   }];
}

def XeTile_StoreScatterOp: XeTile_Op<"store", [AllElementTypesMatch<["value", "tile"]>,
                                               AllShapesMatch<["value", "mask"]>]> {
   let summary = "load a set of data to scattered memory locations.";
   let description = [{
        The `store` operation is used to store data into scattered tile (each element in the tile
        is interpreted as location, one location per data element). the `mask` operand masks out
        memory access so that it is safe to pass out-of-boundary addresses/offsets as long as they
        are masked.
   }];
   let arguments = (ins XeTile_1DOr2DVector: $value,
                        XeTile: $tile,
                        XeTile_MaskType: $mask,
                        OptionalAttr<XeTile_CacheHintAttr>: $l1_hint,
                        OptionalAttr<XeTile_CacheHintAttr>: $l2_hint,
                        OptionalAttr<XeTile_CacheHintAttr>: $l3_hint);
   let assemblyFormat = [{
    $value `,` $tile `,` $mask attr-dict `:` type($value) `,` qualified(type($tile)) `,` type($mask)
   }];
}



#endif // _XETILE_OPS_TD_INCLUDED_
