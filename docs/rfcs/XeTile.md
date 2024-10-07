# RFC for XeTile Dialect

## Summary
This RFC proposes XeTile Dialect to support efficient code generation for GEMM operation.

## Motivation
Lowering GEMM (General matrix multiplication) to an efficient tiled loop structure that can be mapped to GPU compute hierarchy is a complicated task, with multiple factors determining the efficiency. After decomposing the task into workgroup kernels and further down to subgroup kernels, each subgroup kernel executes a GEMM operation on submatrices. Generating efficient code for GEMM requires a good decomposition that creates enough subtasks to drive high core utilization and large enough subgroup-level submatrix size for code efficiency. Besides the parallel decomposition, each hardware has its recipe for the best code sequence for GEMM, which includes high-level algorithms, like cooperative prefetch/load, k-slicing and software pipelining, and target-specific code sequences for submatrix operations.

To facilitate efficient code generation for GEMM, we introduce XeTile dialect. XeTile dialect supports the tile-based programming model and decomposes the GEMM kernel to large pre-defined tile sizes at the subgroup and workgroup level. With the XeTile dialect, the high-level GEMM algorithm can be easily expressed. Underneath XeTile, the implementation uses target-specific recipes and HW features to get the best performance on specific hardware. Based on XeTile representation, as the GEMM is decomposed at submatrix granularity and mapped to registers, it naturally supports optimization like fusing with neighbor operations.

The XeTile dialect provides microkernel-level functionality to build high-performance GEMM using a pure MLIR lowering pipeline. It supports matrix operations on tile sizes larger than hardware matrix tiles so that the lowering pass optimizes a larger scope of multiple matrix instructions with target-specific recipes that give the best performance. For example, it uses the most efficient 2D block loader instead of a general but inefficient load instruction whenever the 2D block fits HW requirements. Xe GPU’s 2D block load loads a large chunk of data and autopad the out-of-boundary access, when the matrix address and sizes meet HW requirements. As XeTile abstracts out the HW difference, the same XeTile-based code works on any type of GPUs.

Based on XeTile, users can implement different GEMM algorithms. Based on the input GEMM shapes and microarchitecture info, the GEMM lowering chooses one high-level algorithm, decides decomposition parameters, and generates XeTile code.

## Proposal
### XeTile Dialect

XeTile provides a middle-level abstraction for matmul operation and sits between Linalg matmul named op and XeGPU Dpas op. The implementation starts from Xe GPU, but it is not tied to specific Xe architecture. The XeTile dialect design facilitates optimization using hardware auto-padding, which generates simpler and more efficient code than the software padding. Using the tile dialect, the user doesn’t need to detect the out-of-boundary case, and the dialect takes care of unaligned shapes, so the same code runs for the unaligned use case. Users can focus on high-level optimization like software pipelining, cooperative prefetch, and K-slicing.

| Ops	| Syntax	| Example |
| :---   | :----   | :--- |

|init_tile	| operation ::= xetile.init_tile $base_memref $offset0, $offset1: type($base_memref), index, index, attr-dict-> type($tile, attr-dict)	| %block = xetile.init_tile %base_memref, %tile_offset:2 memref<128x128xbf16> into tile<8x16xbf16> |
|load_tile	| operation ::=xetile.load_tile $tile attr-dict:type($tile) ->type($res)	 | %vector_a = xetile.load_tile %tile_a {padding=0} : tile<64x32xbf16> into vector<32x64xbf16>|
|store_tile	| operation ::=xetile.store_tile $value, $tile attr-dict: type($value), type($tile) | xetile.store_tile %tile_a, %vector_a: vector<64x64xbf16> into tile<64x64xbf16> |
|update_tile_offset	| operation ::=xetile.update_tile_offset $tile, $delta0, $delta1: type($tile), index, index-> type($tile)	| %tdesc_updated = xetile.update_nd_offset %tdesc, %offset_x, offset_y tensor_desc<32x64xbf16>, index, index -> tensor_desc<32x64xbf16> |
|prefetch_tile	| operation ::=xetile.prefetch_tile $tile, attr-dict: type($tile)	  | xetile.prefetch_tile %coop_tile: tile<16x32xbf16> |
|tile_mma	| operation ::=xetile.tile_mma $matA, $matB, $matC attr_dict: type($matC), type($matA), type($matB)-> type($res)	 | %vector_c = xetile.tile_mma %vector_a, %vector_b, %vector_c : vector<64x32xbf16>, vector<32x128xbf16>, vector<64x128xfloat> into vector<64x128xfloat>  |
|atomic_rmw_tile| operation ::=xetile.atomic_rmw_tile \<$kind\>, $vec, $tile: type($vec), type($tile) -> type($res)	 | %vector_a = xetile.atomic_rmw_tile \<add\> %value, %tile: vector<8x16xbf16>, tile<8x16xbf16> to vector<8x16xbf16>  |
|tile_transpose	| operation ::=xetile.tile_transpose $vec $permuation_dims attr_dict: type($vec) -> type($res)	 | %vector_a = xetile.tile_transpose %vector_b [1, 0]: vector<64x32xfloat> into vector<32x64xfloat>  |
|tile_reduce	| operation ::=xetile.tile_reduce \<$kind\> $src  $reduction_dims attr_dict: type($value) -> type($res)	 | %vector_a = xetile.tile_reduce \<add\> %vector_b [1]: vector<64x32xfloat> into vector<64x1xfloat>  |
|tile_broadcast	| operation ::=xetile.tile_broadcast $src $broadcast_dims attr_dict: type($value) -> type($res)	 | %vector_a = xetile.tile_broadcast %vector_b[0]: vector<1x32xfloat> into vector<64x32xfloat>  |
|tile_pack*	| operation ::=xetile.tile_pack $matA attr_dict: type($value) -> type($res)	 | %vector_a = xetile.tile_pack %vector_b {inner_blocks=array<i64: 16, 16>} : vector<64x32xfloat> into vector<4x2x16x16xfloat>  |
|tile_unpack*	| operation ::=xetile.tile_upack $matA attr_dict: type($value) -> type($res)	 | %vector_a = xetile.tile_unpack %vector_b {inner_blocks=array<i64: 16, 16>} : vector<1x2x64x16xfloat> into vector<64x32xbf16> |

*Operations only used to support internal lowering.

**OP name convention:  init_tile, load_tile, prefetch_tile, store_tile, and update_offset operates on the tile type and involves memory access. tile_xxx operates on vector data type only.

To create a 2D Tile memory descriptor, the user needs to set up a tile (init_tile) describing a 2D region within the global memory. Setting up a tile requires the shape of the parent tile and the underneath physical memory buffer size, known as the base matrix. The base matrix must be 2D and must be contiguous. The XeTile takes the base matrix address pointer, shape, and strides, and the tile’s offsets and shape. Offsets, strides, and shapes are for two dimensions and in the number of elements. base_stride[0] describes the number of elements between the two rows, describing the width of the underneath physical memory buffer, and *%base_strides[1] must be 1, as the innermost dimension of the base matrix must be contiguous. The current version only supports 2D memref with a row-major layout.

`init_tile` takes memref as the description of the base matrix with the offsets of the specific tile. The tile shape and element data type are specified in the output tile data type, and they must be known at compile-time.

`init_tile` with memref of static shape. Tile uses memref’s shape and strides as base_shape and base_strides.
```mlir
  %tile0 = xetile.init_tile %base_memref, [%tile_offset:2] :
     memref<128x128xbf16> into tile<8x16xbf16>
```
`init_tile` with memref of dynamic shape. The memref has a dynamic shape, so that its shape and strides have to be passed as runtime parameters to init_tile.
```mlir
  %tile0 = xetile.init_tile %base_memref, [%tile_offset:2], [%base_shape:2], [%base_strides:2]:
     memref<?x?xbf16> into tile<8x16xbf16>
```
 `init_tile` with an address for the base matrix. This form is to support the use case which doesn’t use a memref to describe the base matrix.
```mlir
  %tile0 = xetile.init_tile %base_addr, [%tile_offset:2], [%base_shape:2], [%base_strides:2]:
     i64 into tile<8x16xbf16>
```

`init_tile` with an `order` to access the base matrix. The `order` attribute describes the order of the tile elements stored in the memory. "0" indicates the fastest-changing dimension. The order must be consistent with the data layout specified by the memref represening the base matirx. If the base matrix is stored as row-major, the order is specified as [1, 0]. If the base matrix is stored as column-major, the order is specified as [0, 1]. The default is row-major. The output tile carries the `order` attribute in its attribute set.

```mlir
  #tile_attr = #xetile.tile_attr<order = [0, 1]>
  %tile0 = xetile.init_tile %base_memref, [%tile_offset:2]:
     memref<128x128xbf16, affine_map=<(d0, d1)->(d1, d0)> into tile<64x32xbf16, #tile_attr>
```

With the tile date type, XeTile supports load_tile, prefetch_tile, and store_tile.

`load_tile` loads a tile to a 2D vector, which could be backed by a register region.
```mlir
  %vector_a = xetile.load_tile %tile_a :
     tile<64x64xbf16> into vector<64x64xb16>
```

Attribute `padding` specifies the padding value for the out-of-boundary access. The default value is zero.
```mlir
  %vector_a = xetile.load_tile %tile_a {padding = 1.0} :
     tile<64x64xbf16> into vector<64x64xb16>
```
`load_tile` needs to be used with the tile_mma.

`load_tile` loads a tile according to the tile's `order` attribute. Regardless of the `order` attribute value, the vector's dimensions must match exactly the Tile's dimensions.
```mlir
  #tile_attr = #xetile.tile_attr<order = [0, 1]>
  %vector_a = xetile.load_tile %tile_a :
     tile<64x32xbf16, #tile_attr> into vector<64x32xb16>
```

`store_tile` stores a vector to memory. Padding attributes are not supported.
```mlir
  xetile.store_tile %tile_a, %vector_a :
   vector<64x64xbf16> into tile<64x64xbf16>
```
`store_tile` stores a tile according to the tile's `order` attribute. Regardless of the `order` attribute value, the vector's dimensions must match exactly the Tile's dimensions.
```mlir
  #tile_attr = #xetile.tile_attr<order = [0, 1]>
  %vector_a = xetile.store_tile %tile_a :
     vector<64x32xb16> to tile<64x32xbf16, #tile_attr>
```

`prefetch_tile` prefetches the tile to cache. Just like memref.preftech, the locality hint ranges from locality<0> (no locality) to locality<3> (extremely local keep in cache).
```mlir
  xetile.prefetch_tile %coop_tile locality<3>: tile<8x32xbf16>
```
`tile_mma` represents the matrix multiplication on 2D vectors. The semantics can be represented by vector.contract, so tile_mma works more like a syntax sugar. This also means that the code can be lowered to vector.contract and mapped to HW without DPAS support nicely.
```mlir
  %vector_c = xetile.tile_mma %vector_b, %vector_a, %vector_c:
     vector<64x32xbf16>, vector<32x128xbf16>, vector<64x128xfloat>
	   into vector<64x128xfloat>
```
A `tile_mma` variant without vector_c initialization.
```mlir
  %vector_c = xetile.tile_mma %vector_a, %vector_b :
     vector<64x32xbf16>, vector<32x128xbf16>
	   into vector<64x128xfloat>
```
`update_tile_offset` updates tile with offset_x and offset_y, to move the current tile to a new position. These offsets are relative offset to the current position and counted in the number of elements.  Usually only one value is needed to update since the tile is only moving along the K dimension. Users should avoid initializing new tiles repeatedly. For best performance, the user should only initialize one tile as a base tile and update the tile offset to move to a new tile.
```mlir
  %tile_updated = xetile.update_tile_offset %tile, %offset_x, offset_y :
		tile<64x64xbf16>, index, index into tile <64x64xbf16>
```


`atomic_rmw_tile` atomically reads, modifies, and writes back data to the memory specified by the tile.

```mlir
  %ret_value = xetile.atomic_rmw <addf> %value, %tile:
          vector<8x16xbf16>, tile<8x16xbf16> to vector<8x16xbf16>
```
xetile.atomic_rmw reuses the arith dialect attribute, mlir::arith::AtomicRMWKindAttr.


`tile_transpose` transpose a 2D vector. It has the same semantics as the vector.transpose, but restricts the vector dimension to 2D.
```mlir
   %vector_a = xetile.tile_transpose [1, 0] %vector_b: vector<64x32xfloat> into vector<32x64xfloat>
```
`tile_reduce` performs a reduction operation over a 2D vector. The result is a 2D vector with the size of reduced axis being 1. It has the same semantics as the vector.multi_dimesnion, but restricts the vector dimension to 2D. The reduce operation are the same as vector.multi_dimension:add/mul/minsi/minui/maxsi/maxui /and/or/xor for integers, and add/mul/minnumf/maxnumf/minimumf /maximumf for floats.
```mlir
   %vector_a = xetile.tile_reduce <add> %vector_b [1]: vector<64x32xfloat> into vector<64x1xfloat>
```
`tile_broadcast` broadcast from 1D vector to a 2D vector.
```mlir
   %vector_a = xetile.tile_broadcast %vector_b [0]: vector<1x32xfloat> into vector<64x32xfloat>
```

## Internal Operations to support gradual lowering
The 2D XeTile IR needs to be lowered in an intermediate form to support `blocking` optimization. The `blocking` optimization loads the tile in blocks and feed the block to matrix hardware. Since the load block size and matrix hardware size are not necessary same, we need to represent the data block in some form to assist the optimization. Conceptually, when a 2D tile data being loaded with a specified block size, the vector represents the 2D tile in 4D block layout. So we uses 4D dimension vector to describe the data being loaded with the block size.

`init_tile` with an `inner_block` for 2D block access of the base matrix. The `inner_blocks` attribute describes the block size for each memory load and store operation when the tile is being loaded. The block size for load may be larger than the block size for MMA operation. The output tile carries the `inner_block` attribute in its attribute set.

```mlir
  #tile_attr = #xetile.tile_attr<inner_blocks=[16,16]>
  %tile0 = xetile.init_tile %base_memref, [%tile_offset:2]:
     memref<128x128xbf16> into tile<64x32xbf16, #tile_attr>
```

`load_tile` loads a 2D tile with an `inner_block` attribute  to 4D vector.
```mlir
  #tile_attr = #xetile.tile_attr<inner_blocks=[16,16]>
  %vector_a = xetile.load_tile %tile_a :
     tile<64x32xbf16, #tile_attr> into vector<4x2x16x16xb16>
```
`store_tile` stores a 4D vector to a 2D tile with an `inner_block`.
```mlir
  #tile_attr = #xetile.tile_attr<inner_blocks=[16,16]>
  xetile.store_tile %vector_a, %tile_a :
     vector<4x2x16x16xb16> into tile<64x32xbf16, #tile_attr>
```
`atomic_rmw_tile` performs atomic operation on 4D vectors.
```mlir
#tile_attr = #xetile.tile_attr<inner_blocks=[8,16]>
%vector_a = atomic_rmw_tile <add> %value, %tile: vector<8x48x16xbf16>, tile<64x64xbf16, #tile_attr> to vector<8x4x8x16xbf16>
```

With the data being presented as 4D vector, all the vector based XeTile operations are required to support blocking.
`tile_mma` works on 4D vectors. Since dimension 1 is split into dimensions 1 and 3, the reduction of matrix multiplication is along these two dimensions.
```mlir
   %vector_c = xetile.tile_mma %vector_a, %vector_b, %vector_c :
     vector<8x4x8x8xbf16>, vector<4x8x8x16xbf16>, vector<8x8x8x16xfloat>
	   into vector<8x8x8x16xfloat>
```
`tile_reduce` follows the vector.multi-reduction semantics and can be applied to 4D vector. The tile_reduce on 4D vector is an internal operation and only used in the transformation passes to support gradual lowering.
```mlir
   %vector_a = xetile.tile_reduce <add> %vector_b [1, 3]: vector<8x4x8x16xfloat> into vector<8x1x8x1float>
```

`tile_broadcast` broadcast 4D vector. The input is expected to be first reshaped from 1D vector to 2D vector, and then blocked to 4D.
```mlir
   %vector_a = xetile.tile_broadcast %vector_b [1, 3]: vector<8x1x8x1xfloat> into vector<8x4x8x16xfloat>
```

`tile_transpose` doesn't have support 4D vector. The transpose is usually implemented by saving and restoring from the share local memory. To support this, we relax the restriction of tile_load and tile_store so that they can load 2D from share local memory.

`tile_pack` and `tile_unpack` are introduced to support the gradual lowering. It allows the XeTile IR to be blocked with different block size, and then try to find a good blocking strategy with minimum tile_pack and tile_unpack overhead.

`tile_pack` packs a 2D vector, representing the loaded value from 2D tile, to a 4D vector with an inner block size. The 4D vector was introduced to support blocking to fit the hardware matrix operation sizes.  The blocking follows an implicit rule: out_dim[0] = in_dim[0]/inner_blocks[0] , out_dim[1] = in_dim[1]/inner_blocks[1], out_dim[2] = inner_blocks[0], and out_dim[3] = inner_blocks[1]. The dim[2] and dim[3] of result 4D vector must be same as the size of `inner_blocks` attribute.

```mlir
  %0 = xetile.tile_pack %1 {inner_blocks = array<i64: 16, 16>}
    : vector<64x32xf32> -> vector<4x2x16x16xf32>
```
`tile_unpack` unpacks a 4D blocked vector back to original unpacked 2D vector.
`tile_unpack`
```mlir
  %0 = xetile.tile_unpack %1 {inner_blocks = array<i64: 64, 16>}
    : vector<1x2x64x16xf32> -> vector<64x32xf32>
```
The tile_pack and tile_unpack operation is similar to pack and unpack operation of tensor dialect. The source vector must be a 2D dimension vector, and no permutation is allowed for the result 4D vector, so effectively the blocking effect is identical to tensor pack/unpack operation with inner_dims_pos = [0,1] inner_dims_pos = [0, 1].

## support for load_gather and store_scatter (experimental)
`init_tile` can create a tile with each element's address being explictly specified. The tile is created with a base address and offsets for all elements to be loaded. The result tile has a `scatter` attribute to distinguish it from the regular tile.
```mlir
  %tile0 = xetile.init_tile %base_addr, %tile_offsets:
     i64, vector<1x256xindex> into tile<1x256xbf16, #scatter>
```
`load_gather` (aka. load) loads data with prepared tile and mask. Attribute `padding` specifies the padding value for the out-of-boundary access. The default value is zero.
```mlir
  %vector_a = xetile.load_gather %tile_0, %mask, {padding = 1.0} :
     tile<1x256xbf16, #scatter> into vector<1x256xbf16>
```
`store_scatter` stores a 2d vector to a 2D tile with `scatter` attribute.
```mlir
  xetile.store_scatter %vector_a, %mask, %tile_0 :
     vector<1x256xbf16> into tile<1x256xbf16, #scatter>
```

## Workgroup Level XeTile extension (experimental)
`xetile.wg_map` mapping attribute allows XeTile operation to work at the workgroup level. XeTile operations work by default at the subgroup level without wg_map attribute. With wg_map attributes, XeTile operations can be applied to workgroup-level tile sizes. The attribute `xetile.wg_map` guides the lowering from the workgroup level to the subgroup level by specifying how the data is distributed across parallel subgroups. It gives the user full control over the lowering process so that the user can tune the block size for both the workgroup and subgroup for optimal performance.

Below is an example.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
   #tile_attr = #xetile.tile_attr<wg = #wg_map_a, order = [0, 1], inner_blocks = array<i64: 32,16> >

   %wg_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #tile_attr>
```
Within the `xetile.wg_map`, `sg_layout` specifies the subgroup layout, and `sg_data` specifies the tile size owned by each subgroup. The tile created by init_tile is a workgroup-level tile. In the example above, sg_layout [2,2] means that each workgroup has 4 subgroups with 2 rows and 2 columns. When mapping sg_layout to linear subgroup id, sg_layout is always mapped to subgroup id in row-major ordering. sg_data [32,128] means that each subgroup works on a submatrix [32, 128]. The data elements assigned to each subgroup thread must be contiguous.

For each dimension, the size of `sg_layout` multiplying `sg_data` must be divisible by the wg_tile size or vice versa. The wg_tile is distributed to sg_data x sg_layout in a round-robin fashion. If sg_data[i] x sg_layout[i] < wg_tile[i], we have data left after all subgroups are assigned for the first round. In this case, we continue to assign the rest data starting from the first subgroup until the data is completely assigned. If sg_data[i] x sg_layout[i] >= wg_tile[i], we may have already used up all the data before all subgroups are assigned. In this case, we wrap around the wg_tile and continue the assignment, and the rest subgroups along that dimension share the same data.

For example, for the tile size [128, 128] and sg_data [32, 128], along the second dimension, there is no more data left to assign after the first subgroup, it wraps around and moves to the beginning of the tile and continues the assignment. Instead, for the first dimension, there is more data left after the first round of distribution, so it move to the next subtile and continue the assignement. As a result, the tile would be sliced to four subtiles with size [32,128], with the following mapping for sg_layout [2,2]:

| subtiles	| 2D subgroup id	|  Linearized subgroup id
| :---   | :----   | :----   |
| [  0:31, 0:127] | [0, 0] , [0, 1] | 0 , 1 |
| [ 32:63, 0:127] | [1, 0] , [1, 1] | 2 , 3 |
| [ 64:95, 0:127] | [0, 0] , [0, 1] | 0 , 1 |
| [96:127, 0:127] | [1, 0] , [1, 1] | 2 , 3 |

With the `xetile.wg_map` attribute being included in the tile data type, the tile memory related operations (xxx_tile) can be distributed to subgroup. The vector based operations (tile_xxx) requires extra handling, since we can't attatch the the `xetile.wg_map` attribute to MLIR vector data type.

The proposal is to attach the `xetile.wg_map` attribute to the vector based XeTile operations as illustrated below. The attribute applies only to the output value of each operation. The input values `xetile.wg_map` are determined by their respective defining operations.
| Ops	| Syntax	| Example |
| :---   | :----   | :--- |

|tile_mma	| operation ::= xetile.tile_mma $matA, $matB, $matC attr_dict: type($matA), type($matB), type($matC)-> type($res)	 | %vector_c = xetile.tile_mma %vector_a, %vector_b, %vector_c {#mp_c} : vector<64x32xbf16>, vector<32x128xbf16>, vector<64x128xfloat> into vector<64x128xfloat>  |
|tile_transpose	| operation ::= xetile.tile_transpose $permuation_dims attr_dict $vec : type($vec) -> type($res)	 | %vector_a = xetile.tile_transpose %vector_b {#mp_a}: vector<64x32xfloat> into vector<32x64xfloat>  |
|tile_reduce	| operation ::= xetile.tile_reduce $kind $src $reduction_dims attr_dict: type($value) -> type($res)	 | %vector_a = xetile.tile_reduce <add> %vector_b [1] {#mp_a}: vector<64x32xfloat> into vector<64x1xfloat>  |
|tile_broadcast	| operation ::= xetile.tile_broadcast $src $broadcast_dims attr_dict : type($value) -> type($res)	 | %vector_a = xetile.tile_broadcast %vector_b  [0] {#mp_a}: vector<1x32xfloat> into vector<64x32xfloat>  |
|tile_conv_layout	| operation ::= xetile.conv_layout $src attr_dict: type($value) -> type($res)	 | %vector_a = xetile.tile_conv_layout %vector_b {#mp_a} : vector<256x256xfloat> into vector<256x256xfloat>  |

With the `wg_map` attribute attached for the output vector, `tile_mma` does a matrix multiplication at a work group level vector.
```mlir
   #wg_map_d = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>

   %vector_d = xetile.tile_mma %vector_a, %vector_b, %vector_c {#wg_map_d}:
     vector<256x256xfloat>, vector<256x32xbf16>, vector<32x256xbf16>
	   into vector<256x256xfloat>
```
The `wg_map` attribute of input vector operands can be derived from the wg_map_d. They must have the same sg_layout, and sg_data for m and n dimenion must be same as wg_map_d, and sg_data for k dimension must be same as operand A and B. These attributes may be retrieved from their producer ops, and the retrieved attributes must be consistent with the derived ones. Below is the derived wg_map for the three vector operands in the example above.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]> //wg_map for %vector_a
   #wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]> //wg_map for %vector_b
   #wg_map_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]> //wg_map for %vector_c
```

`tile_reduce` with `wg_map` does the reduction over a workgroup level vector.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 1]>
   %vector_a = xetile.tile_reduce <add> %vector_b [1] {#wg_map_a}: vector<256x128xfloat> into vector<256x1xfloat>
```
The `wg_map` attribute of the input vector can be derived from the wg_map_a. sg_layout must be same, sg_data for the dimension being reduced must be same as the input vector, and the other dimension must be same as the wg_map_a. The input vector's wg_map attribute may be retrieved from its producer op, and the retrieved attribute must be consistent with the derived one. Below is the derived wg_map for the input vector in the example above.
```mlir
   #wg_map_b = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 128]>  //wg_map for %vector_b
```

`tile_broadcast` with `wg_map` attribute broadcast at workgroup level.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [16, 1], sg_data = [16, 256]>
   %vector_a = xetile.tile_broadcast %vector_b [1] {#wg_map_a}: vector<256x1xfloat> into vector<256x256xfloat>
```
The `wg_map` attribute of the input vector can be derived from the wg_map_a. sg_layout must be same, sg_data for the dimension being broadcast must be "1", and the other dimension must be same as the wg_map_a. The input vector's wg_map attribute may be retrieved from its producer op, and the retrieved attribute must be consistent with the derived one. Below is the derived wg_map for the input vector in the example above.
```mlir
   #wg_map_b = #xetile.wg_map<sg_layout = [16, 1], sg_data = [16, 1]>  //wg_map for %vector_b
```

`tile_transpose` with `wg_map` attribute transpose a workgroup level vector.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 64]>
   %vector_a = xetile.tile_transpose %vector_b {#wg_map_a}: vector<512x128xfloat> into vector<128x512xfloat>
```

The `wg_map` attribute of the input vector can be derived from the wg_map_a. The two dimension of sg_layout and sg_data must be swapped. The input vector's wg_map attribute may be retrieved from its producer op, and the retrieved attribute must be consistent with the derived one. Below is the derived wg_map for the input vector in the example above.
```mlir
   #wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 32]>  //wg_map for %vector_b
```
The tile_transpose can be implemented by saving and restoring from the shared local memory. It can be conceptually viewed as a composition of two operations: 1) store the vector to to shared memory with the #wg_map_b mapping assuming row_major and 2) use wg_map_a mapping to load the data from shared memory to vector assuming column_major. To support this, we relax the restriction of tile_load and tile_store so that they can load 2D from share local memory.

An optimization is to analyze the load op which produces %vector_b, carefully arrange its mapping so that each subgroup thread loads its corresponding subgroup tile, and then either combine transpose function to the load op or do an in-register transpose.

`tile_conv_layout` with `wg_map` attributes remaps the workgroup level vector to subgroup threads. The second `wg_map` attribute is optional and describes the input operand. The input vector's wg_map attribute may be retrieved from its producer op, and the retrieved attribute must be consistent with the second `wg_map` attribute if it is present.

Example with the wg_map specified for both input and output operands.
```mlir
   #wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>  // used for cooperative load/prefetch
   #wg_map_a = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]> // used as mma's input matrix A
   %vector_a = xetile.tile_conv_layout %vector_b {#wg_map_a #wg_map_b}: vector<256x256xfloat> into vector<256x256xfloat>
```
Example without the wg_map specified for the input operand.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]> // used as mma's input matrix A
   %vector_a = xetile.tile_conv_layout %vector_b {#wg_map_a}: vector<256x256xfloat> into vector<256x256xfloat>
```
The tile_conv_layout could be implemented by saving and restoring from the shared local memory. It can be conceptually viewed as a composition of two operations: 1) store the vector to to shared memory with the #wg_map_b mapping assuming row_major and 2) use wg_map_a mapping to load the data from shared memory to vector assuming same row_major. To support this, we relax the restriction of tile_load and tile_store so that they can load 2D from share local memory.


## Alternative design considerations

The alternative design of tile data type is to reuse the memref data type. The memref data type needs to be enhanced to allow attributes. So the XeTile's tile data type can be expressed with memref associated with Tile attributes. xetile.wg_map and xetile.sg_map are examples of these attributes.

## Appendix 1 - use case for xetile.order attribute and tile_transpose

xetile.tile describes a 2D block in memory . The default layout of xetile.tile is raw-major contiguous.  So tile[i][j] refers to the position i*stride_i + j in the associated memory. The stride_j must be 1 since it is contiguous. This maps well the underlying  2d block loader, which loads data in raw-major layout only and no stride in innermost dimension.
Below is the example code for the most common use case of xetile.tile.
```mlir
   BF16 A[M][K], B[K][N], C[M][N];   // C = MM(A, B)
   For i = 0, M-1, M_tile  Do
    For j = 0, N-1, N_tile Do
        For k = 0, K-1, K_tile  Do
            %a = init_tile &A, [i, k], [M, K], [K, 1] : tile<64x32xbf16>;  // M_tile=64, K_tile=32
            %b = init_tile &B, [k, j], [K, N], [N, 1] : tile<32x64xbf16>;  // K_tile=32, N_tile=64
            %c = init_tile &C, [i, j], [M, N], [N, 1] : tile<64x64xbf16>;  // M_tile=64, N_tile=64
             %va = load_tile %a : vector<64x32xbf16>;
             %vb = load_tile %b : vector<32x64x bf16>;
             %vc = tile_mma %va, %vb : vector<64x32xbf16>, vector<32x64x bf16> into vector<64x64xbf16>;
```
The order attribute was introduced to support a  second use case where  the user has a row-major matrix, but likes to view it as col major. One example is the Triton flash attention code using the order attribute introduced by Triton block pointer programming (such programming mixes the row-major and column-major).  With the col major view, the user can swap the i, j in the program. To support such a programming style, we introduced the order attribute to xetile.tile data type. It provides an abstraction on top of row-major only XeGPU ops.

This is a use case for the order attribute of xetile.tile. In this use case, the matrix B has a transposed memory layout to start with, for example BT [N,K] instead of B[K, N].  But the user likes  to use the original  program to index the matrix as if it is B[K, N], the order attribute is introduced to support this programming. User can flip the 2d block offset and size, and swap the stride from [K, 1] to [1, K].
```mlir
   BF16 A[M][K], BT[N, K], C[M][N];    // C = MM(A, BT)
   For i = 0, M-1, M_tile  Do
    For j = 0, N-1, N_tile Do
        For k = 0, K-1, K_tile  Do
            %a = init_tile &A, [i, k], [M, K], [K, 1] : tile<64x32xbf16>;                	// M_tile=64, K_tile=32
            %b = init_tile &BT, [k, j], [K, N], [1, K] : tile<32x64xbf16, order = [0, 1]>;  	// K_tile =32, N_tile=64
            %c = init_tile &C, [i, j], [M, N], [N, 1] : tile<64x64xbf16>;               	// M_tile=64, N_tile=64
            %va = load_tile %a : vector<64x32xbf16>;
             %vb = load_tile %b : vector<32x64x bf16>;
             %vc = tile_mma %va, %vb : vector<64x32xbf16>, vector<32x64x bf16> into vector<64x64xbf16>;
```

Alternatively, the user may just writes the program according to the given memory layout but apply a tile_transpose after the code being loaded. This is also an valid code.
```
BF16 A[M][K], BT[N, K], C[M][N];    // C = MM(A, BT)
For i = 0, M-1, M_tile  Do
    For j = 0, N-1, N_tile Do
        For k = 0, K-1, K_tile  Do
            %a = init_tile &A, [i, k], [M, K], [K, 1] : tile<64x32xbf16>;                // M_tile=64, K_tile=32
            %bt = init_tile &BT, [j, k], [N, K], [K, 1] : tile<64x32xbf16>;  	 	// N_tile=64, K_tile=32
            %c = init_tile &C, [i, j], [M, N], [N, 1] : tile<64x64xbf16>;               // M_tile=64, N_tile=64
             %va = load_tile %a : vector<64x32xbf16>;
             %vbt = load_tile %bt : vector<64x 32x bf16>;
             %vb = tile_transpose %vbt: vector<64x32xbf16> into vector<32x64xbf16>;
             %vc = tile_mma %va, %vbt : vector<64x32xbf16>, vector<32x64xbf16> into vector<64x64xbf16>;
```

All these three use cases can be programed by using memref and vector dialect. User may run into the same issue that matrix B is given as BT, so it is presented in the memory as a transposed matrix. User also have the same two choices to write the program, either use the plain layout memref reflecting the physical memory layout (3rd use case), or try to use the stride or affine_map attribute to represent it as “col-major” layout  (2nd use case).   Memref has a stride and affine_map attribute, both can be used to describe the memory layout. So a memref a[i, j] could be refer to the position to i*stride_i + j*stride_j (using stride), j*stride_i + i (using affien_map to swap index).  This effectively creates the same effect that order[0, 1] attribute try to provide to user. User now can swap the i, j in the program.

Below is a code example that user uses the “col-major” layout for BT matrix.  This is corresponding to the XeTile’s 2nd user case.
```mlir
//BF16 A[M][K], BT[N, K], C[M][N];    // C = MM(A, BT)
REFA = memref.alloc &A, [M, K] : memref<MxKxbf16>;
REFB = memref.alloc &B, [K, N]: memref<KxNxbf16, strided [K, 1] >;  //  “col-major” layout
// alternative: REFB = memref.alloc &B, [K, N]: memref<KxNxbf16, affine_map=<(d0, d1)->(d1, d0)>;
REFC = memref.alloc &C, [M, N] : memref<MxNxbf16>;
For i = 0, M-1, M_tile  Do
    For j = 0, N-1, N_tile Do
        For k = 0, K-1, K_tile  Do
            %a = memref.subview REFA, [i, k] : memref<MxKxbf16> to memref<64x32xbf16>;
            %b = memref.subview REFB, [k, j]: memref<KxNxbf16,strided [K, 1]> to memref<32x64xbf16,strided [K, 1]>;
            %c = memref.subview REFC, [i, j] : memref<MxNxbf16> to memref<64x64xbf16>;
            %va = vector.transfer_read %a : vector<64x32xbf16>;
             %vb = vector.transfer_read %b : vector<32x64x bf16>;
             %vc = vector.contract %va, %vb : vector<64x32xbf16>, vector<32x64x bf16> into vector<64x64xbf16>;
```

Below is a code example that user load BT matrix as is and transpose it in vector.  This is corresponding to the XeTile’s 3rd user case.
```mlir
//BF16 A[M][K], BT[N, K], C[M][N];    // C = MM(A, BT)
A = memref.alloc [M, K] : memref<MxKxbf16>;
BT = memref.alloc [N, K]: memref<NxKxbf16>;
C = memref.alloc [M, N] : memref<MxNxbf16>;
For i = 0, M-1, M_tile  Do
    For j = 0, N-1, N_tile Do
        For k = 0, K-1, K_tile  Do
            %a = memref.subview A, [i, k] : memref<MxKxbf16> to memref<64x32xbf16>;
            %bt = memref.subview BT, [j, k] : memref<NxKxbf16> to memref<64x32xbf16>;
            %c = memref.subview C, [i, j]:  memref<MxNxbf16> to memref<64x64xbf16>;
            %va = vector.transfer_read %a : vector<64x32xbf16>;
             %vbt = vector.transfer_read %bt : vector<64x32xbf16>;
             %vb = vector.transpose%bt : vector<64x32xbf16> to vector<32x64xbf16>;
             %vc = vector.contract %va, %vb : vector<64x32xbf16>, vector<32x64x bf16> into vector<64x64xbf16>;
```
The vector/memref dialect code example can be lowered to XeTile using simple one-to-one mapping: subview maps to init_tile, transfer_read to load_tile, and contract to tile_mma. To lower the subview op to init_tile, the lowering first identifies what "layout" the input memref has, then decide whether to use the order attribute for the tile created by init_tile.  The tile should have a consistent layout view with the given memref.  Since Memref stride and affine_map is very generic, we limit the xetile.tile to only support memref with the plain view (row-major) or the transposed view (col-major).

The xetile.tile order attribute needs to be consistent as the base memref’s memory layout.
Correct lowering -
```mlir
    init_tile: %0[0, 0]: memref<1024x1024xf16> -> tile<64x32xf16, order=[1, 0]>
    init_tile: %0[0, 0]: memref<1024x1024xf16> -> tile<64x32xf16>
    init_tile: %0[0, 0]: memref<1024x1024xf16, affine_map=<(d0, d1)->(d1, d0)> -> tile<64x32xf16, order=[0, 1]>
```
Incorrect lowering -
```mlir
   init_tile: %0[0, 0]: memref<1024x1024xf16, affine_map=<(d0, d1)->(d1, d0)>> -> tile<64x32xf16, order=[1, 0]>
   init_tile: %0[0, 0]: memref<1024x1024xf16, affine_map=<(d0, d1)->(d1, d0)>> -> tile<64x32xf16>
   init_tile: %0[0, 0]: memref<1024x1024xf16> -> tile<64x32xf16, order=[0, 1]>
```


## Appendix 2 - Code examples for work group level XeTile using wg_map attribute

## Appendix 2.1 Simple Gemm with prefetch
The first example shows a simple gemm. It demonstrates the different wg_map we used for prefetch and load.
```mlir
Pseudo code for simple gemm
C[4096, 4096] = matmul (A[4096, 4096], B[4096, 4096])
```

```mlir
#mp_a     = #wg_map<sg_layout=[8,4], sg_data=[32,32]>
#mp_a_pfh = #wg_map<sg_layout=[32,1], sg_data=[8,32]>
#mp_b     = #wg_map<sg_layout=[8,4], sg_data=[32,64]>
#mp_b_pfh = #wg_map<sg_layout=[4,8], sg_data=[8,32]>
#mp_c     = #wg_map<sg_layout=[8,4], sg_data=[32,64]>

func.func @test_gemm(%a : memref<4096x4096xf16>,
       %b: memref<4096x4096xf16>,
       %c: memref<4096xf32> ) {
  scf.for %i = %c0 to %c4096 step %c256 {
    scf.for %j = %c0 to %c4096 step %c256 {
       %1 = init_tile %a[%i, %c0] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_a>   // sg_layout=[8,4], sg_data=[32,32]
       %2 = init_tile %b[%c0, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b> // sg_layout=[8,4], sg_data=[32,64]
       %1p = init_tile %a[%i, %c96] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_a_pfh]>  // sg_layout=[32,1]
       %2p = init_tile %b[%c96, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_pfh> // sg_layout=[4,8]

       %3 = init_tile %c[%i, %j] : memref<4096x4096xf32> -> tile<256x256xf32, #mp_c>           // sg_layout=[32, 1]

       scf.for %k= %c0 to %c4096 step %c32 {
           %4  = load_tile %1 : tile<256x32xf16  #mp_a > -> vector<256x32xf16>	             // sg_layout=[8,4], sg_data=[32,32]
           %10 = load_tile %2  : tile<32x256xf16 #mp_b> -> vector<32x256xf16>                // sg_layout=[8,4], sg_data=[32,64]
          
           prefetch_tile %1 : tile<256x32xf16, #mp_a_pfh>             			      // sg_layout=[32,1]
           prefetch_tile %2  : tile<32x256xf16, #mp_a_pfh>                                    // sg_layout=[4,8]
           %6 = tile_mma %4, %5 {#mp_a #mp_b #mp_c} %4, %10 : (vector<256x32xf16>, vector<32x256xf16>) -> vector<256x256xf32> //sg_layout=[8,4]
           %1 = update_tile_offset   %1, %c0, %c32 :  tile<256x32xf16, #mp_a> -> tile<256x32xf16, #mp_a>
           %2 = update_tile_offset   %2, %c32, %c0 :  tile<32x256xf16, #mp_b> -> tile<256x32xf16, #mp_b>
           %1p = update_tile_offset   %1p, %c0, %c32 :  tile<256x32xf16, #mp_a_pft> -> tile<256x32xf16, #mp_a_pft>
           %2p = update_tile_offset   %2p, %c32, %c0 :  tile<32x256xf16, #mp_b_pft> -> tile<32x256xf16, #mp_b_pft>
         } 
         store_tile %3, %6: (tile<256x256xf32, #mp_c>, vector<256x256xf32>)                    // sg_layout=[8, 4]
    } 
  }
```
## Appendix 2.2 Gemm with transpose, broadcast, and reduction
The second example contains transpose, broadcast, and reduction.
```mlir
Pseduo code for the original problem.
C[4096, 4096] = matmul (A[4096, 4096], BT[4096, 4096]) + broad_cast(bcast[4096], dim=0)
Reduce[4096] = reduce_add(C[4096, 4096], dim=1)
```

```mlir
#mp_a     = #wg_map<sg_layout=[8,4], sg_data=[32,32]>
#mp_a_pfh = #wg_map<sg_layout=[32,1], sg_data=[8,32]>
#mp_b     = #wg_map<sg_layout=[8,4], sg_data=[32,64]>
#mp_bt    = #wg_map<sg_layout=[4,8], sg_data=[64,32]>
#mp_bt_pfh = #wg_map<sg_layout=[32,1], sg_data=[8,32]>
#mp_c     = #wg_map<sg_layout=[8,4], sg_data=[32,64]>

#mp_bcast = #wg_map<sg_layout=[8, 4], sg_data=[1,64]>
#mp_reduce= #wg_map<sg_layout=[32, 1], sg_data=[8, 1]>
#mp_reduce2= #wg_map<sg_layout=[32, 1], sg_data=[8, 256]>

func.func @test_gemm(%a : memref<4096x4096xf16>,
       %b: memref<4096x4096xf16>,
       %bcast: memref<4096xf32>
       %res: memref<4096xf32> ) {
  scf.for %i = %c0 to %c4096 step %c256 {
    scf.for %j = %c0 to %c4096 step %c256 {
       %1 = init_tile %a[%i, %c0] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_a>   // sg_layout=[8,4], sg_data=[32,32]
       %2 = init_tile %bt[%j, %c0] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_bt> // sg_layout=[4,8], sg_data=[64,32]
       %1p = init_tile %a[%i, %c192] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_a_pfh]>  // sg_layout=[32,1]
       %2p = init_tile %bt[%j, %c192] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_bt_pfh> // sg_layout=[32,1]

       %bcast'= memref.cast %bcast: memref<4096xf32> -> memref<1x4096xf32>
       %7 = init_tile %bcast'[%j] : memref<1x4096xf32> -> tile<1x256xf32, #mp_bast>           // sg_layout=[4, 8], sg_data=[1,32]

       %res'= memref.cast %res: memref<4096xf32> -> memref<4096x1xf32>
       %3 = init_tile %res'[%i] : memref<4096x1xf32> -> tile<256x1xf32, #mp_reduce>           // sg_layout=[32, 1]

       scf.for %k= %c0 to %c4096 step %c32 {
           %4  = load_tile %1 : tile<256x32xf16  #mp_a > -> vector<256x32xf16>	             // sg_layout=[8,4], sg_data=[32,32]
           %10 = load_tile %2  : tile<256x32xf16 #mp_bt> -> vector<256x32xf16>               // sg_layout=[4,8], sg_data=[64,32]
           %5  = tile_transpose %10 {#mp_bt #mp_b}: vector<256x32xf16> -> vector<32x256xf16>   // sg_layout=[4,8] -> sg_layout=[8,4]

           prefetch_tile %1 : tile<256x32xf16, #mp_a_pfh>             			      // sg_layout=[32,1]
           prefetch_tile %2  : tile<256x32xf16, #mp_a_pfh>                                    // sg_layout=[32,1]
           %6 = tile_mma %4, %5 {#mp_a #mp_b #mp_c} : (vector<256x32xf16>, vector<32x256xf16>) -> vector<256x256xf32> //sg_layout=[8,4]
           %1 = update_tile_offset   %1, %c0, %c32 :  tile<256x32xf16, #mp_a> -> tile<256x32xf16, #mp_a>
           %2 = update_tile_offset   %2, %c0, %c32 :  tile<256x32xf16, #mp_bt> -> tile<256x32xf16, #mp_bt>
           %1p = update_tile_offset   %1p, %c0, %c32 :  tile<256x32xf16, #mp_a_pft> -> tile<256x32xf16, #mp_a_pft>
           %2p = update_tile_offset   %2p, %c32, %c0 :  tile<256x32xf16, #mp_bt_pft> -> tile<256x32xf16, #mp_bt_pft>
         } 

         %12  = load_tile %7  : tile<1x256xf32, #mp_bcast> -> vector<1x256xf16>                          // sg_layout=[8, 4], sg_data=[1,64]
         %13 = tile_broadcast {#mp_bcast #mp_c} %12 [0]: vector<1x256xf32> => vector<256x256xf32>   	 // sg_layout=[8, 4]
         %14 = add %6, %13 : vector<256x256xf32>
         %15 = tile_conv_layout {#mp_c #mp_reduce2} %14 :  vector<256x256xf32>				   // sg_layout=[8, 4] -> sg_layout=[32, 1]
         %16 = tile_reduce {#mp_reduce2 #mp_reduce} <add> %15 [1], vector<256x256xf32> => vector<256x1xf32>  // sg_layout=[32, 1]
         store_tile %3, %7: (tile<256x1xf32, #mp_reduce>, vector<256x1xf32>)                               // sg_layout=[32, 1]
    } 
  }
```

## Appendix 2.3 Transpose optimization
The transpose in the program above can be optimized to use a slightly different mapping to remove the cross subgroup data shuffle requires for the first mapping.
```mlir
#mp_b     = #wg_map<sg_layout=[8,4], sg_data=[32,64]>
#mp_bt    = #wg_map<sg_layout=[4,8], sg_data=[64,32]>
%10 = load_tile %2  : tile<256x32xf16 #mp_bt> -> vector<256x32xf16>               // sg_layout=[4,8], sg_data=[64,32]
%5  = tile_transpose %10 {#mp_bt #mp_b}: vector<256x32xf16> -> vector<32x256xf16>   // sg_layout=[4,8] -> sg_layout=[8,4]
```

With the optimized mapping, the tile_transpose below could be implemented with in-register transpose.
```mlir
#mp_b     = #wg_map<sg_layout=[8,4], sg_data=[32,64]>
#mp_bt    = #wg_map<sg_layout=[32,1], sg_data=[64,32]>
%10 = load_tile %2  : tile<256x32xf16 #mp_bt> -> vector<256x32xf16>// sg_layout=[32,1], sg_data=[64,32]
%5  = tile_transpose %10 {#mp_bt #mp_b}: vector<256x32xf16> -> vector<32x256xf16>   // sg_layout=[32,1] ->sg_layout=[8,4]
```

## Appendix 2.4 Gemm implementation using cooperative load through shared local memory
For GPU doesn't support high-performance prefetch, the example code shows how to overlap the mma operation and tile load through shared local memory buffer to hide the load latency.
```mlir
#mp_a     = #wg_map<sg_layout=[8,8], sg_data=[32,32]>
#mp_a_cop = #wg_map<sg_layout=[64,1], sg_data=[4,32]>
#mp_b     = #wg_map<sg_layout=[8,8], sg_data=[32,32]>
#mp_b_cop = #wg_map<sg_layout=[8,8], sg_data=[4,32]>
#mp_c     = #wg_map<sg_layout=[8,8], sg_data=[32,32]>

func.func @test_gemm(%a : memref<4096x4096xf16>,
       %b: memref<4096x4096xf16>,
       %c: memref<4096xf32> ) {
  scf.for %i = %c0 to %c4096 step %c256 {
    scf.for %j = %c0 to %c4096 step %c256 {
       %a1_glb = init_tile %a[%i, %c0] : memref<4096x4096xf16 > -> tile<256x32xf16, #mp_a_cop >   // sg_layout=[64,1]
       %b1_glb = init_tile %b[%c0, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_cop >   // sg_layout=[8,8]
       %a2_glb = init_tile %a[%i, %c32] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_a_cop]>  // sg_layout=[64,1]
       %b2_glb = init_tile %b [%c32, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_cop> // sg_layout=[8,8]
       %a3_glb = init_tile %a[%i, %c64] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_a_cop]>  // sg_layout=[64,1]
       %b3_glb = init_tile %b [%c64, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_cop> // sg_layout=[8,8]
       %a4_glb = init_tile %a[%i, %c96] : memref<4096x4096xf16> -> tile<256x32xf16, #mp_a_cop]>  // sg_layout=[64,1]
       %b4_glb = init_tile %b [%c96, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_cop> // sg_layout=[8,8]

       %a1_slm = init_tile %a[%i, %c0] : memref<4096x4096xf16, slm> -> tile<256x32xf16, #mp_a_cop >   // sg_layout=[64,1]
       %b1_slm = init_tile %b[%c0, %j] : memref<4096x4096xf16, slm > -> tile<32x256xf16, #mp_b_cop >   // sg_layout=[8,8]
       %a2_slm = init_tile %a[%i, %c32] : memref<4096x4096xf16, slm > -> tile<256x32xf16, #mp_a_cop]>  // sg_layout=[64,1]
       %b2_slm = init_tile %b [%c32, %j] : memref<4096x4096xf16, slm > -> tile<32x256xf16, #mp_b_cop> // sg_layout=[8,8]
       %a3_slm = init_tile %a[%i, %c64] : memref<4096x4096xf16, slm > -> tile<256x32xf16, #mp_a_cop]>  // sg_layout=[64,1]
       %b3_slm = init_tile %b [%c64, %j] : memref<4096x4096xf16, slm > -> tile<32x256xf16, #mp_b_cop> // sg_layout=[8,8]
       %a4_slm = init_tile %a[%i, %c96] : memref<4096x4096xf16, slm > -> tile<256x32xf16, #mp_a_cop]>  // sg_layout=[64,1]
       %b4_slm = init_tile %b [%c96, %j] : memref<4096x4096xf16, slm > -> tile<32x256xf16, #mp_b_cop> // sg_layout=[8,8]

       %a1_r  = load_tile %a1_glb : tile<256x32xf16  #mp_a_cop > -> vector<256x32xf16>
       %b1_r = load_tile %b1_glb : tile<32x256xf16 #mp_b_cop> -> vector<32x256xf16>
       %a2_r  = load_tile %a2_glb : tile<256x32xf16  #mp_a_cop > -> vector<256x32xf16>
       %b2_r = load_tile %b2_glb  : tile<32x256xf16 #mp_b_cop> -> vector<32x256xf16>
       %a3_r  = load_tile %a3_glb : tile<256x32xf16  #mp_a_cop > -> vector<256x32xf16>
       %b3_r = load_tile %b3_glb  : tile<32x256xf16 #mp_b_cop> -> vector<32x256xf16>

       gpu.barrier
       store_tile %a1_r, %a1_slm: tile<256x32xf16, #mp_a_cop>, vector<256x256xf32>
       store_tile %b1_r, %b1_slm: tile<32x256xf16, #mp_b_cop>, vector<256x256xf32>
       store_tile %a2_r, %a2_slm: tile<256x32xf16, #mp_a_cop>, vector<256x256xf32>
       store_tile %b2_r, %b2_slm: tile<32x256xf16, #mp_b_cop>, vector<256x256xf32>
       store_tile %a3_r, %a3_slm: tile<256x32xf16, #mp_a_cop>, vector<256x256xf32>
       store_tile %b3_r, %b3_slm: tile<32x256xf16, #mp_b_cop>, vector<256x256xf32>

       gpu.barrier

       %a1_load = init_tile %a[%i, %c0] : memref<4096x4096xf16, slm> -> tile<256x32xf16, #mp_a >   // sg_layout=[8, 8]
       %b1_load = init_tile %b[%c0, %j] : memref<4096x4096xf16, slm > -> tile<32x256xf16, #mp_b >   // sg_layout=[8,8]

       %c_glb = init_tile %c[%i, %j] : memref<4096x4096xf32> -> tile<256x256xf32, #mp_c>         // sg_layout=[8, 8]

       %slm_offset = 0

       scf.for %k= %c0 to %c4096 step %c32 {
           // cooperative load from global
           %a4_r  = load_tile %a4_glb : tile<256x32xf16#mp_a_cop > -> vector<256x32xf16> // sg_layout=[64,1],sg_data=[4,32]
           %b4_r = load_tile %b4_glb: tile<32x256xf16 #mp_b_cop> -> vector<32x256xf16>  // sg_layout=[8,8], sg_data=[4,32]

           // load from slm
           %a1_rr  = load_tile %a1_load : tile<256x32xf16  #mp_a > -> vector<256x32xf16> // sg_layout=[8,8], sg_data=[32,32]
           %b1_rr = load_tile %b1_load : tile<32x256xf16 #mp_b> -> vector<32x256xf16>  // sg_layout=[8,8], sg_data=[32,32]

           %slm_offset = add %slm_offset,  %c32
           %slm_offset = mod %slm_offset,  %c128

           %a1_load = update_tile_offset  %a1_load, %c0, %slm_offset :  tile<256x32xf16, #mp_a> -> tile<256x32xf16, #mp_a>
           %b1_load = update_tile_offset  %b1_load, %slm_offset, %c0 :  tile<32x256xf16, #mp_b> -> tile<256x32xf16, #mp_b>
           %a4_glb = update_tile_offset   %a4_glb, %c0, %c32 : tile<256x32xf16, #mp_a_pft> -> tile<256x32xf16, #mp_a_pft>
           %b4_glb = update_tile_offset   %b4_glb, %c32, %c0 : tile<32x256xf16, #mp_b_pft> -> tile<32x256xf16, #mp_b_pft>
           %a4_slm’ = update_tile_offset  %a4_slm, %c0, %slm_offset: tile<256x32xf16, #mp_a_pft> -> tile<256x32xf16, #mp_a_pft>
           %b4_slm’ = update_tile_offset  %b4_slm, %slm_offset, %c0 : tile<32x256xf16, #mp_b_pft> -> tile<32x256xf16,#mp_b_pft>

           %c_r = tile_mma %a1_rr, %b1_rr #mp_a #mp_b #mp_c:
                   (vector<256x32xf16>, vector<32x256xf16>) -> vector<256x256xf32> // sg_layout=[8,8], sg_data=[32,32]

           gpu.barrier

           // cooperative save to slm
	   store_tile %a4_r, %a4_slm: tile<256x32xf16, #mp_a_cop>, vector<256x256xf32>
           store_tile %b4_r, %b4_slm: tile<32x256x f16, #mp_b_cop>, vector<256x256xf32>

           %a4_slm = %a4_slm’
           %b4_slm = %b4_slm’
        }
        store_tile %c_r, %c_glb: (tile<256x256xf32, #mp_c>, vector<256x256xf32>)                    // sg_layout=[8, 8]
    }
  }
}
```

## Appendix 2.5 Gemm implementation with two cache levels
For GPU support high-performance prefetch through two level of caches.
```mlir
#mp_a = #wg_map<sg_layout=[8,4], sg_data=[64,32]>
#mp_b = #wg_map<sg_layout=[8,4], sg_data=[32,64]>
#mp_c = #wg_map<sg_layout=[8,4], sg_data=[64,64]>

#mp_a_copl2 = #wg_map<sg_layout=[32,1], sg_data=[16,128]>
#mp_b_copl2 = #wg_map< sg_layout=[16,2], sg_data=[8,128]>

#mp_a_copl1 = #wg_map<sg_layout=[32,1], sg_data=[16,32]>
#mp_b_copl1 = #wg_map< sg_layout=[4, 8], sg_data=[8,32]>

func.func @test_gemm(%a : memref<4096x4096xf16>,
       %b: memref<4096x4096xf16>,
       %c: memref<4096xf32> ) {
   scf.for %i = %c0 to %c4096 step %c256 {
     scf.for %j = %c0 to %c4096 step %c256 {
        %a1_l2 = init_tile %a[%i, %c0] : memref<4096x4096xf16> -> tile<512x128xf16, #mp_a_copl2>
        %b1_l2 = init_tile %b[%c0, %j] : memref<4096x4096xf16> -> tile<128x256xf16, #mp_b_copl2>
        %a2_l2 = init_tile %a[%i, %c256] : memref<4096x4096xf16> -> tile<512x128xf16, #mp_a_copl2>
        %b2_l2 = init_tile %b[%c256, %j] : memref<4096x4096xf16> -> tile<128x256xf16, #mp_b_copl2>

        prefetch_tile %a1_l2 locality<2>: tile<512x128xf16, #mp_a_copl2>
        prefetch_tile %b1_l2 locality<2>: tile<128x256xf16, #mp_b_copl2>
	prefetch_tile %a2_l2 locality<2>: tile<512x128xf16, #mp_a_copl2>
        prefetch_tile %b2_l2 locality<2>: tile<128x256xf16, #mp_b_copl2>
        %a2_l2’ = update_tile_offset   %a2_l2, %c0, %c32 :  tile<512x128xf16, #mp_b_copl2>
        %b2_l2’ = update_tile_offset   %b2_l2, %c32, %c0 :  tile<128x256xf16, #mp_b_copl2>

        %a1_l1 = init_tile %a[%i, %c0] : memref<4096x4096xf16> -> tile<512x32xf16, #mp_a_copl1>
        %b1_l1 = init_tile %b[%c0, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_copl1>
        %a2_l1 = init_tile %a[%i, %c32] : memref<4096x4096xf16> -> tile<512x32xf16, #mp_a_copl1>
        %b2_l1 = init_tile %b[%c32, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_copl1>
        %a3_l1 = init_tile %a[%i, %c64] : memref<4096x4096xf16> -> tile<512x32xf16, #mp_a_copl1>
        %b3_l1 = init_tile %b[%c64, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_copl1>
        %a4_l1 = init_tile %a[%i, %c96] : memref<4096x4096xf16> -> tile<512x32xf16, #mp_a_copl1>
        %b4_l1 = init_tile %b[%c96, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b_copl1>

        prefetch_tile %a1_l1 locality<3>: tile<512x32xf16, #mp_a_copl1>
        prefetch_tile %b1_l1 locality<3>: tile<32x256xf16, #mp_b_copl1>
        prefetch_tile %a2_l1 locality<3>: tile<512x32xf16, #mp_a_copl1>
        prefetch_tile %b2_l1 locality<3>: tile<32x256xf16, #mp_b_copl1>
        prefetch_tile %a3_l1 locality<3>: tile<512x32xf16, #mp_a_copl1>
        prefetch_tile %b3_l1 locality<3>: tile<32x256xf16, #mp_b_copl1>
        prefetch_tile %a4_l1 locality<3>: tile<512x32xf16, #mp_a_copl1>
        prefetch_tile %b4_l1 locality<3>: tile<32x256xf16, #mp_b_copl1>
        %a4_l1’ = update_tile_offset   % a4_l1, %c0, %c128 :  tile<512x32xf16, #mp_a_copl1>
        %b4_l1’ = update_tile_offset   % b4_l1, %c128, %c0 :  tile<32x256xf16, #mp_b_copl1>

        %a1_load = init_tile %a[%i, %c0] : memref<4096x4096xf16> -> tile<512x32xf16, #mp_a>
        %b1_load = init_tile %b[%c0, %j] : memref<4096x4096xf16> -> tile<32x256xf16, #mp_b>

        %c = init_tile %c[%i, %j] : memref<4096x4096xf32> -> tile<512x256xf32, #mp_c>

        scf.for %k= %c0 to %c4096 step %c32 {
            %a1_r = load_tile %a1_load : tile<256x32xf16  #mp_a > -> vector<512x32xf16>
            %b1_r = load_tile %b1_load  : tile<32x256xf16 #mp_b> -> vector<32x256xf16>

            Scf.if (%k %4 == 0) {
                gpu.barrier
                prefetch_tile %a2_l2’ locality<2>: tile<512x128xf16, #mp_a_copl2>
                prefetch_tile %b2_l2’ locality<2>: tile<128x256xf16, #mp_b_copl2>
                %a2_l2’ = update_tile_offset   %a2_l2’, %c0, %c128 :  tile<512x128xf16, #mp_a_copl2>
                %b2_l2’ = update_tile_offset   %b2_l2’, %c128, %c0 :  tile<128x256xf16, #mp_b_copl2>
            }
            prefetch_tile %a4_l1’ locality<3>: tile<512x32xf16, #mp_a_copl1>
            prefetch_tile %b4_l1’ locality<3>: tile<32x256xf16, #mp_b_copl1>
            %a4_l1’ = update_tile_offset   %a4_l1’, %c0, %c32 :  tile<512x32xf16, #mp_a_copl1>
            %b4_l1’ = update_tile_offset   %b4_l1’, %c32, %c0 :  tile<32x256xf16, #mp_b_copl1>

            %a1_load = update_tile_offset   %a1_load, %c0, %c32 :  tile<512x32xf16, #mp_a>
            %a2_load = update_tile_offset   %b1_load, %c32, %c0 :  tile<32x256xf16, #mp_b>

            %6 = tile_mma %4, %5 #mp_a #mp_b #mp_c %4, %10 : (vector<512x32xf16>, vector<32x256xf16>) -> vector<512x256xf32>
        }
       store_tile %3, %6: (tile<512x256xf32, #mp_c>, vector<512x256xf32>)
     }
   }
}
```
