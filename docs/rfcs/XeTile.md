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
|init_tile	| operation ::= XeTile.init_tile $base_memref $offset0, $offset1: type($base_memref), index, index -> type($tile, attr-dict)	| %block = XeTile.init_tile %base_memref, %tile_offset:2 memref<128x128xbf16> into tile<8x16xbf16> |
|load_tile	| operation ::=XeTile.load_tile $tile attr-dict:type($tile) ->type($res)	 | %vector_a = XeTile.load_tile %tile_a {transpose = [1,0], padding=0} : tile<64x32xbf16> into vector<32x64xbf16>|
|store_tile	| operation ::=XeTile.store_tile $value, $tile attr-dict: type($value), type($tile) | XeTile.store_tile %tile_a, %vector_a: vector<64x64xbf16> into tile<64x64xbf16> |
|update_tile_offset	| operation ::=XeTile.update_tile_offset $tile, $delta0, $delta1: type($tile), index, index-> type($tile)	| %tdesc_updated = XeTile.update_nd_offset %tdesc, %offset_x, offset_y tensor_desc<32x64xbf16>, index, index -> tensor_desc<32x64xbf16> |
|prefetch_tile	| operation ::=XeTile.prefetch_tile $tile, attr-dict: type($tile)	  | XeTile.prefetch_tile %coop_tile: tile<16x32xbf16> |
|tile_mma	| operation ::=XeTile.tile_mma $matC, $matA, $matB attr_dict: type($matC), type($matA), type($matB)-> type($res)	 | %vector_c = XeTile.tile_mma %vector_c, %vector_a, %vector_b : vector<64x128xfloat>, vector<64x32xbf16>, vector<32x128xbf16> into vector<64x128xfloat>  |

To create a 2D Tile memory descriptor, the user needs to set up a tile (init_tile) describing a 2D region within the global memory. Setting up a tile requires the shape of the parent tile and the underneath physical memory buffer size, known as the base matrix. The base matrix must be 2D and must be contiguous. The XeTile takes the base matrix address pointer, shape, and strides, and the tile’s offsets and shape. Offsets, strides, and shapes are for two dimensions and in the number of elements. base_stride[0] describes the number of elements between the two rows, describing the width of the underneath physical memory buffer, and *%base_strides[1] must be 1, as the innermost dimension of the base matrix must be contiguous. The current version only supports 2D memref with a row-major layout.  

`init_tile` takes memref as the description of the base matrix with the offsets of the specific tile. The tile shape and element data type are specified in the output tile data type, and they must be known at compile-time.  

`init_tile` with memref of static shape. Tile uses memref’s shape and strides as base_shape and base_strides.
```mlir
  %block = XeTile.init_tile %base_memref, [%tile_offset:2] :
     memref<128x128xbf16> into tile<8x16xbf16>
```
`init_tile` with memref of dynamic shape. The memref has a dynamic shape, so that its shape and strides have to be passed as runtime parameters to init_tile.
```mlir
  %block = XeTile.init_tile %base_memref, [%tile_offset:2], [%base_shape:2], [%base_strides:2]:
     memref<?x?xbf16> into tile<8x16xbf16>
```
 `init_tile` with an address for the base matrix. This form is to support the use case which doesn’t use a memref to describe the base matrix.
```mlir
  %block = XeTile.init_tile %base_addr, [%tile_offset:2], [%base_shape:2], [%base_strides:2]:
     i64 into tile<8x16xbf16>
```
With the tile date type, XeTile supports load_tile, prefetch_tile, and store_tile.

`load_tile` loads a tile to a vector, which could be backed by a register region.
```mlir
  %vector_a = XeTile.load_tile %tile_a :
     tile<64x64xbf16> into vector<64x64xb16>
```
Attribute `transpose` specifies the dimensions being transposed along the load. It is commonly used for the GEMM on the backward path of DNN model, where one of input matrices needs to be transposed for matmul operation.
```mlir
  %vector_a = XeTile.load_tile %tile_a {transpose = [1, 0]} :
     tile<32x64xbf16> into vector<64x32xbf16>
```
Attribute `padding` specifies the padding value for the out-of-boundary access. The default value is zero.  
```mlir
  %vector_a = XeTile.load_tile %tile_a {padding = 1.0} :
     tile<64x64xbf16> into vector<64x64xb16>
```
`load_tile` need to be used together with the tile_mma. 

`store_tile` stores a vector to memory. Transpose and padding attributes are not supported.
```mlir  
  XeTile.store_tile %tile_a, %vector_a :
   vector<64x64xbf16> into tile<64x64xbf16>
```
`prefetch_tile` prefetches the tile to cache.  
```mlir
  XeTile.prefetch_tile %coop_tile: tile<8x8xbf16>
```
`tile_mma` represents the matrix multiplication on 2D vectors. The semantics can be represented by vector.contract, so tile_mma works more like a syntax sugar. This also means that the code can be lowered to vector.contract and mapped to HW without DPAS support nicely.  
```mlir
  %vector_c = XeTile.tile_mma %vector_a, %vector_b, %vector_c :
     vector<64x128xfloat>, vector<64x32xbf16>, vector<32x128xbf16>
	   into vector<64x128xfloat>  
```
A `tile_mma` variant without vector_c initialization.
```mlir
  %vector_c = XeTile.tile_mma %vector_a, %vector_b :
     vector<64x32xbf16>, vector<32x128xbf16>
	   into vector<64x128xfloat>  
```
`update_tile_offset` updates tile with offset_x and offset_y, to move the current tile to a new position. These offsets are relative offset to the current position and counted in the number of elements.  Usually only one value is needed to update since the tile is only moving along the K dimension. Users should avoid initializing new tiles repeatedly. For best performance, the user should only initialize one tile as a base tile and update the tile offset to move to a new tile.  
```mlir
  %tile_updated = XeTile.update_tile_offset %tile, %offset_x, offset_y :
		tile<64x64xbf16>, index, index into tile <64x64xbf16>
```

`xetile.wg_map` mapping attribute allows XeTile operation to work at the workgroup level. Without these attributes, the XeTile works at the subgroup level. With wg_map attributes, XeTile operations can be applied to workgroup-level tile sizes. The attribute `xetile.wg_map` guide the lowering from the workgroup level to the subgroup level by specifying how the data is distributed across parallel subgroups. 
`xetile.sg_map` attributes allows the user to further specify the mapping of each data element to each work item thread. It works the same way as `xegpu.sg_map` defined in XeGPU dialect. 
`xetile.wg_map` and `xeTile.sg_map` maps give the user full control over the lowering process so that the user can tune the tiling size for both the workgroup and subgroup to tune the performance.

Below is an example.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
   #sg_map_a = #xetile.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
   #xe_map_a = #xetile.xe_map<wg = #wg_map_a, sg = #sg_map_a>

   %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xe_map_a>
```
Within the `xetile.wg_map`, `sg_layout` specifies the subgroup layout, and `sg_data` specifies the tile size owned by each subgroup. In the example above, sg_layout [2,2] means that each workgroup has 4 subgroups with 2 rows and 2 columns. sg_data [32,128] means that each subgroup works on a submatrix [32, 128]. The data elements assigned to each subgroup thread must be contiguous.

For each dimension, the tile size must be divisible by `sg_data`, and the size of `sg_layout` multiplying `sg_data` must be divisible by the tile size. The tile is distributed to sg_data in a round-robin fashion. When the tile size is smaller than the size of sg_layout multiplying sg_data, the tile data is wrapped around and continues to be distributed to subgroup threads. For example, for the tile size [128, 128] and sg_data [32, 128], along the second dimension, there is no more data left to assign after the first subgroup, it wraps around and moves to the beginning of the tile and continues the assignment. Instead, for the first dimension, there is more data left after the first round of distribution, so it move to the next subtile and continue the assignement. As a result, the tile would be sliced to four subtiles with size [32,128], with the following mapping: 

| subtiles	| threads	| 
| :---   | :----   | 
| [  0:31, 0:127] | [0, 0] , [0, 1] |
| [ 32:63, 0:127] | [1, 0] , [1, 1] |
| [ 64:95, 0:127] | [0, 0] , [0, 1] |
| [96:127, 0:127] | [1, 0] , [1, 1] |


Within the `xetile.sg_map`, `wi_layout` specifies the layout in which WI threads correspond to the memory, and `wi_data` describes the data block accessed by each WI thread. In the example above, wi_layout=[2, 8] means that each subgroup has 16 WI threads in 2 rows and 8 columns, and wi_data=[1,2] means that each WI thread owns a [1,2] data fragment. The data elements with each data fragment assigned to a WI thread must be contiguous. So the sg_map describes a total [2,16] submatrix at the subgroup level.

The size of `sg_data` within `xetile.wg_map` must be divisible by sg_map size, which equals to `wi_layout` multiplying with `wi_data` within `xetile.sg_map`. More specifically, for each dimension, the `sg_data` size must be divisible by `wi_layout` x `wi_data`. The sg_data size must be larger than or equal to the sg_map size. When the 2D subtensor size is larger than the sg_map size, it is distributed to WI threads in a round-robin fashion.


## Alternative design considerations

The alternative design of tile data type is to reuse the memref data type. The memref data type needs to be enhanced to allow attributes. So the XeTile's tile data type can be expressed with memref associated with Tile attributes. XeTile.wg_map and XeTile.sg_map are examples of these attributes.  

