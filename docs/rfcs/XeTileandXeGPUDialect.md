# RFC for XeTile and XeGPU Dialect

## Summary
This RFC proposes XeTile and XeGPU Dialect to support efficient code generation for Xe GPU. 

## Motivation
Lowering GEMM (General matrix multiplication) to an efficient tiled loop structure that can be mapped to GPU compute hierarchy is a complicated task, with multiple factors determining the efficiency. After decomposing the task into workgroup kernels and further down to subgroup kernels, each subgroup kernel executes a GEMM operation on submatrices. Generating efficient code for GEMM requires a good decomposition that creates enough subtasks to drive high core utilization and large enough subgroup-level submatrix size for code efficiency. Besides the parallel decomposition, each hardware has its recipe for the best code sequence for GEMM, which includes high-level algorithms, like cooperative prefetch/load, k-slicing and software pipelining, and target-specific code sequences for submatrix operations. 

To facilitate efficient code generation for GEMM, we introduce two new dialects, XeTile and XeGPU dialects. XeTile dialect supports the tile-based programming model and decomposes the GEMM kernel to large pre-defined tile sizes at the subgroup and workgroup level. With the XeTile dialect, the high-level GEMM algorithm can be easily expressed. Underneath XeTile, the implementation uses target-specific recipes and HW features to get the best performance on specific hardware. Based on XeTile representation, as the GEMM is decomposed at submatrix granularity and mapped to registers, it naturally supports optimization like fusing with neighbor operations. 

The XeTile dialect provides microkernel-level functionality to build high-performance GEMM using pure MLIR lowering pipeline. It supports matrix operations on tile sizes larger than hardware matrix tiles, so that the lowering pass optimize a larger scope of multiple matrix instructions with target-specific recipes that gives the best performance. For example, it uses the most efficient 2D block loader instead of a general but inefficient load instruction whenever the 2D block fits HW requirements. Xe GPU’s 2D blocker load which loads a large chunk of data and autopad the out-of-boundary access, when the matrix address and sizes meet HW requirements. As XeTile abstracts out the HW difference, the same XeTile-based code works on any type of Xe GPU that XeTile supports. 

Based on XeTile, users can implement different GEMM algorithms. Based on the input GEMM shapes and micro architecture info, the GEMM lowering chooses one high-level algorithm, decides decomposition parameters, and generates XeTile code. 

The XeGPU dialect provides almost 1:1 mapping to match Xe instructions like DPAS and 2D block load. The matrix instructions being processed at this level exactly match the hardware instructions’ semantics including the matrix sizes. The lowering and optimizations built on top of the XeGPU dialect are target-specific. 


## Proposal
### XeTile Dialect

XeTile provides a middle-level abstraction for matmul operation, and sits between Linalg matmul named op and XeGPU Dpas op. It is not tied to specific Xe architecture. The XeTile dialect design facilitates optimization using hardware auto-padding, which generates simpler and more efficient code than the software padding. Using the tile dialect, the user doesn’t need to detect the out-of-boundary case, and the dialect takes care of unaligned shapes, so the same code runs for the unaligned use case.  Users can focus on high-level optimization like software pipelining, cooperative prefetch, and K-slicing. 

| Ops	| Syntax	| Example |
| :---   | :----   | :--- |
|init_tile	| operation ::= XeTile.init_tile $base_memref $offset0, $offset1: type($base_memref), index, index -> type($tile, attr-dict)	| %block = XeTile.init_tile %base_memref, %tile_offset:2 memref<128x128xbf16> into tile<8x16xbf16> |
|load_tile	| operation ::=XeTile.load_tile $tile  attr-dict:type($tile) ->type($res)	 | %vector_a = XeTile.load_tile %tile_a  transpose = [1,0] padding=0 tile<64x32xbf16> into vector <32x64xbf16>|
|store_tile	| operation ::=XeTile.store_tile $value, $tile attr-dict: type($value), type($tile) | XeTile.store_tile %tile_a  %vector_a  vector <64x64xbf16> into tile<64x64xbf16> |
|update_tile_offset	| operation ::=XeTile.update_tile_offset $tile, $delta0, $delta1: type($tile), index, index-> type($tile)	| %tdesc_updated = XeTile.update_nd_offset %tdesc, %offset_x, offset_y tensor_desc<32x64xbf16>, index, index -> tensor_desc<32x64xbf16> |
|prefetch_tile	| operation ::=XeTile.prefetch_tile $tile, attr-dict: type($tile)	  | XeTile.prefetch_tile %coop_tile:  tile<16x32xbf16> | 
|tile_mma	| operation ::=XeTile.dpas $matC, $matA, $matB attr_dict: type($matC), type($matA), type($matB)-> type($res)	 | %vector_c = XeTile.tile_mma %vector_c, %vector_a, %vector_b : vector <64x128xfloat>, vector <64x32xbf16>, vector<32x128xbf16>  into vector <64x128xfloat>  |

To create a 2D Tile memory descriptor, the user needs to set up a tile (init_tile) describing a 2D region within the global memory.  Setting up a tile requires the shape of the parent tile and the underneath physical memory buffer size, known as the base matrix.  The base matrix must be 2D and must be contiguous.  The XeTile takes the base matrix address pointer, shape, and strides, and the tile’s offsets and shape.  Offsets, strides, and shapes are for two dimensions and in the number of elements. base_stride[0] describes the number of elements between the two rows, describing the width of the underneath physical memory buffer, and *%base_strides[1] must be 1, as the innermost dimension of the base matrix must be contiguous. The current version only supports 2D memref with a row-major layout.  

`init_tile` takes memref as the description of the base matrix with the offsets of the specific tile. The tile shape and element data type are specified in the output tile data type, and they must be known at compile-time.  

`init_tile` with memref of static shape. Tile uses memref’s shape and strides as base_shape and base_strides. 
```mlir 
  %block = XeTile.init_tile %base_memref, [%tile_offset:2] :
     memref<128x128xbf16> into tile<8x16xbf16 >
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
  %vector_a = XeTile.load_tile %tile_a   
     tile<64x64xbf16> into vector <64x64xb16>
```
Attribute `transpose` specifies the dimensions being transposed along the load. It is commonly used for the GEMM on the backward path of DNN model, where one of input matrices needs to be transposed for matmul operation. 
```mlir
  %vector_a = XeTile.load_tile  %tile_a  { transpose = [1, 0] } :
     tile<32x64xbf16> into vector <64x32xbf16>
```
Attribute `padding` specifies the padding value for the out-of-boundary access. The default value is zero.  
```mlir
  %vector_a = XeTile.load_tile %tile_a  { padding = 1.0 }
     tile<64x64xbf16> into vector <64x64xb16>
```
`load_tile` need to be used together with the tile_mma.  The VNNI layout is not exposed to tile dialect users.  A lowering pass will add the VNNI transformation at the XeGPU dialect. 

`store_tile` stores a vector to memory. Transpose and padding attributes are not supported. 
```mlir  
  XeTile.store_tile %tile_a  %vector_a  :
   vector <64x64xbf16> into tile<64x64xbf16> 
```
`prefetch_tile` prefetches the tile to cache.  
```mlir
  XeTile.prefetch_tile %coop_tile:  tile<8x8xbf16> 
```
`tile_mma` represents the matrix multiplication on 2D vectors. The semantics can be represented by vector.contract, so tile_mma works more like a syntax sugar. This also means that the code can be lowered to vector.contract and mapped to HW without DPAS support nicely.  
```mlir
  %vector_c = XeTile.tile_mma %vector_a, %vector_b, %vector_c   :
     vector <64x128xfloat>, vector <64x32xbf16>, vector<32x128xbf16>
	   into vector <64x128xfloat>  
```
A `tile_mma` variant without vector_c initialization. 
```mlir
  %vector_c = XeTile.tile_mma %vector_a, %vector_b   :
     vector <64x32xbf16>, vector<32x128xbf16>
	   into vector <64x128xfloat>  
```
`update_tile_offset` updates tile with offset_x and offset_y, to move the current tile to a new position.  These offsets are relative offset to the current position and counted in the number of elements.  Usually only one value is needed to update since the tile is only moving along the K dimension. Users should avoid initializing new tiles repeatedly. For best performance, the user should only initialize one tile as a base tile and update the tile offset to move to a new tile.  
```mlir
  %tile_updated = XeTile.update_tile_offset %tile, %offset_x, offset_y :
		tile<64x64xbf16>, index, index into tile <64x64xbf16>
```
XeTile mapping attributes are experimental features, which maps an XeTile-based operation to subgroup threads and further to WI threads.  Without these attributes, the XeTile works at the subgroup level. With wg_map attributes, XeTile operations can be mapped to workgroup-level functions. The attributes guide the lowering from workgroup level to subgroup level by specifying how the data distributed across parallel subgroups. With sg_map attributes, user can further specify the mapping of each data element to each work item thread. These maps gives user full control on the lowering process, so that the user can tune the tiling size for both the workgroup and subgroup to tune the performance. 
Below is an example.
```mlir
   #wg_map_a = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
   #sg_map_a = #xetile.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
   #xe_map_a = #xetile.xe_map<wg = #wg_map_a, sg = #sg_map_a>

   %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xe_map_a>
``` 
`wg_map` describes the mapping between subgroup thread and the memory specified by tile. 
`wg_map.sg_layout` specifies the subgroup layout, and `wg_map.sg_data` specifies the tile size owned by each subgroup. In the example above, sg_layout=[2,2] means that each workgroup has 4 subgroups with 2D layout [2,2]. sg_data = [32,128] means that each subgroup works on a submatrix [32, 128].

The tile size must be divisible by wg_map.sg_data.  For each dimension, the size of wg_map.sg_data must be divisible by wg_map.sg_data, and the data elements assigned to each subgroup thread must be contiguous. When the tile size is smaller than the submatrix size specified by wg_map.sg_layout and wg_map.sg_data, it is distributed to subgroup threads in a round-robin fashion. If there is no more data to assign along a certain dimension, it wraps around to the beginning of the tile along that dimension.  For example, for the tile size [128, 128], the tile would be sliced to four subtiles with size [32,128], with the first and third subtile assigned to subgroup thread 0 and 1, and the second and fourth to thread 2 and 3. 

`sg_map` describes the mapping between WI thread and the memory specified by the tensor descriptor. `sg_map.wi_layout` specifies the layout in which WI threads corresponding to the memory, and `sg_map.wi_data` describes the data block accessed by each WI thread.  In the example above, wi_layout=[2, 8] means that each subgroup has 16 WI threads, and wi_data=[1,2] means that each WI thread owns a [1,2] data fragment from total [2,16] submatrix at the subgroup level.  

The wg_map.sg_data size must be divisible by sg_map.wi_layout multiplying with sg_map.wi_data.  For each dimension, the size of wg_map.sg_data must be divisible by wi_layout x wi_data, and the data elements assigned to each WI thread must be contiguous. When subgroup owned submatrix is larger than the submatrix size specified by sg_map.wi_layout and sg_map.wi_data, it is distributed to WI threads in a round-robin fashion. The full wg_map.sg_data[0:1] is distributed with the submatrix size from multiplying wi_layout[0:1] and wi_data[0:1], so each element is mapped to one and only WI thread.


### XeGPU dialect
XeGPU dialect models a subset of Xe GPU’s ISA. This is the counterpart of NVGPU and AMDGPU  dialects, which provide a bridge dialect in the MLIR gradual lowering.  XeGPU dialect works with MLIR memref and vector type and complements with Arith/Math/Vector/Memref dialect. XeGPU operations are introduced when there is a special Xe instruction not modeled by LLVM/SPIRV dialect. In some cases, one XeGPU op is mapped to multiple hardware instructions when there is no performance disadvantage by grouping them. For example, create_tdesc is mapped to a fixed sequence of instructions to create the 32-byte long address description. 
Below is a summary. 

| Ops	| Syntax	| Example |
| :---   | :----   | :--- |
|create_tdesc	| operation ::= XeGPU.create_tdesc $base_addr , $offset attr-dict : type($base_addr) , type($offset) -> type($tdesc)	| %scatter_tdesc1 = XeGPU.create_tdesc %mem_addr, %offset: int64, Vector<16 x index> -> tensor_desc <16 x bf16, #scattered,  memory_scope=slm, chunk_size_per_lane=1 > |
|load_gather	| operation ::= XeGPU.load_gather $tdesc , $mask attr-dict : type($tdesc) , type($mask) -> type($res)	| %result = XeGPU.load_gather %scatter_tdesc2, %mask {L1 = cached, L2 = uncached, transpose=[1,0]}: tensor_desc <16x8xbf16, #Scattered>, vector<16xi1> -> vector<8x16xbf16> |
|store_scatter	| operation ::= XeGPU.store_scatter $value , $tdesc , $mask attr-dict : type($value) , type($tdesc) , type($mask)	| XeGPU.store_scatter %value, %scatter_tdesc2, %mask {L1 = cached, L2 = uncached}: vector<16xbf16>, tensor_desc <16xbf16, #scattered>, vector<16xi1> |
|update_offset	| operation ::= XeGPU.update_offset $tdesc , $delta : type($tdesc) , type($delta) -> type($tdesc)	| %tdesc_updated = XeGpu.update_offset %tdesc, %offsets: tensor_desc<16xbf16, #scattered>, vector<16x index> -> tensor_desc<16xbf16, #scattered> |
|Prefetch	| operation ::= XeGPU.prefetch $tdesc  attr-dict : type($tdesc) 	| XeGPU.prefetch %scatter_tdesc1 {L1 = cached, L2 = uncached}: tensor_desc <16xbf16, #scattered> |
|atomic_rmw	| operation ::= XeGPU.atomic_rmw $kind ,  $value , $tdesc , $mask attr-dict : type($value) , type($tdesc) , type($mask) 	| %ret_value = XeGPU.atomic_rmw “addf”, %value, %scatter_mem2, %mask : vector<16xbf16>, tensor_desc <16xbf16, #scattered>, vector<16xi1> |
|create_nd_tdesc	| operation ::= XeGPU.create_nd_tdesc $base_addr  , $offset0 , $offset1 , $tdim0 , $tdim1 , $tstride0 attr-dict : type($base_addr) , index , index , index , index , index , index  -> type($tdesc)	| %tdesc2 = XeGPU.create_nd_tdesc %mem_addr, %tile_offset:2, %base_shape:2,%base_strides:2: int64, index, index, index, index, index, index  -> tensor_desc <8x16xbf16, memory_scope=global> |
|load_nd	| operation ::= XeGPU.load_nd $tdesc  attr-dict : type($tdesc)  -> type($res)	| %result = XeGPU.load_nd %tdesc2 {L1_hint = uncached, L3_hint = uncached}: tensor_desc <8x16xbf16> -> vector<8x16xbf16> |
|dpas	| operation ::= XeGPU.dpas $matC, $matA, $matB attr_dict : type($matC) , type($matA) , type($matB) -> type($res)	| %vector_c = XeGPU.dpas %vector_c, %vector_a, %vector_b    vector <8x16xfloat> , vector <8x8x2xbf16>, vector<8x16x2xbf16>  -> vector <8x16xfloat> |
|store_nd	| operation ::= XeGPU.store_nd $value , $tdesc attr-dict : type($value) , type($tdesc) | XeGPU.store_nd %value, %tdesc2  {L1_hint = uncached, L3_hint = uncached}: vector<8x16xbf16>, tensor_desc <8x16xbf16> |
|update_nd_offset	| operation ::= XeGPU.update_nd_offset $tdesc , $delta0 , $delta1 : type($tdesc) , index , index -> type($tdesc)	| %tdesc_updated = XeGpu.update_nd_offset %tdesc, %offset_x, offset_y, tensor_desc<8x16xbf16>, index, index -> tensor_desc<8x16xbf16> |
|prefetch_nd	| operation ::= XeGPU.prefetch_nd $tdesc , attr-dict : type($tdesc) | XeGPU.prefetch_nd %tdesc2: tensor_desc <8x16xbf16> |
|alloc_nbarrier	| operation ::= XeGPU.alloc_nbarrier $barrier_couter : uint8_t | XeGPU.alloc_nbarrier %nbarrier_count: Uint8_t |
|create_nbarrier	| operation ::= XeGPU.create_nbarrier $nbarrier_id , $nbarrier_role  attr-dict : uint8_t , type($nbarrier_role)   -> type($nbarrier) | %nbarrier = XeGPU.create_nbarrier %nbarrier_id, %nbarrier_role {num_producers = 2, num_consumers = 2}: Uint8_t, nbarrier_role -> !XeGPU.nbarrier |
|nbarrier_arrive	| operation ::= XeGPU.nbarrier_arrive $nbarrier_id   : type($nbarrier) | XeGPU.nbarrier_arrive %nbarrier : !XeGPU.nbarrier |
|nbarrier_wait	| operation ::= XeGPU.nbarrier_wait $nbarrier_id   : type($nbarrier) | XeGPU.nbarrier_wait %nbarrier : !XeGPU.nbarrier |
|Mfence	| operation ::= XeGPU.mfence attr-dict | XeGPU.mfence {fence_scope = global} | 
|complile-hint	| operation ::= XeGPU.compile_hint attr-dict	| XeGPU.compile_hint {scheduling_barrier} |

The XeGPU dialect supports lowering from XeTile dialects, so the tile-based XeTile operation can be further decomposed to multiple XeGPU ops. For example, XeTile.load_tile operation could be lowered to XeGPU’s load_nd or load_gather operations. Compared with the XeTile dialect, the XeGPU dialect works with smaller memory size, since the core XeGPU operation maps to one hardware instruction underneath.  

XeGPU supports two flavors of load/store operations: n-dimension load (nd load) and scattered load. Both need to create a tensor descriptor to describe the addresses/offsets to the tensor data, use it to load/store/prefetch, and then update it to the next data blocks.  Nd_load can be used to map to PVC’s 1D load, 2D load, or future nd load. Scattered load requires a special tensor descriptor, which contains one separate address offset for each WI thread.  

`create_nd_tdesc` creates a tensor descriptor for an n-dimensional tensor, which describes a subview of n-dimensional base tensor. The information of the base tensor is passed as operands including base address, offsets, and strides. The shape and element data type of the tensor view (subtensor) are specified in the output tensor_desc data type, and they must be known at the compile-time. The tensor_desc design is extensible for future Xe hardware if it will cover higher rank. To create a n-dimension tensor descriptor, the user needs to pass “n” number of base_shape and base_stride for the base nd-tile, and “n” number of offesets, and the shape in the result tensor_desc has be “n” numbers.  

The example below creates a 2D tensor_desc with base matrix address, shapes, strides, and the offsets of the 2D subtensor. the tensor_desc “remembers” the base tensor buffer’s information, so when it is used to load the subtensor, the lowering will handle the out-of-boundary access implicitly and preferably using hardware auto-padding features for the out-of-boundary elements.  On PVC, the stride of the innermost dimension (base_stride[0]) must be 1.   
```mlir
 #sg_map_a = xegpu.sg_map<{wi_layout = [2, 8], wi_data = [1, 2]}>
%tdesc2 = XeGPU.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2 
		: uint64, index, index, index, index, index, index
     	into tensor_desc <8x1xbf16, #sg_map_a>

%tdesc2 = XeGPU.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2 {mode =vc}
		: uint64, index, index, index, index, index, index
     	into tensor_desc <8x16xbf16>
```
Attribute `xegpu.sg_map` follows the same definition used in xeTile.sg_map. 
create_nd also accepts memref as input instead of memory address.  The example below ignores the mode and sg_map attribute for simplicity, but works for both cases. 
```mlir
  %tdesc2 = XeGPU.create_nd_tdesc %mref, %offsets:2 
		: memref<1024x1024xbf16>, index, index
     	into tensor_desc <8x16xbf16>
```

The example below accepts a memory address and an offset and creates a 1D tensor_desc.  The tensor_desc describes a 1D vector that can be loaded cooperatively by all work item (WI) threads within the subgroup. 
```mlir
  #sg_map_a = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
  %tdesc1 = XeGPU.create _nd_tdesc %mem_addr, %offset  
		{memory_scope=slm, boundary_check=false}: 
		uint64, index into tensor_desc <16xbf16, #sg_map_a>
```
The outer dimension of wi_layout must be 1 for 1D tensor_desc.  
Attribute `memory_scope` indicates whether the tensor is located in the global or shared local memory. The default value is global. 
Attribute `boundary_check` indicates whether the operation detects the boundary and pads with zero for out-of-boundary access. The default value is true. 

Attribute `mode` indicates whether the XeGPU operation is working under “Vector Compute” (VC) mode.  Under this mode, the XeGPU op is carried out by all the WI threads within a subgroup. There is no need to specify the mapping of each individual WI thread to the data fragments. The XeGPU operation works on the vectors as a whole. 
```mlir
  %tdesc1 = XeGPU.create _nd_tdesc %mem_addr, %offset  
		{memory_scope=slm, boundary_check=false, mode = vc}: 
		uint64, index into tensor_desc <16xbf16>
```
 Any XeGPU operation working at VC mode needs to explicitly declare this attribute. The default mode is SIMT mode. When VC mode is on, the sg_map attribute should not be presented in the associated tensor_desc. 

For 1D tensor description, the base_shape and base_stride are optional, the attribute “boundary_check” must be false, “%mem_add + %offset” must not access out-of-boundary memory to avoid undefined behavior. 

`load_nd` works with create_nd_tdesc and loads the memory specified by tensor_desc to a multi-dimension vector.  
```mlir
  #sg_map_a = xegpu.sg_map<{wi_layout = [2, 8], wi_data = [1, 2]}>
  %result = XeGPU.load_nd %tdesc2 {L1_hint = uncached, L3_hint = uncached }: 
          tensor_desc <8x16xbf16, #sg_map_a > into vector<8x1xbf16 >

  %result = XeGPU.load_nd %tdesc2 {L1_hint = uncached, L3_hint = uncached, mode = vc }: 
          tensor_desc <8x16xbf16> into vector<8x16xbf16>
```
Attributes `L1_hint`, `L2_hint`, and `L3_hint` can be applied to Load_nd, specifying hint directives for different levels of cache hierarchy. On PVC, cache directive for load could be "uncached, cached, streaming, read_invaldiate".  Streaming means that the data is cached but is more likely to be swapped out, and read_invaldiate simply invalidates the cache line after read. For write, cache policy could be "uncached, write_through, write_back, streaming". Write_through writes to the next level cache immediately, and write_back holds the modification until the cache line is kicked out due to the cache replacement policy.  PVC uses L1_hint and L3_hint and omits L2_hint.  There are only a few valid combinations between L1_hint and L3_hint for PVC.  

Attribute `transpose` specifies the dimensions to be transposed during the load. On the backward path of training model computation, the input matrix needs to be transposed. The operation definition supports all data types, but hardware may have limitations.  PVC only supports data types with 4-byte (DW) and 8-byte (DQ).  
```mlir
  #sg_map_a = xegpu.sg_map<{wi_layout = [1, 16], wi_data = [1, 1]}>
  %at = XeGPU.load_nd %block_a   {transpose = [1,0] }:
     tile<8x16xf32, #sg_map > into vector <1x8xf32>

  %at = XeGPU.load_nd %block_a   {transpose = [1,0]  mode = vc}:
     tile<8x16xf32> into vector <16x8xf32>
```

Attribute `vnni_axis` supports VNNI transform for low-precision data types like fp16, bf16, and int8. VNNI transformation takes multiple low-precision data elements along the column dimension and fits them into 32-bit data along the row dimension. It effectively splits a 2D matrix [col, row] to be 3-d matrix [col/vnni_factor, row, vnni_factor] when vnni_axis is specified to be axis 0.  When vnni_axis is specified as axis 1, the VNNI transformation doesn’t change the layout but splits the VNNI axis to 2 axes.  

PVC only supports loading with VNNI transformation for low-precision data types like fp16, bf16, and int8. The VNNI layout must be applied to the weight matrix for the DPAS operation, with vnni_axis being set to 0.
```mlir  
  #sg_map_b = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [2, 1]}>
  %bt = XeGPU.load_nd %block_b   {vnni_axis = 0 }:
     tile<16x16xbf16, #sg_map_b> into vector <8x1x2xbf16>
  %bt = XeGPU.load_nd %block_b   {vnni_axis = 0, mode = vc }:
     tile<16x16xbf16> into vector <8x16x2xbf16>
```
For the sg_map, the wi_data size along the vnni_axis must match with the size required to “pack” the low-precision data into 32-bit. 

When setting vnni_axis to 1, VNNI transformation has no impact on the physical data layout, so it can be optimized away from code sequence. However, it does contribute to the code readability, since it is much easier to understand that A[8, 8, 2]  x B[8, 16, 2], vs. A[8, 16] x B[8, 16, 2]. 
```mlir    
  #sg_map_a = xegpu.sg_map<{ wi_layout = [2, 8], wi_data = [1, 2]}>
  %at = XeGPU.load_nd %block_a   {vnni_axis  = 1 }:
     tile<8x16xbf16, #sg_map_a> into vector <4x1x2xbf16>

  %at = XeGPU.load_nd %block_a   {vnni_axis  = 1, mode = vc}:
     tile<8x16xbf16> into vector <8x8x2xbf16>
``` 
VNNI transformation and transpose can be combined for low-precision data type like fp16, bf16, int8. The example below shows that a bf16 matrix [8row, 16col] is transposed to [16col, 8row], and then VNNI transform to [8col, 8row, 2col].  On PVC, this specific combination can be fulfilled by a hardware transpose on DW data type, the bf16 [8row, 16col] is viewed as DW [8row, 8col], and transposed to DW[8col, 8row], and then can be viewed as bf16 [8col, 8row, 2col].  
```mlir  
  #sg_map_a = xegpu.sg_map<{ wi_layout = [2, 8], wi_data = [1, 2]}>
  %at = XeGPU.load_nd %block_a   {transpose = [1, 0]  vnni_axis = 0 }:
     tile<8x16xbf16, #sg_map_a> into vector <4x1x2bf16>

  %at = XeGPU.load_nd %block_a   {transpose = [1, 0]  vnni_axis = 0, mode = vc }:
     tile<8x16xbf16> into vector <8x8x2bf16>
``` 
`dpas` does the matrix multiplication on the 2D matrix represented as 2D or 3-d vectors.  When the input matrix is a lower-precision data type (lower than 32bit), the matrix B must be in VNNI layout, meaning the reduction dimension needs to be split into 2 dimensions and the 3rd inner dimension has multiple data elements fitting the 32bit. 
```mlir  
  %vector_c = XeGPU.dpas %vector_a, %vector_b :
     vector <4x2xbf16>, vector<8x2xbf16> 
	   into vector <8x1xfloat>   

  %vector_c = XeGPU.dpas %vector_a, %vector_b   { mode = vc}:
     vector <8x8x2xbf16>, vector<8x16x2xbf16> 
	   into vector <8x16xfloat>   
```
`store_nd` stores a vector to memory specified by tensor_desc. 
Attributes L1_hint, L2_hint, and L3_hint can be applied to store_nd. 
```mlir  
  #sg_map_c = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
  XeGPU.store_nd %value, %tdesc2:
          vector<8x1xfp32>, tensor_desc <8x16xfp32, #sg_map_c > 

  XeGPU.store_nd %value, %tdesc2  { mode = vc}:
          vector<8x16xbf16>, tensor_desc <8x16xbf16> 
```
`prefetch_nd` prefetches the memory specified by tensor_desc to cache. 
Attributes L1_hint, L2_hint, L3_hint, and memory_scope can be applied to prefetch_nd. 
```mlir  
  XeGPU.prefetch_nd %tdesc2: tensor_desc <8x16xbf16, #sg_map_a>  
  XeGPU.prefetch_nd %tdesc2 {mode = vc} : tensor_desc <8x16xbf16 >  
  XeGPU.prefetch_nd %tdesc2: tensor_desc <16xbf16, #sg_map_a>
  XeGPU.prefetch_nd %tdesc2 {mode = vc}: tensor_desc <16xbf16 >
```
`update_nd_offset` updates the subtensor’s offsets for the tensor descriptor. These offsets are relative offset to the current position in the number of elements.  The operation is used when the processing over one subtensor is completed and moves to a new one. Usually, only one offset value is changed since the subtensor is only moving along one dimension. 
```mlir  
  %tdesc_updated = XeGpu.update_nd_offset %tdesc, %offsets:2:
  	  tensor_desc<8x16xbf16, #sg_map_a>, index, index into tensor_desc<8x16xbf16, #sg_map_a>

  %tdesc_updated = XeGpu.update_nd_offset %tdesc, %offsets:2 {mode = vc}:
  	  tensor_desc<8x16xbf16>, index, index into tensor_desc<8x16xbf16>
``` 
`create_tdesc` creates a tensor descriptor for scattered load. It accepts a memory address and a vector of offsets. The element data type and size are specified in the output tensor_desc data type, and they must be known at the compile-time.
```mlir  
// SIMD lane == 16. max is 32. Meaning 32 number of individual WI elements 
  #sg_map_a = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
  %scatter_tdesc0 = XeGPU.create_tdesc %mem_addr, %offsets { mode = vc}:
     	uint64, index, into tensor_desc<16 x uint8, #scattered, #sg_map_a>

  %scatter_tdesc0 = XeGPU.create_tdesc %mem_addr, %offsets, mode = vc}:
     	uint64, Vector<16 x index>, into tensor_desc<16 x uint8, #scattered>
```
The example above creates a tensor_desc, which describes the memory base address and offsets for 16 uint8 values in the memory.  For PVC, the number of work items (SIMD lanes) on PVC can be 1, 2, 4, 8, 16, 32. 
```mlir  
  #sg_map_a = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [8, 1]}>
  %scatter_tdesc_chunk = XeGPU.create_tdesc, %base_addr, %offsets 
		{memory_scope=slm, chunk_size_per_lane=8 }: 
		uint64, index into tensor_desc <16x8xuint16, #scattered, #sg_map_a>

  %scatter_tdesc_chunk = XeGPU.create_tdesc, %base_addr, %offsets 
		{memory_scope=slm, chunk_size_per_lane=8, mode = vc}: 
		uint64, vector<16xindex> into tensor_desc <16x8xuint16, #scattered>
```
Attribute `memory_scope` indicates whether the tensor is located in the global (default) or shared local memory. 

Attribute `chunk_size_per_lane` specifies the size being loaded per each work item (WI).  Its default value is 1, but can be set to 2, 3, 4, 8 on PVC. Each WI thread may load a consecutive chunk of data elements from the memory but put them along the column dimension. The chunk_size_per_lane attribute is VC mode only.

`load_gather` (aka. load) load data per each work item. The output vector size is consistent with the number of WI threads, as the output describes the data being loaded at the subgroup level. 

```mlir  
  #sg_map_a = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
  %result0 = XeGPU.load_gather %scatter_tdesc0, %mask {L1_hint = cached, L2_hint = uncached }: 
        	  tensor_desc <16xuint8, #Scattered, #sg_map_a>, i1 into uint8

  %result0 = XeGPU.load_gather %scatter_tdesc0, %mask {L1_hint = cached, L2_hint = uncached, mode = vc}: 
        	  tensor_desc <16xuint8, #Scattered>, vector<16xi1> into vector<16xuint8>
```

When loading a tensor_desc with chunk_size_per_lane attribute, the output vector must be 2D vector, with the chunk being treated as a separate dimension. On PVC, the consecutive 1D tensor data being loaded can be viewed as a 2D tensor loaded with transposition, with the chunk dimension transposed to the outer dimension. As the chunk_size_per_lane attribute is VC mode only, the load_gather operation must be annotated with the VC mode. 
```mlir  
%result = XeGPU.load_gather %scatter_tdesc_chunk, %mask  {L1 = cached, L2 = uncached, transpose=[1,0], mode = vc}: 
          tensor_desc <16x8xbf16, #Scattered>, vector<16xi1> -> vector<8x16xbf16>
```
The mask operand masks out memory access so that it is safe to pass out-of-boundary addresses/offsets as long as they are masked. There is no modification to the result vector registers for the masked SIMD lanes.  For tensor_desc with chunk_size_per_lane attribute, the mask applies to the first dimension in memory and not the second dimension (Chunk Size). 

Load_gather is a slightly higher level operation than PVC’s native hardware instruction. When PVC load gather, it loads each low-precision element to a uint32, then a separate instruction is needed to further gather them from the registers to fully-packed vectors. Load_gather returns a vector of uint8 fully packed. 
The data type being loaded could be uint8, uint16, uint32, uint64.  

`store_scatter` (aka. store) stores data to the memory specified by tensor_desc.  
```mlir  
#sg_map_a = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
XeGPU.store_scatter %value, %scatter_tdesc1, %mask : 
     	 uint16, i1, tensor_desc <16xuint16, #scattered, #sg_map_a>

XeGPU.store_scatter %value, %scatter_tdesc1, %mask  { mode = vc}: 
     	 vector<16xuint16>, vector<16xi1>, tensor_desc <16xuint16, #scattered>
```
Attributes `L1_hint`, `L2_hint`, `L3_hint`, and `memory_scope` can be applied to store_scatter. 

`prefetch` prefetches data from the memory specified by tensor_desc.  
```mlir  
#sg_map = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
XeGPU.prefetch %scatter_tdesc0: tensor_desc <16xuint8, #scattered, #sg_map>

XeGPU.prefetch %scatter_tdesc0  {mode = vc}: tensor_desc <16xuint8, #scattered>
```
Attributes `L1_hint`, `L2_hint`, `L3_hint` can be applied to prefetch. 

`update_offset` updates the tensor descriptor for scatter load.
```mlir  
#sg_map = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
%tdesc_updated = XeGpu.update_offsets %scatter_tdesc1, %offsets
  tensor_desc <16xuint16, #scattered, #sg_map>,uint16,into tensor_desc <16xuint16, #scattered, #sg_map>

%tdesc_updated = XeGpu.update_offsets %scatter_tdesc1, %offsets  {mode = vc}:
  	tensor_desc <16xuint16, #scattered>, vector<16xuint16>, into tensor_desc <16xuint16, #scattered>
```
`atomic_rmw` atomically reads, modifies, and writes back data to the memory specified by the tensor_desc.  XeGPU.atomic_rmw reduce to a subtensor described by the tensor_desc. 

```mlir
  #sg_map = xegpu.sg_map<{ wi_layout = [1, 16], wi_data = [1, 1]}>
  %ret_value = XeGPU.atomic_rmw “addf” %value, %scatter_Desc1, %mask}: 
          bf16, tensor_desc <16xbf16, #scattered, #sg_map>, i1 to bf16

  %ret_value = XeGPU.atomic_rmw “addf” %value, %scatter_Desc1, %mask {mode = vc}: 
          vector<16xbf16>, tensor_desc <16xbf16, #scattered>, vector<16xi1> to vector<16xbf16>
``` 
XeGPU.atomic_rmw reuses the arith dialect attribute, ::mlir::arith::AtomicRMWKindAttr. 
PVC doesn’t support atomic operation on BF16/FP16 add. The BF16/FP16 matrix needs to be converted to FP32 to perform the reduction. 
alloc_nbarrier allocates a number of named barriers. Named barrier is workgroup level resource, shared by all subgroups. 
```mlir  	
  XeGPU.alloc_nbarrier %nbarrier_count: i8
```
`create_nbarrier` assigns a role for a specific named barrier to be producer and/or consumer. The returned nbarrier object holds a description to the specified barrier, which encodes all the barrier information.  It also binds the current thread with the named barrier by holding the returned nbarrier object.  Multiple threads may bind to a same nbarrier so that they can sync with each other. 
```mlir  
  %nbarrier  = XeGPU.create_nbarrier  %nbarrier_id, %nbarrier_role, {num_producers = 2, num_consumers = 2} : i8, i8, nbarrier_role into nbarrier
```
enum class nbarrier_role : uint8_t {producer_consumer = 0,  producer = 1, consumer = 2 };

`nbarrier_arrive` notifies other threads sharing the same named barrier that it has arrived. 
```mlir   
  XeGPU.nbarrier_arrive %nbarrier  
```
`nbarrier_wait` waits until all other threads sharing the same named barrier has signaled the arrival. 
```mlir  
  XeGPU. nbarrier_wait %nbarrier  
```

`mfence` synchronizes the memory access between write and following read or write. 
```mlir  
  XeGPU.mfence { {memory_kind = "ugm" , fence_op = "none", fence_scope = "local"}
```
    Fence_op: {"none",    "evict", "invalidate", "discard", "clean", "flushl3"};
    Fence_scope : {"group", "local",  "tile",  "gpu",  "gpus",  "system", "sysacq"};
    Memory_kind: {"ugm", "ugml", "tgm", "slm"};

`compile_hint` passes performance hint to the lower-level compiler. The schedule_barrier hint prevents instructions from being reordered by a lower-level compiler. For example, a prefetch instruction is location-sensitive, but the lower-level compiler may schedule it to an undesired location.  
```mlir  
XeGPU.compile_hint {hint=schedule_barrier}
```
The same syntax of nbarrrier, mfence, and compile_hint operations work on VC mode, since they access uniform values. 


## Alternative design considerations

The alternative design of tile data type is to reuse the memref data type. The memref data type needs to be enhanced to allow attributes. So the XeTile's tile data type can be expressed with memref associated with Tile attributes. XeTile.wg_map and XeTile.sg_map are examples of these attributes.    

## Notes

Currently, there is no lower-level GPU IR like NVVM available for Intel GPU compiler toolchain. XeGPU dialect uses SPIRV Intel extension to access joint-matrix or SPRIV external function to access intel GPU VC intrinsics. This may change in the future, so we expect XeGPU lowering may change accordingly.  
