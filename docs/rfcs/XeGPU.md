# RFC for XeGPU Dialect

## Summary
The XeGPU dialect provides an abstraction that closely models Xe instructions to support high-performance GEMM code generation. The matrix instructions at this level exactly match the hardware instructions’ semantics including the matrix sizes. The lowering and optimizations built on top of the XeGPU dialect are target-specific.

## Proposal
XeGPU dialect models a subset of Xe GPU’s ISA. This is the counterpart of NVGPU and AMDGPU dialects, which provide a bridge dialect in the MLIR gradual lowering. XeGPU dialect works with MLIR memref and vector type and complements Arith, Math, Vector, and Memref dialects. XeGPU operations are introduced when there is a special Xe instruction not modeled by LLVM/SPIR-V dialect, for example, like DPAS and 2D block load. In some cases, one XeGPU op may lower to a sequence of instructions for a dedicated and performance-critical function. For example, create_tdesc is mapped to a fixed sequence of instructions to create an address description.

Below is a summary.

| Ops	| Syntax	| Example |
| :---   | :----   | :--- |
|create_tdesc	| operation ::= XeGPU.create_tdesc $base_addr, $offset attr-dict : type($base_addr), type($offset) -> type($tdesc)	| %scatter_tdesc1 = XeGPU.create_tdesc %mem_addr, %offset: int64, Vector<16 x index> -> tensor_desc<16 x bf16, #scattered, memory_scope=slm, chunk_size_per_lane=1> |
|load_gather	| operation ::= XeGPU.load_gather $tdesc, $mask attr-dict : type($tdesc), type($mask) -> type($res)	| %result = XeGPU.load_gather %scatter_tdesc2, %mask {L1 = cached, L2 = uncached, transpose=[1,0]} : tensor_desc<16x8xbf16, #Scattered>, vector<16xi1> -> vector<8x16xbf16> |
|store_scatter	| operation ::= XeGPU.store_scatter $value, $tdesc, $mask attr-dict : type($value), type($tdesc), type($mask)	| XeGPU.store_scatter %value, %scatter_tdesc2, %mask {L1 = cached, L2 = uncached} : vector<16xbf16>, tensor_desc<16xbf16, #scattered>, vector<16xi1> |
|update_offset	| operation ::= XeGPU.update_offset $tdesc, $delta : type($tdesc), type($delta) -> type($tdesc)	| %tdesc_updated = XeGpu.update_offset %tdesc, %offsets: tensor_desc<16xbf16, #scattered>, vector<16x index> -> tensor_desc<16xbf16, #scattered> |
|Prefetch	| operation ::= XeGPU.prefetch $tdesc attr-dict : type($tdesc) 	| XeGPU.prefetch %scatter_tdesc1 {L1 = cached, L2 = uncached} : tensor_desc<16xbf16, #scattered> |
|atomic_rmw	| operation ::= XeGPU.atomic_rmw $kind, $value, $tdesc, $mask attr-dict : type($value), type($tdesc), type($mask) 	| %ret_value = XeGPU.atomic_rmw “addf”, %value, %scatter_mem2, %mask : vector<16xbf16>, tensor_desc<16xbf16, #scattered>, vector<16xi1> |
|create_nd_tdesc	| operation ::= XeGPU.create_nd_tdesc $base_addr, $offset0, $offset1, $tdim0, $tdim1, $tstride0 attr-dict : type($base_addr), index, index, index, index, index, index -> type($tdesc)	| %tdesc2 = XeGPU.create_nd_tdesc %mem_addr, %tile_offset:2, %base_shape:2,%base_strides:2: int64, index, index, index, index, index, index -> tensor_desc<8x16xbf16, memory_scope=global> |
|load_nd	| operation ::= XeGPU.load_nd $tdesc attr-dict : type($tdesc) -> type($res)	| %result = XeGPU.load_nd %tdesc2 {L1_hint = uncached, L3_hint = uncached} : tensor_desc<8x16xbf16> -> vector<8x16xbf16> |
|dpas	| operation ::= XeGPU.dpas $matC, $matA, $matB attr_dict : type($matC), type($matA), type($matB) -> type($res)	| %vector_c = XeGPU.dpas %vector_c, %vector_a, %vector_b: vector<8x16xfloat>, vector<8x8x2xbf16>, vector<8x16x2xbf16> -> vector<8x16xfloat> |
|store_nd	| operation ::= XeGPU.store_nd $value, $tdesc attr-dict : type($value), type($tdesc) | XeGPU.store_nd %value, %tdesc2 {L1_hint = uncached, L3_hint = uncached} : vector<8x16xbf16>, tensor_desc<8x16xbf16> |
|update_nd_offset	| operation ::= XeGPU.update_nd_offset $tdesc, $delta0, $delta1 : type($tdesc), index, index -> type($tdesc)	| %tdesc_updated = XeGpu.update_nd_offset %tdesc, %offset_x, offset_y, tensor_desc<8x16xbf16>, index, index -> tensor_desc<8x16xbf16> |
|prefetch_nd	| operation ::= XeGPU.prefetch_nd $tdesc, attr-dict : type($tdesc) | XeGPU.prefetch_nd %tdesc2: tensor_desc<8x16xbf16> |
|alloc_nbarrier	| operation ::= XeGPU.alloc_nbarrier $total_barrier_num attr-dict: index | XeGPU.creat_nbarrier %total_nbarrier_num: Uint8_t |
|init_nbarrier	| operation ::= XeGPU.init_nbarrier $nbarrier_id, $participant_thread_num attr-dict : Uint8_t, Uint8_t -> type($nbarrier) | %nbarrier = XeGPU.alloc_nbarrier %nbarrier_id, %participant_thread_num : Uint8_t, Uint8_t -> !XeGPU.nbarrier |
|nbarrier_arrive	| operation ::= XeGPU.nbarrier_arrive $nbarrier : type($nbarrier) | XeGPU.nbarrier_arrive %nbarrier : !XeGPU.nbarrier |
|nbarrier_wait	| operation ::= XeGPU.nbarrier_wait $nbarrier : type($nbarrier) | XeGPU.nbarrier_wait %nbarrier : !XeGPU.nbarrier |
|fence	| operation ::= XeGPU.fence attr-dict | XeGPU.fence {scope = gpu, memory_kind = global} |

The XeGPU dialect supports lowering from [XeTile dialects]{./XeTile.md}. The tile-based XeTile operation can be further decomposed to multiple XeGPU ops. For example, XeTile.load_tile operation is lowered to XeGPU’s load_nd or load_gather operations. Compared with the XeTile dialect, the XeGPU dialect works with even smaller matrix sizes, since XeGPU operations map to one hardware instruction in most cases.

XeGPU supports two flavors of load/store operations: n-dimension load (nd load) and scattered load. Both need a tensor descriptor to describe the addresses/offsets to a data block. The descriptor is used for load/store/prefetch, and then updated for reuse with the next data block. Nd_load can be used to map to 1D load, 2D load, or nd load. Scattered load requires a special tensor descriptor, which contains one separate address offset for each WI thread.

`create_nd_tdesc` creates a tensor descriptor for an n-dimensional tensor, which describes a subview of an n-dimensional base tensor. The information of the base tensor is passed as operands including base address, offsets, and strides. The shape and element data type of the tensor view (subtensor) are specified in the output tensor_desc data type, and they must be known at the compile time. The tensor_desc design is extensible for future Xe hardware to support higher-dimension tensors. n-dimension tensor descriptor requires “n” number of base_shape and base_stride for the base nd-tile, “n” number of offsets.

The example below creates a 2D tensor_desc with base matrix address, shapes, strides, and the offsets of the 2D subtensor. The tensor_desc “remembers” the base tensor buffer’s information, so when it is used to load the subtensor, lowering will handle the out-of-boundary access implicitly and preferably using hardware auto-padding features for the out-of-boundary elements. For most Xe GPU targets, the stride of the innermost dimension (base_stride[0]) must be 1.

```mlir
 #sg_map_a = xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
%tdesc1 = XeGPU.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16, #sg_map_a>

%tdesc2 = XeGPU.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2 {mode =vc}
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16>
```
Attribute `xegpu.sg_map` describes the mapping between WI threads and the 2D subtensor specified by the tensor descriptor. With the sg_map, `wi_layout` specifies the layout in which WI threads correspond to the memory, and `wi_data` describes the data block accessed by each work item (WI) thread. In the example above, wi_layout=[2, 8] means that each subgroup has 16 WI threads in 2 rows and 8 columns, and wi_data=[1,2] means that each WI thread owns a [1,2] data fragment. The data elements with each data fragment assigned to a WI thread must be contiguous. So the sg_map describes a total [2,16] submatrix at the subgroup level.

The size of the 2D subtensor referred by the result tensor_desc must be divisible by sg_map size, which comes from wi_layout multiplied by wi_data. For each dimension, the size of the 2D subtensor must be divisible by wi_layout x wi_data. The 2D subtensor size must be larger than or equal to the sg_map size. When the 2D subtensor size is larger than the sg_map size, it is distributed to WI threads in a round-robin fashion.

Attribute `mode` indicates whether the XeGPU operation is working under “Vector Compute” (VC) mode. Under this mode, the XeGPU op is carried out by all the WI threads within a subgroup. There is no need to specify the mapping of each WI thread to the data fragments. The XeGPU operation works on the vectors as a whole.

Any XeGPU operation working at VC mode needs to explicitly declare this attribute. The default mode is SIMT mode. When VC mode is on, the sg_map attribute should not be presented in the associated tensor_desc.

create_nd_tdesc creates a tensor descriptor that covers an array of 2D subtensor. The size being covered by the tensor_desc is multiplied with the array_length along the innermost dimension. The subtensor being created in the example below covers 8x32xbf16.
```mlir
%tdesc2 = XeGPU.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2 {mode =vc}
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16, array_length=2>
```

create_nd_tdesc also accepts a memref as input instead of a memory address, shapes, and sizes. The memref can be high-dimension, and the result tensor descriptor is created out of the innermost 2 dimension of input memref.
```mlir
 #sg_map_a = xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
 %tdesc1 = XeGPU.create_nd_tdesc %mref, %offsets:2
		: memref<1024x1024xbf16>, index, index
     	into tensor_desc<8x16xbf16, #sg_map_a>

 %tdesc2 = XeGPU.create_nd_tdesc  %mref, %offsets:2 {mode =vc}
		: memref<1024x1024xbf16>, index, index
     	into tensor_desc<8x16xbf16>

 %tdesc2 = XeGPU.create_nd_tdesc  %mref, %offsets:4 {mode =vc}
		: memref<4x4x1024x1024xbf16>, index, index
     	into tensor_desc<8x16xbf16>
```

The example below accepts a memory address and an offset and creates a 1D tensor_desc. The tensor_desc describes a 1D vector that is loaded by all WI threads combined within the subgroup.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #tdesc_attr1 = !xegpu.tdesc_attr< memory_scope=slm, boundary_check=false, sg= #sg_map_a>
  %tdesc1 = XeGPU.create_nd_tdesc %mem_addr, %offset :
		uint64, index into tensor_desc<16xbf16, #tdesc_attr1>

  #tdesc_attr2 = !xegpu.tdesc_attr< memory_scope=slm, boundary_check=false>
  %tdesc2 = XeGPU.create_nd_tdesc %mem_addr, %offset {mode = vc} :
		uint64, index into tensor_desc<16xbf16, #tdesc_attr2>
```

Attribute `memory_scope` indicates whether the tensor is located in the global or shared local memory. The default value is global.
Attribute `boundary_check` indicates whether the operation detects the boundary and pads with zero for out-of-boundary access. The default value is true.
For 1D tensor description, the base_shape and base_stride are optional, the attribute “boundary_check” must be false, “%mem_add + %offset” must not access out-of-boundary memory to avoid undefined behavior. The outer dimension of `wi_layout` and `wi_data` in `xegpu.sg_map` must be 1 for 1D tensor_desc.

`load_nd` works with create_nd_tdesc and loads the memory specified by tensor_desc to a multi-dimension vector.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
  %result = XeGPU.load_nd %tdesc1 {L1_hint = uncached, L3_hint = uncached} :
          tensor_desc<8x16xbf16, #sg_map_a> into vector<4x2xbf16>

  %result = XeGPU.load_nd %tdesc2 {L1_hint = uncached, L3_hint = uncached, mode = vc} :
          tensor_desc<8x16xbf16> into vector<8x16xbf16>
```
Attributes `L1_hint`, `L2_hint`, and `L3_hint` can be applied to Load_nd. They serve as hint directives for different levels of the cache hierarchy. The cache directive for load could be "uncached, cached, streaming, read_invaldiate".  Streaming means that the data is cached but is more likely to be swapped out, and read_invaldiate simply invalidates the cache line after read. For write, cache policy could be "uncached, write_through, write_back, streaming". Write_through writes to the next level cache immediately, and write_back holds the modification until the cache line is kicked out due to the cache replacement policy.  An Xe GPU target may use L1_hint and L3_hint and omits L2_hint.  There are only a few valid combinations between L1_hint and L3_hint for a certain Xe GPU target.

Attribute `transpose` specifies the dimensions to be transposed during the load. On the backward path of training model computation, the input matrix needs to be transposed. The operation definition supports all data types, but hardware may have limitations. An Xe GPU target may only support data types with size of 4-byte (DW) or 8-byte (DQ).
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %at = XeGPU.load_nd %tdesc1 {transpose = [1,0]} :
     tile<8x16xf32, #sg_map> into vector<16x8xf32>

  %at = XeGPU.load_nd %tdesc2 {transpose = [1,0], mode = vc} :
     tile<8x16xf32> into vector<16x8xf32>
```
Attribute `packed` supports VNNI transform for low-precision data types like fp16, bf16, and int8. VNNI transformation takes multiple low-precision data elements along the row dimension and fits them into 32-bit data along the column dimension. It effectively splits a 2D matrix [col, row] to be 3-d matrix [col/vnni_factor, row, vnni_factor]. The first dimension needs to be split by a `vnni_factor`, which represents the number of elements needed to fit 32-bit. The result tensor is always in 2D.

An Xe GPU target may only support loading with VNNI transformation for low-precision data types like fp16, bf16, and int8.

```mlir
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  %bt = XeGPU.load_nd %tdesc1 {packed} :
     tile<16x16xbf16, #sg_map_b> into vector<8x16x2xbf16>
  %bt = XeGPU.load_nd %tdesc2 {packed, mode = vc} :
     tile<16x16xbf16> into vector<8x16x2xbf16>
```
For the sg_map, the wi_data size along the vnni axis (axis 0) must match with the size required to “pack” the low-precision data into 32-bit.

VNNI transformation and transpose can not be combined.

Attribute `transpose_bit_width` specifies the bit_width of the data unit for the transpose during the load. The `transpose_bit_width` attribute overrides the element data type size for the transpose. For example, the transpose with `transpose_bit_width == 32` may be applied to a tile with fp16 data type, which transposes the tile as if it is a tile of "fp16 pairs".  `transpose_bit_width` is vc mode only.

```mlir
  %at = XeGPU.load_nd %tdesc1 {transpose = [1,0], transpose_bit_width = 32, mode = vc} :
     tile<32x16xf16, #sg_map> into vector<8x64xf16>
```

The `transpose_bit_width` attribute can be used to transpose B matrix and at the same time perform a VNNI transformation on the transposed B matrix. The example below shows that a tile<32x16xbf16> is transposed with `transpose_bit_width = 32`, which overrides the bf16 data type for the transpose and treats the tile as <32x8xi32>. The transpose changes the output vector's layout to be <8x32xi32>, which is represented as vector<8x64xbf16> using tile's element data type. User can use vector.shape_cast to explicitly represent the VNNI layout for the output vector without introducing any data movement.

```mlir
  %at = XeGPU.load_nd %block_a {transpose = [1, 0], transpose_bit_width = 32, mode = vc} :
     tile<32x16xf16> into vector<8x64xf16>
  %bt = vector.shape_cast %at :  vector<8x64xf16> into vector<8x32x2xf16>
```

`dpas` does the matrix multiplication on the 2D matrix represented as 2D. This is the official representation regardless the hardware requires VNNI layout for the B matrix or not.

```mlir
  // WI mapping for vector_a, vector_b, vector_c refer to the sg_mapping associated with tensor_desc
  %vector_c = XeGPU.dpas %vector_a, %vector_b, %vector_c :
     vector<4x2xbf16>, vector<8x2xbf16>
	   into vector<8x1xfloat>

  // A `dpas`  variant without vector_c initialization.
  %vector_c = XeGPU.dpas %vector_a, %vector_b :
     vector<4x2xbf16>, vector<8x2xbf16>
	   into vector<8x1xfloat>
```

When the input matrix is a lower-precision data type (lower than 32bit), the input vectors is optionally to use 3D representation. When this variant is used, the matrix B must be in VNNI layout, and the matrix A may be in original 2D or reshaped to 3D represenation. The reshape for matrix A simply split the second dimension by `vnni_factor`, to match with matrix B.

```mlir
  // logically %vector_a is <8x16xbf16> and %vector_b is <16x16xbf16>
  // %vector_a is in original 2D shape, and %vector_b in VNNI layout
  %vector_c = XeGPU.dpas %vector_a, %vector_b, %vector_c {mode = vc} :
     vector<8x16xbf16>, vector<8x16x2xbf16>
	   into vector<8x16xfloat>
  // %vector_a reshaped to 3D shape, to match %vector_b's VNNI layout
  %vector_c = XeGPU.dpas %vector_a, %vector_b, %vector_c {mode = vc} :
     vector<8x8x2xbf16>, vector<8x16x2xbf16>
	   into vector<8x16xfloat>

  ```

`store_nd` stores a vector to memory specified by tensor_desc.
Attributes `L1_hint`, `L2_hint`, and `L3_hint` can be applied to store_nd.
```mlir
  #sg_map_c = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  XeGPU.store_nd %value, %tdesc2:
          vector<8x1xfp32>, tensor_desc<8x16xfp32, #sg_map_c>

  XeGPU.store_nd %value, %tdesc2 {mode = vc} :
          vector<8x16xbf16>, tensor_desc<8x16xbf16>
```
`prefetch_nd` prefetches the memory specified by tensor_desc to cache.
Attributes `L1_hint`, `L2_hint`, `L3_hint`, and `memory_scope` can be applied to prefetch_nd.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
  XeGPU.prefetch_nd %tdesc1: tensor_desc<8x16xbf16, #sg_map_a>
  XeGPU.prefetch_nd %tdesc2 {mode = vc} : tensor_desc<8x16xbf16>

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  XeGPU.prefetch_nd %tdesc1: tensor_desc<16xbf16, #sg_map_a>
  XeGPU.prefetch_nd %tdesc2 {mode = vc} : tensor_desc<16xbf16>
```
`update_nd_offset` updates the subtensor’s offsets for the tensor descriptor. These offsets are relative offset to the current position in the number of elements.  The operation is used when the processing over one subtensor is completed and moves to a new position. Usually, only one offset value is changed since the subtensor is only moving along one dimension.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
  %tdesc_updated = XeGpu.update_nd_offset %tdesc, %offsets:2 :
  	  tensor_desc<8x16xbf16, #sg_map_a>, index, index into tensor_desc<8x16xbf16, #sg_map_a>

  %tdesc_updated = XeGpu.update_nd_offset %tdesc, %offsets:2 {mode = vc} :
  	  tensor_desc<8x16xbf16>, index, index into tensor_desc<8x16xbf16>
```
`create_tdesc` creates a tensor descriptor for a scattered load. It accepts a memory address and a vector of offsets. The element data type and size are specified in the output tensor_desc data type, and they must be known at the compile-time. `create_tdesc` is VC mode only.
```mlir
  %scatter_tdesc0 = XeGPU.create_tdesc %mem_addr, %offsets, mode = vc} :
     	uint64, Vector<16 x index>, into tensor_desc<16 x uint8, #scattered>
```
The example above creates a tensor_desc, which describes the memory base address and offsets for 16 uint8 values in the memory.  The number of work items (SIMD lanes) can be 1, 2, 4, 8, 16, 32.
```mlir

  #tdesc_attr = !xegpu.tdesc_attr< memory_scope=slm, scattered=true>
  %scatter_tdesc_chunk = XeGPU.create_tdesc, %base_addr, %offsets
		{chunk_size_per_lane=8, mode = vc} :
		uint64, vector<16xindex> into tensor_desc<16x8xuint16, #tdesc_attr>
```
Attribute `memory_scope` indicates whether the tensor is located in the global (default) or shared local memory.

Attribute `chunk_size_per_lane` specifies the size being loaded per each work item (WI). Its default value is 1, but can be set to 2, 3, 4, 8. Each WI thread may load a consecutive chunk of data elements from the memory but put them along the column dimension.

`load_gather` (aka. load) load data per each work item. The output vector size is consistent with the number of WI threads, as the output describes the data being loaded at the subgroup level. `load_gather` is VC mode only.

```mlir
  %result0 = XeGPU.load_gather %scatter_tdesc0, %mask {L1_hint = cached, L2_hint = uncached, mode = vc} :
        	  tensor_desc<16xuint8, #Scattered>, vector<16xi1> into vector<16xuint8>
```

When loading a tensor_desc with chunk_size_per_lane attribute, the output vector must be 2D vector, with the chunk being treated as a new dimension. The consecutive 1D tensor data being loaded can be viewed as a 2D tensor loaded with transposition, with the chunk dimension transposed to the outer dimension.

```mlir
%result = XeGPU.load_gather %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose=[1,0], mode = vc} :
          tensor_desc<16x8xbf16, #Scattered>, vector<16xi1> -> vector<8x16xbf16>
```
The mask operand masks out memory access so that it is safe to pass out-of-boundary addresses/offsets as long as they are masked. There is no modification to the result vector registers for the masked SIMD lanes.  For tensor_desc with chunk_size_per_lane attribute, the mask applies to the first dimension in memory and not the second dimension (Chunk Size).

Load_gather is a slightly higher level operation than native hardware instruction. When the hardware performs load_gather, it may load each low-precision element to a uint32. In this case, the lowering uses an additional instruction to further gather the value from the registers to fully-packed vectors. Load_gather returns a vector of uint8 fully packed.
The data type being loaded could be uint8, uint16, uint32, uint64.

`store_scatter` (aka. store) stores data to the memory specified by tensor_desc.  `store_scatter` is VC mode only.
```mlir
XeGPU.store_scatter %value, %scatter_tdesc1, %mask {mode = vc} :
     	 vector<16xuint16>, vector<16xi1>, tensor_desc<16xuint16, #scattered>
```
Attributes `L1_hint`, `L2_hint`, `L3_hint`, and `memory_scope` can be applied to store_scatter.

`prefetch` prefetches data from the memory specified by tensor_desc.  `prefetch` is VC mode only.
```mlir
XeGPU.prefetch %scatter_tdesc0 {mode = vc} : tensor_desc<16xuint8, #scattered>
```
Attributes `L1_hint`, `L2_hint`, and `L3_hint` can be applied to prefetch.

`update_offset` updates the tensor descriptor for scatter load. `update_offset` is VC mode only.
```mlir
%tdesc_updated = XeGpu.update_offsets %scatter_tdesc1, %offsets {mode = vc} :
  	tensor_desc<16xuint16, #scattered>, vector<16xuint16>, into tensor_desc<16xuint16, #scattered>
```
`atomic_rmw` atomically reads, modifies, and writes back data to the memory specified by the tensor_desc. XeGPU.atomic_rmw reduce to a subtensor described by the tensor_desc. `atomic_rmw` is VC mode only.

```mlir
  %ret_value = XeGPU.atomic_rmw “addf” %value, %scatter_Desc1, %mask {mode = vc} :
          vector<16xbf16>, tensor_desc<16xbf16, #scattered>, vector<16xi1> to vector<16xbf16>
```
XeGPU.atomic_rmw reuses the arith dialect attribute, ::mlir::arith::AtomicRMWKindAttr.
In case that certain Xe GPU target does not support atomic operation for a certain data type, the user needs to convert the matrix to the supported datatype to perform the atomic operation.

`alloc_nbarrier` allocates a set of named barriers with the specified number. Named barrier is workgroup level resource, shared by all subgroups.
```mlir
  XeGPU.alloc_nbarrier %total_nbarrier_num: i8
```
`init_nbarrier` returns one named barrier with the specified barrier ID to the current thread. Multiple threads may bind to the same named barrier, and the input specifies the number of total participant threads. The returned nbarrier object holds a description of the specified barrier, which encodes all the barrier information.
```mlir
  %nbarrier = XeGPU.init_nbarrier %nbarrier_id, %participant_thread_num : i8, i8 into nbarrier
```

`nbarrier_arrive` notifies other threads sharing the same named barrier that it has arrived.
```mlir
  XeGPU.nbarrier_arrive %nbarrier
```
`nbarrier_wait` waits until all other threads sharing the same named barrier have signaled the arrival.
```mlir
  XeGPU.nbarrier_wait %nbarrier
```

`fence` synchronizes the memory access between write and following read or write.
```mlir
  XeGPU.fence {scope = "gpu",  memory_kind = "global", }
```
Attribute `scope` describes the scope of fence. "workgroup" means that the scope is within each work group. "gpu" means the scope is across work groups within the gpu.
Attribute `Memory_kind` describes the memory kind. "global" means the global memory, "shared" means the shared local memory.

`nbarrier` and `fence` operations lower to uniform instructions, so there is no need to specify the sg_map or VC mode.

## Notes

Currently, there is no lower-level dialect for the Intel GPU compiler toolchain to represent GPU ops with values based on LLVM data types such as NVVM dialect for the Nvidia GPU compiler toolchain. XeGPU dialect uses LLVM or SPIR-V intrinsic to access advanced intel GPU instructions. When the lower-level software changes, we expect XeGPU lowering passes to change accordingly.
