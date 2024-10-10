# RFC for XeGPU Dialect

## Summary
The XeGPU dialect provides an abstraction that closely models Xe instructions to support high-performance GEMM code generation.
The matrix instructions at this level exactly match the hardware instructions’ semantics including the matrix sizes.
The lowering and optimizations built on top of the XeGPU dialect are target-specific.

## Proposal
XeGPU dialect models a subset of Xe GPU’s ISA. This is the counterpart of NVGPU and AMDGPU dialects, which provide a bridge dialect
in the MLIR gradual lowering. XeGPU dialect works with MLIR memref and vector type and complements Arith, Math, Vector, and Memref dialects.
XeGPU operations are introduced when there is a special Xe instruction not modeled by LLVM/SPIR-V dialect, for example, DPAS and 2D block
load and store. In some cases, one XeGPU op may lower to a sequence of instructions for a dedicated and performance-critical function.
For example, create_tdesc is mapped to a fixed sequence of instructions to create an address description.

Below is a summary.

| Ops	| Syntax	| Example |
| :---   | :----   | :--- |
|create_tdesc	| operation ::= xegpu.create_tdesc $base_addr, $offset attr-dict : type($base_addr), type($offset) -> type($tdesc)	| %scatter_tdesc = xegpu.create_tdesc %mem_addr, %offset: int64, Vector<16 x index> -> tensor_desc<16 x bf16, #xegpu.scatter_tdesc_attr<memory_space=slm>> |
|load_gather	| operation ::= xegpu.load_gather $tdesc, $mask attr-dict : type($tdesc), type($mask) -> type($res)	| %result = xegpu.load_gather %scatter_tdesc, %mask {L1 = cached, L2 = uncached, transpose} : tensor_desc<16x8xbf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1> -> vector<8x16xbf16> |
|store_scatter	| operation ::= xegpu.store_scatter $value, $tdesc, $mask attr-dict : type($value), type($tdesc), type($mask)	| xegpu.store_scatter %value, %scatter_tdesc, %mask {L1 = cached, L2 = uncached} : vector<16xbf16>, tensor_desc<16xbf16, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> |
|update_offset	| operation ::= xegpu.update_offset $tdesc, $delta : type($tdesc), type($delta) -> type($tdesc)	| %tdesc_updated = xegpu.update_offset %tdesc, %offsets: tensor_desc<16xbf16, #xegpu.scatter_tdesc_attr<>>, vector<16xindex> -> tensor_desc<16xbf16, #xegpu.scatter_tdesc_attr<>> |
|Prefetch	| operation ::= xegpu.prefetch $tdesc attr-dict : type($tdesc) 	| xegpu.prefetch %scatter_tdesc1 {L1 = cached, L2 = uncached} : tensor_desc<16xbf16, #xegpu.scatter_tdesc_attr<>> |
|atomic_rmw	| operation ::= xegpu.atomic_rmw $kind, $value, $tdesc, $mask attr-dict : type($value), type($tdesc), type($mask) 	| %ret_value = xegpu.atomic_rmw “addf”, %value, %scatter_mem2, %mask : vector<16xbf16>, tensor_desc<16xbf16, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> |
|create_nd_tdesc	| operation ::= xegpu.create_nd_tdesc $base_addr, $offset0, $offset1, $tdim0, $tdim1, $tstride0 attr-dict : type($base_addr), index, index, index, index, index, index -> type($tdesc)	| %tdesc = xegpu.create_nd_tdesc %mem_addr, %tile_offset:2, %base_shape:2,%base_strides:2: int64, index, index, index, index, index, index -> tensor_desc<8x16xbf16, #xegpu.block_tdesc_attr<memory_space=global>> |
|load_nd	| operation ::= xegpu.load_nd $tdesc attr-dict : type($tdesc) -> type($res)	| %result = xegpu.load_nd %tdesc {L1_hint = uncached, L3_hint = uncached} : tensor_desc<8x16xbf16> -> vector<8x16xbf16> |
|dpas	| operation ::= xegpu.dpas $matC, $matA, $matB attr_dict : type($matC), type($matA), type($matB) -> type($res)	| %vector_c = xegpu.dpas %vector_c, %vector_a, %vector_b: vector<8x16xfloat>, vector<8x8x2xbf16>, vector<8x16x2xbf16> -> vector<8x16xfloat> |
|store_nd	| operation ::= xegpu.store_nd $value, $tdesc attr-dict : type($value), type($tdesc) | xegpu.store_nd %value, %tdesc {L1_hint = uncached, L3_hint = uncached} : vector<8x16xbf16>, tensor_desc<8x16xbf16> |
|update_nd_offset	| operation ::= xegpu.update_nd_offset $tdesc, $delta0, $delta1 : type($tdesc), index, index -> type($tdesc)	| %tdesc_updated = xegpu.update_nd_offset %tdesc, %offset_x, offset_y, tensor_desc<8x16xbf16>, index, index -> tensor_desc<8x16xbf16> |
|prefetch_nd	| operation ::= xegpu.prefetch_nd $tdesc, attr-dict : type($tdesc) | xegpu.prefetch_nd %tdesc: tensor_desc<8x16xbf16> |
|alloc_nbarrier	| operation ::= xegpu.alloc_nbarrier $total_barrier_num attr-dict: index | xegpu.creat_nbarrier %total_nbarrier_num: Uint8_t |
|init_nbarrier	| operation ::= xegpu.init_nbarrier $nbarrier_id, $participant_thread_num attr-dict : Uint8_t, Uint8_t -> type($nbarrier) | %nbarrier = xegpu.alloc_nbarrier %nbarrier_id, %participant_thread_num : Uint8_t, Uint8_t -> !xegpu.nbarrier |
|nbarrier_arrive	| operation ::= xegpu.nbarrier_arrive $nbarrier : type($nbarrier) | xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier |
|nbarrier_wait	| operation ::= xegpu.nbarrier_wait $nbarrier : type($nbarrier) | xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier |
|fence	| operation ::= xegpu.fence attr-dict | xegpu.fence {scope = gpu, memory_kind = global} |

The XeGPU dialect supports lowering from [XeTile dialects]{./XeTile.md}. The tile-based XeTile operation can be further decomposed to
multiple XeGPU ops.  For example, XeTile.load_tile operation is lowered to XeGPU’s load_nd or load_gather operations. Compared with the
XeTile dialect, the XeGPU dialect works with even smaller matrix sizes, since XeGPU operations map to one hardware instruction in most cases.

XeGPU supports two flavors of load/store operations: n-dimension load (nd load) and scattered load. Both need a tensor descriptor to
describe the addresses/offsets to a data block. The descriptor is used for load/store/prefetch, and then updated for reuse with the
next data block. Nd_load can be used to map to 1D load, 2D load, or nd load. Scattered load requires a special tensor descriptor, which
contains one separate address offset for each work item.

`create_nd_tdesc` creates a tensor descriptor for an n-dimensional tensor, which describes a subview of an n-dimensional base tensor.
The information of the base tensor is passed as operands including base address, offsets, and strides. The shape and element data type
of the tensor view (subtensor) are specified in the output tensor_desc data type, and they must be known at the compile time. The
tensor_desc design is extensible for future Xe hardware to support higher-dimension tensors. n-dimension tensor descriptor requires “n”
number of base_shape and base_stride for the base nd-tile, “n” number of offsets.

The example below creates a 2D tensor_desc with base matrix address, shapes, strides, and the offsets of the 2D subtensor. The tensor_desc
“remembers” the base tensor buffer’s information, so when it is used to load the subtensor, lowering will handle the out-of-boundary access
implicitly and preferably using hardware auto-padding features for the out-of-boundary elements. For Xe GPU targets, the stride of the
innermost dimension (base_stride[0]) must be 1.

```mlir

%tdesc2 = xegpu.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16>
```

XeGPU op is carried out by all the work items within a subgroup. The `sg_map` attribute specifies the mapping of each work item to the
data fragments and will be introduced in the next section in details. XeGPU operation without `sg_map` attribute works on the vectors as a whole.

`create_nd_tdesc` can also accept an optional `block_tdesc_attr` to extend its capablity. The `block_tdesc_attr` could encode the following
optional attributes:
- `memory_space`. It describes where the data block being described is located. `global` means device memory, or `slm` means shared local memory.
  It is default to `global`. However, it has to match with the memory scope of the base addresses. If the base address is for shared local memory,
  than the memory scope of the tensor_desc has to be shared local memory too.
- `array_length`. It is only used for load. It describes how many horizontally consecutive blocks will be loaded by a hardware load instruction.
  If the TensorDesc shape is m x n. The actual loaded block shape will be array_length x (m x n). Its default value is 1.
- `boundary_check`. It is used to indicates the hardware whether to do out-of-boundary check. The default value is true.

In the following example, create_nd_tdesc creates a tensor descriptor that covers an array of 2D subtensor. The size being covered by the tensor_desc
is multiplied with the array_length along the innermost dimension. The subtensor being created actually covers 8x32xbf16.

```mlir
%tdesc2 = xegpu.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16, #xegpu.block_tdesc_attr<array_length=2>>
```

create_nd_tdesc also accepts a memref as input instead of a memory address, shapes, and sizes. The memref can be high-dimension, and the result tensor descriptor is created out of the innermost 2 dimension of input memref.
```mlir
 %tdesc2 = xegpu.create_nd_tdesc  %mref, %offsets:2
		: memref<1024x1024xbf16>, index, index
     	into tensor_desc<8x16xbf16>

 %tdesc2 = xegpu.create_nd_tdesc  %mref, %offsets:4 {mode =vc}
		: memref<4x4x1024x1024xbf16>, index, index
     	into tensor_desc<8x16xbf16>
```

The example below accepts a memory address and an offset and creates a 1D tensor_desc. The tensor_desc describes a 1D vector that is loaded by all work items combined within the subgroup.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #tdesc_attr1 = !xegpu.block_tdesc_attr<memory_space=slm, boundary_check=false, sg= #sg_map_a>
  %tdesc1 = xegpu.create_nd_tdesc %mem_addr, %offset :
		uint64, index into tensor_desc<16xbf16, #tdesc_attr1>

  #tdesc_attr2 = !xegpu.block_tdesc_attr<memory_space=slm, boundary_check=false>
  %tdesc2 = xegpu.create_nd_tdesc %mem_addr, %offset :
		uint64, index into tensor_desc<16xbf16, #tdesc_attr2>
```

Attribute `memory_space` indicates whether the tensor is located in the global or shared local memory. The default value is global.
Attribute `boundary_check` indicates whether the operation detects the boundary and pads with zero for out-of-boundary access. The default value is true.
For 1D tensor description, the base_shape and base_stride are optional, the attribute “boundary_check” must be false, “%mem_add + %offset” must not access out-of-boundary memory to avoid undefined behavior.

`load_nd` works with create_nd_tdesc and loads the memory specified by tensor_desc to a multi-dimension vector.
```mlir
  %result = xegpu.load_nd %tdesc2 {L1_hint = uncached, L3_hint = uncached} :
          tensor_desc<8x16xbf16> into vector<8x16xbf16>
```
Attributes `L1_hint`, `L2_hint`, and `L3_hint` can be applied to Load_nd. They serve as hint directives for different levels of the cache hierarchy. The cache directive for load could be "uncached, cached, streaming, read_invaldiate".  Streaming means that the data is cached but is more likely to be swapped out, and read_invaldiate simply invalidates the cache line after read. For write, cache policy could be "uncached, write_through, write_back, streaming". Write_through writes to the next level cache immediately, and write_back holds the modification until the cache line is kicked out due to the cache replacement policy.  An Xe GPU target may use L1_hint and L3_hint and omit L2_hint. There are only a few valid combinations between L1_hint and L3_hint for a certain Xe GPU target.

Attribute `transpose` specifies the dimensions to be transposed during the load. On the backward path of training model computation, the input matrix needs to be transposed. The operation definition supports all data types, but hardware may have limitations. An Xe GPU target may only support data types with size of 4-byte (DW) or 8-byte (DQ).
```mlir
  %at = xegpu.load_nd %tdesc2 {transpose = [1,0]} :
     tensor_desc<16x8xf32> into vector<8x16xf32>
```
Attribute `packed` supports VNNI transform for low-precision data types like fp16, bf16, and int8. VNNI transformation takes multiple low-precision
data elements along the row dimension and fits them into 32-bit data along the column dimension. It effectively splits a 2D matrix [col, row] to be
3-d matrix [col/vnni_factor, row, vnni_factor]. The first dimension needs to be split by a `vnni_factor`, which represents the number of elements
needed to fit 32-bit. The result tensor is always in 2D.

An Xe GPU target may only support loading with VNNI transformation for low-precision data types like fp16, bf16, and int8.

```mlir
  %bt = xegpu.load_nd %tdesc2 {packed} :
     tensor_desc<16x16xbf16> into vector<8x16x2xbf16>

```

VNNI transformation and transpose can not be combined.

Attribute `transpose_bit_width` specifies the bit_width of the data unit for the transpose during the load. The `transpose_bit_width` attribute overrides the element data type size for the transpose. For example, the transpose with `transpose_bit_width == 32` may be applied to a tile with fp16 data type, which transposes the tile as if it is a tile of "fp16 pairs".

```mlir
  %at = xegpu.load_nd %tdesc1 {transpose = [1,0], transpose_bit_width = 32} :
     tensor_desc<16x16xfp16> into vector<8x32xfp16>
```

The `transpose_bit_width` attribute can be used to transpose B matrix and at the same time perform a VNNI transformation on the transposed B matrix.
The example below shows that a tile<32x16xbf16> is transposed with `transpose_bit_width = 32`, which overrides the bf16 data type for the transpose
and treats the tile as <32x8xi32>. The transpose changes the output vector's layout to be <8x32xi32>, which is represented as vector<8x64xbf16>
using tile's element data type. User can use vector.shape_cast to explicitly represent the VNNI layout for the output vector without introducing
any data movement.

```mlir
  %at = xegpu.load_nd %block_a {transpose = [1, 0], transpose_bit_width = 32} :
     tensor_desc<32x16xfp16> into vector<8x64xfp16>
  %bt = vector.shape_cast %at :  vector<8x64xfp16> into vector<8x32x2xfp16>
```

`dpas` does the matrix multiplication on the 2D matrix represented as 2D. This is the official representation regardless the hardware
requires VNNI layout for the B matrix or not.

```mlir
  // `dpas` on 2D shape of plain layout
  %vector_c = xegpu.dpas %vector_a, %vector_b, %vector_c :
     vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xfloat>
	   into vector<8x16xfloat>

  // `dpas` variant without vector_c initialization.
  %vector_c = xegpu.dpas %vector_a, %vector_b :
     vector<8x16xbf16>, vector<16x16xbf16>
	   into vector<8x16xfloat>
```

When the input matrix is a lower-precision data type (lower than 32bit), the input vectors may use 3D representation.
When this variant is used, the matrix B must be in VNNI layout, and the matrix A may be in original 2D or reshaped to
3D represenation. The reshape for matrix A simply split the second dimension by `vnni_factor`, to match with matrix B.

```mlir
  // logically %vector_a is <8x16xbf16> and %vector_b is <16x16xbf16>
  // %vector_a is in original 2D shape, and %vector_b in VNNI layout
  %vector_c = xegpu.dpas %vector_a, %vector_b, %vector_c:
     vector<8x16xbf16>, vector<8x16x2xbf16>
	   into vector<8x16xfloat>

  // %vector_a reshaped to 3D shape, to match %vector_b's VNNI layout
  %vector_c = xegpu.dpas %vector_a, %vector_b, %vector_c:
     vector<8x8x2xbf16>, vector<8x16x2xbf16>
	   into vector<8x16xfloat>

  ```

`store_nd` stores a vector to memory specified by tensor_desc. Attributes `L1_hint`, `L2_hint`, and `L3_hint` can be applied to store_nd.
```mlir
  xegpu.store_nd %value, %tdesc2:
          vector<8x16xbf16>, tensor_desc<8x16xbf16>
  xegpu.store_nd %value, %tdesc2:
          vector<16xbf16>, tensor_desc<16xbf16>
```

`prefetch_nd` prefetches the memory specified by tensor_desc to cache.
Attributes `L1_hint`, `L2_hint`, `L3_hint`, and `memory_space` can be applied to prefetch_nd.
```mlir
  xegpu.prefetch_nd %tdesc2: tensor_desc<8x16xbf16>
  xegpu.prefetch_nd %tdesc2: tensor_desc<16xbf16>
```

`update_nd_offset` updates the subtensor’s offsets for the tensor descriptor.
These offsets are relative offset to the current position in the number of elements.
The operation is used when the processing over one subtensor is completed and moves
to a new position. Usually, only one offset value is changed since the subtensor is only moving along one dimension.
```mlir
  %tdesc_updated = xegpu.update_nd_offset %tdesc, %offsets:2 :
  	  tensor_desc<8x16xbf16>, index, index into tensor_desc<8x16xbf16>
```

`create_tdesc` creates a scattered tensor descriptor for scattered load and store. It accepts a memory
base address and a vector/array of offsets. The element data type and the total accessed size/shape are
specified in the output tensor_desc data type, and they must be known at the compile-time. The
`scatter_tdesc_attr` indicates the created tensor_desc is a scattered tensor descriptor. The size of the
offsets (number of elements) is determined by the subgroup size (the number of work item in a subgroup) of
the hardware architecture.  Each element in the offsets represents represents the offset to be accessed by
the corresponding work item. So for work item 0, it will access base + offsets[0], and for work item 7, it
will access base + offsets[7] etc. The subgroup size (size of offsets) can be 16, or 32. The following example
creates a tensor_desc, which describes the memory base address and offsets for 16 uint8 values in the memory.

```mlir
  #scattered = #xegpu.scatter_tdesc_attr<>
  %scatter_tdesc0 = xegpu.create_tdesc %mem_addr, %offsets :
     	uint64, Vector<16 x index>, into tensor_desc<16 x uint8, #scattered>
```

`scatter_tdesc_attr` could also contain the following optional attributes to extend the capbility of the operator,
as shown in the following example.
- `memory_space`. It has the same semantic to the one in `block_tdesc_attr`, describing where the data block being
  described is located: global means device memory, and slm means shared local memory. It has to match with the memory
  scope of the base addresses. It is default to global.
- `chunk_size`. It specifies the size being loaded per each work item, when each work item may load a consecutive
   chunk of data elements from the memory. With the `chunk_size` attribute, the tensor_desc created has a 2D shape
   of [subgroup_size, chunk_size]. It can be set to 2, 3, 4, 8, 16, 32, 64. However, the total size accessed by all
   work items are restricted to 512 bytes. And for low-precision data, e.g., fp16 and int8, the total accessed size
   per work item should also be 32-bit aligned if it is not accessing 1 element. Therefore, for fp16, for example,
   a valid chunk size could be 2, 4, 8, 16, 32, 64, and for int8, a valid chunk size could be 4, 8, 16, 32, 64.

```mlir
  #tdesc_attr = !xegpu.scatter_tdesc_attr< memory_space=slm, chunk_size=8>
  %scatter_tdesc_chunk = xegpu.create_tdesc, %base_addr, %offsets :
		uint64, vector<16xindex> into tensor_desc<16x8xuint16, #tdesc_attr>
```

`load_gather` (aka. load) load data per each work item. The output vector size is consistent with the subgroup size,
as the output describes the data being loaded at the subgroup level.

```mlir
  %result0 = xegpu.load_gather %scatter_tdesc0, %mask {L1_hint = cached, L2_hint = uncached} :
        	  tensor_desc<16xuint8, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> into vector<16xuint8>
```

When loading a tensor_desc with chunk_size attribute, the output vector must be a 2D vector with the shape of [chunk_size, subgroup_size].
The transpose attribute must be present to explicitly describe the transpose effect.

```mlir
  #tdesc_attr = #xegpu.scatter_tdesc_attr<memory_space=slm, chunk_size=8>
  %result = xegpu.load_gather %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose} :
          tensor_desc<16x8xbf16, #tdesc_attr>, vector<16xi1> -> vector<8x16xbf16>
```
The mask operand masks simd lanes (a.k.a offsets) by setting corresponding position to 0, so that it is safe to pass out-of-boundary
addresses/offsets as long as they are masked. There is no modification to the result vector registers for the masked SIMD lanes. For
tensor_desc with `chunk_size` attribute, the mask applies to the first dimension in memory and not the second dimension (Chunk Size).

Load_gather is a slightly higher level operation than native hardware instruction. When the hardware performs load_gather, it may load
each low-precision element to a uint32. In this case, the lowering uses an additional instruction to further gather the value from the
registers to fully-packed vectors. Load_gather returns a vector of uint8 fully packed. The data type being loaded could be uint8, uint16,
uint32, uint64.

`store_scatter` (aka. store) stores data to the memory specified by tensor_desc.
```mlir
xegpu.store_scatter %value, %scatter_tdesc1, %mask :
     	 vector<16xuint16>, vector<16xi1>, tensor_desc<16xuint16, #xegpu.scatter_tdesc_attr<>>
```
Attributes `L1_hint`, `L2_hint`, `L3_hint`, and `memory_space` can be applied to `store_scatter`. Similar to `load_gather`,
when the `chunk_size` of `tensor_desc` is specified, the `value` is a 2D vector with the shape of [chunk_size, subgroup_size].

`prefetch` prefetches data from the memory specified by tensor_desc.
```mlir
xegpu.prefetch %scatter_tdesc0 : tensor_desc<16xuint8, #xegpu.scatter_tdesc_attr<>>
```
Attributes `L1_hint`, `L2_hint`, and `L3_hint` can be applied to prefetch.

`update_offset` updates the tensor descriptor for scatter load.
```mlir
%tdesc_updated = xegpu.update_offsets %scatter_tdesc1, %offsets :
  	tensor_desc<16xuint16, #xegpu.scatter_tdesc_attr<>>, vector<16xuint16>, into tensor_desc<16xuint16, #xegpu.scatter_tdesc_attr<>>
```


`atomic_rmw` atomically reads, modifies, and writes back data to the memory specified by the tensor_desc. xegpu.atomic_rmw reduce to a subtensor described by the tensor_desc.
```mlir
  %ret_value = xegpu.atomic_rmw “addf” %value, %scatter_Desc1, %mask :
          vector<16xbf16>, tensor_desc<16xbf16, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> to vector<16xbf16>
```
xegpu.atomic_rmw reuses the arith dialect attribute, ::mlir::arith::AtomicRMWKindAttr.
In case that certain Xe GPU target does not support atomic operation for a certain data type, the user needs to convert the matrix to the supported datatype to perform the atomic operation.

`alloc_nbarrier` allocates a set of named barriers with the specified number. Named barrier is workgroup level resource, shared by all subgroups.
```mlir
  xegpu.alloc_nbarrier %total_nbarrier_num: i8
```


`init_nbarrier` returns one named barrier with the specified barrier ID to the current thread. Multiple threads may bind to the same named barrier,
and the input specifies the number of total participant threads. The returned nbarrier object holds a description of the specified barrier,
which encodes all the barrier information.
```mlir
  %nbarrier = xegpu.init_nbarrier %nbarrier_id, %participant_thread_num : i8, i8 into nbarrier
```

`nbarrier_arrive` notifies other threads sharing the same named barrier that it has arrived.
```mlir
  xegpu.nbarrier_arrive %nbarrier
```
`nbarrier_wait` waits until all other threads sharing the same named barrier have signaled the arrival.
```mlir
  xegpu.nbarrier_wait %nbarrier
```

`fence` synchronizes the memory access between write and following read or write.
```mlir
  xegpu.fence {scope = "gpu",  memory_kind = "global", }
```
Attribute `scope` describes the scope of fence. "workgroup" means that the scope is within each work group. "gpu" means the scope is across work groups within the gpu.
Attribute `Memory_kind` describes the memory kind. "global" means the global memory, "shared" means the shared local memory.

`nbarrier` and `fence` operations lower to uniform instructions, so there is no need to specify the `sg_map`.

## XeGPU Attributes to support Work Item Level semantics

Attribute `xegpu.sg_map` describes the mapping between work item (WI) and the 2D tensor specified by the tensor descriptor. To distribute the XeGPU operation to work items, the tensor_desc must be specified with the `sg_map` attribute at the tensor description creation time.

Within the sg_map, `wi_layout` specifies the layout of work items, describing the mapping of work items to the tensor. wi_layout[0] x wi_layout[1] must be equal to the subgroup size. `wi_data` specifies the minimum number of data elements assigned to each work item for a single distribution. In the example below, the subgroup has 16 work items in wi_layout=[1, 16], each accessing 1 element as specified by wi_data=[1,1].

```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %tdesc_a = xegpu.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16, #sg_map_a>
```

wi_data_size refers to the data size mapped to indiviudal work item, and sg_map_size to the collective size by all the work items as specified by sg_map. distribute_unit_size represents the minimun size of 2D tensor to be distributed to work items in a subgroup. tensor_size refers to the size of the tensor sepcified by tensor_desc.
In the example above, wi_data_size is 1, sg_map_size is 16, tensor_size is 128.
```mlir
        wi_data_size = wi_data[0] x wi_data[1]
	subgroup_size == wi_layout[0] x wi_layout[1]
 	sg_map_size[0] = wi_layout[0] x wi_data[0]
        sg_map_size[1] = wi_layout[1] x wi_data[1]
 	distribute_unit_size = sg_map_size[0] x sg_map_size[1] = subgroup_size x wi_data_size
  	tensor_size = tensor_desc[0] x tensor_desc[1]
```
wi_data_size can be larger than 1, meaning that each work item operates on multiple elements, which is eventually lowered to "SIMT-flavor" vector, like SPIR-V vector or llvm vector. The multiple elements indicated by wi_data can only be from one dimension and must be contiguous in the memory along either dimension.

To distribute a tensor, tensor_size must be divisible by distribute_unit_size. More specifically, tensor_desc[0] must be divisible by wi_layout[0] x wi_data[0], tensor_desc[1] by wi_layout[1] x wi_data[1]. The 2D subtensor is evenly distributed to work items, so each work item gets a 2D data fragment, which may contain mulitple distribution of wi_data elements.

The size of the result data fragement per work item can be computed by the following:
```mlir
	WI_data_frag[0] = tensor_desc[0]/wi_layout[0]
	WI_data_frag[1] = tensor_desc[1]/wi_layout[1]
```

The WI dsitribution is represented by the shape of the result vector being loaded, which is being reduced from [tensor_decs[0], tensor_desc[1]] to [WI_data_frag[0], WI_data_frag[1]].

With `sg_map` attribute attached to tensor_desc, xegpu.load_nd operates in SIMT flavor and returns back a fragement associated with individual work item. The tensor_desc in the example below specifies a tensor of 8x16 elements, which is decomposed to 8x1 subtensors, each with sg_map_size 1x16. The result vector <8x16xbf16> is loaded and distributed to each WI as <8x1xbf16>.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %vector_a = xegpu.load_nd %tdesc_a:
     tensor_desc<8x16xbf16, #sg_map_a> into vector<8x1xbf16>
```

For load_nd with `packed` attribute, wi_data[0] must equal to the size required to “pack” the low-precision data into 32-bit, also known as `vnni_factor`. The result vector takes wi_data[0] as inner dimension size, to indicate the effects of layout change known as "VNNI" transformation. The data fragement <16x1xbf16> is loaded and packed as <8x1x2xbf16>.

```mlir
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  %vector_b = xegpu.load_nd {packed} %tdesc1:
     tensor_desc<16x16xbf16, #sg_map_b> into vector<8x1x2xbf16>
```

For load_nd with `transpose` attribute, wi_layout is transposed to match with the tensor dimension swap. The tensor is first distributed to WI using `sg_map`, so each WI get 1x8xf32 in the example below, and then transposed to 8x1xf32. The data fragement <1x8xf32> is loaded and transposed as <8x1xf32>.
```mlir
  #sg_map_at = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>
  %at = xegpu.load_nd %tdesc1 {transpose = [1,0]} :
     tensor_desc<16x8xf32, #sg_map_at> into vector<8x1xf32>
```
`xegpu.sg_map` is also applied to 1d vector load for WI data distribution. When the tensor_desc only specify 1d tensor, `sg_map.wi_layout[0]` and `sg_map.wi_data[0]` must be 1, and they are ignored in the WI distribution.

`xegpu.sg_map` is also used to describe the WI data distribution for regular load. Below example shows that each WI loads one fp32 data element. The result vector <16xfp32> is loaded and distributed to each WI as <1xf32>.
```mlir
  #sg_map_t = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #scatter_attr = !xegpu.tdesc_attr< memory_space=slm, scattered=true>
  %scatter_tdesc = xegpu.create_tdesc, %src_addr, %offsets:
		uint64, vector<16xindex> into tensor_desc<16xfp32, #scatter_attr, #sg_map_t>

  %result = xegpu.load_gather %scatter_tdesc, %mask {L1 = cached, L2 = uncached} :
          tensor_desc<16xfp32, #tdesc_attr, #sg_map_t>, vector<1xi1> -> vector<1xfp32>
```

Below example shows that each WI loads 4 fp32 data element with the chunk_size_per_lane. This load with chunk_size_per_lane is effectively load 2D tensor and transpose. The data fragement <1x4xf32> is loaded and transposed as <4x1xf32>.
```mlir
  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 4]>
  #scatter_attr = !xegpu.tdesc_attr< memory_space=slm, scattered=true>
  %scatter_tdesc_chunk = xegpu.create_tdesc, %src_addr, %offsets
		{chunk_size_per_lane=4} :
		uint64, vector<16xindex> into tensor_desc<16x4xfp32, #scatter_attr, #sg_map_t>

  %result = xegpu.load_gather %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose=[1,0]} :
          tensor_desc<16x4xfp32, #tdesc_attr, #sg_map_t>, vector<1xi1> -> vector<4x1xfp32>
```

User must use legal sg_map value for the WI data distribution for certain operations on PVC and ARC. It includes load_nd/store_nd, load/store with chunk_size, and DPAS.

## Rules of sg_map setting for load and store on PVC and ARC
User must use for the WI data distribution of 2d block load and store to work with DPAS on PVC. Not using the sg_map value defined here leads to undefined behavior.
```mlir
# assert (wi_layout[0] x wi_layout[1] == subgroup_size) // PVC subgroup_size = 16
For matrix A load
#sg_map_a_bf16 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>    // WI data distribute from [8, 16] to [8, 1]
#sg_map_a_f16  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>    // WI data distribute from [8, 16] to [8, 1]
#sg_map_a_tf32 = xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>     // WI data distribute from [8, 8] to [4, 1]
#sg_map_a_ui8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>    // WI data distribute from [8, 32] to [8, 2]
#sg_map_a_si8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>    // WI data distribute from [8, 32] to [8, 2]
For matrix B load
#sg_map_b_bf16 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>   // WI data distribute from [16, 16] to [16, 1], packed as [8, 1, 2]
#sg_map_b_f16  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>   // WI data distribute from [16, 16] to [16, 1], packed as [8, 1, 2]
#sg_map_b_tf32 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>   // WI data distribute from [8, 16] to [8, 1]
#sg_map_b_ui8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [4, 1]>   // WI data distribute from [32, 16] to [32, 1], packed as [8, 1, 4]
#sg_map_b_si8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [4, 1]>   // WI data distribute from [32, 16] to [32, 1], packed as [8, 1, 4]
For matrix C load
#sg_map_c_f32  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
#sg_map_c_si32 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
For matrix load/store of any type
#sg_map_anytype = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
#sg_map_anytype = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
For matrix load with transpose for A or B
#sg_map_at_tf32 = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>   // WI data distribute from [16, 8] to [8, 1]
```

User must use for the WI data distribution of 2d block load and store to work with DPAS on ARC.
```mlir
# assert (wi_layout[0] x wi_layout[1] == subgroup_size) // ARC subgroup_size = 8
For matrix A load
#sg_map_a_bf16 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>   // WI data distribute from [8, 16] to [8, 2]
#sg_map_a_f16  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>   // WI data distribute from [8, 16] to [8, 2]
#sg_map_a_tf32 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>   // WI data distribute from [8, 8] to [8, 1]
#sg_map_a_ui8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>   // WI data distribute from [8, 32] to [8, 4]
#sg_map_a_si8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>   // WI data distribute from [8, 32] to [8, 4]
For matrix B load
#sg_map_b_bf16 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [2, 1]>    // WI data distribute from [16, 8] to [16, 1], packed as [8, 1, 2]
#sg_map_b_f16  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [2, 1]>    // WI data distribute from [16, 8] to [16, 1], packed as [8, 1, 2]
#sg_map_b_tf32 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>    // WI data distribute from [8, 8] to [8, 1]
#sg_map_b_ui8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [4, 1]>    // WI data distribute from [32, 8] to [32, 1], packed as [8, 1, 4]
#sg_map_b_si8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [4, 1]>    // WI data distribute from [32, 8] to [32, 1], packed as [8, 1, 4]
For matrix C load
#sg_map_c_f32  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>   // WI data distribute from [8, 8] to [8, 1]
#sg_map_c_si32 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>   // WI data distribute from [8, 8] to [8, 1]
For matrix load/store of any type
#sg_map_anytype = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>   // WI data distribute from [8, 8] to [8, 1]
#sg_map_anytype = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>   // WI data distribute from [8, 8] to [8, 1]
For matrix load with transpose for A or B
#sg_map_a_tf32 = xegpu.sg_map<wi_layout = [8, 1], wi_data = [1, 1]>   // WI data distribute from [8, 8] to [8, 1]
```
A simple rule of thumb is that wi_data size is 16 bit for matrix a (with exception for tf32 data type) on PVC. For all rest mapping, the wi_data size is 32bit, regardless PVC or ARC.
Reference: https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html

user must use for the WI data distribution of 1d block load and regular load with chunk_size on PVC and ARC. Not using this sg_map defined here leads to undefined behavior.
```mlir
  For 1d block load
  # assert (wi_layout[0] x wi_layout[1] == subgroup_size) // PVC subgroup_size = 16
  #sg_map = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>

  For regular load with chunk_size_per_lane  // PVC subgroup_size = 16
  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>

  For 1d block load
  # assert (wi_layout[0] x wi_layout[1] == subgroup_size) // ARC subgroup_size = 8
  #sg_map = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>

  For regular load with chunk_size_per_lane // ARC subgroup_size = 8
  #sg_map_t = xegpu.sg_map<wi_layout = [8, 1], wi_data = [1, 1]>
```

## Rules of sg_map setting for DPAS on PVC and ARC
The sg_map setting rule for DPAS is applied to the input and output vector operand. The sg_map setting rules of 2d block load for matrix A, B, C/D are reused. For matirx B, as the data being loaded from memory is VNNI transformed, so the wi_data needs to change accordingly so that it is consistent with the vector for operand B. It should use the following.

For PVC subgroup_size = 16
#sg_map_b_reg_bf16 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>   // WI data distribute from [8, 16, 2] to [8, 1, 2]
#sg_map_b_reg_f16  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>   // WI data distribute from [8, 16, 2] to [8, 1, 2]
#sg_map_b_reg_ui8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 4]>   // WI data distribute from [8, 16, 4] to [8, 1, 4]
#sg_map_b_reg_si8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 4]>   // WI data distribute from [8, 16, 4] to [8, 1, 4]

For ARC subgroup_size = 8
#sg_map_b_bf16 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>    // WI data distribute from [8, 8, 2] to [8, 1, 2]
#sg_map_b_f16  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>    // WI data distribute from [8, 8, 2] to [8, 1, 2]
#sg_map_b_ui8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>    // WI data distribute from [8, 8, 4] to [8, 1, 4]
#sg_map_b_si8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>    // WI data distribute from [8, 8, 4] to [8, 1, 4]

```mlir
  PVC BF16 example
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #sg_map_c = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #sg_map_b_reg = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>

  %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_a #sg_map_b_reg #sg_map_c} :vector<8x1xbf16>, vector<8x1x2xbf16> into vector<8x1xfloat>

  ARC int8 example
  #sg_map_a_ui8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>
  #sg_map_c = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
  #sg_map_b_ui8_reg = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>

  %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_a_ui #sg_map_b_ui8_reg #sg_map_c} :vector<8x4xui8>, vector<8x1x4xui8> into vector<8x1xfloat>
```

## sg_map use case - 2d load

An example on how to load a 2d block, perform dpas, and store back to memory.

```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  #sg_map_c = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>

  #sg_map_b_reg = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>

  %tdesc1 = xegpu.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16, #sg_map_a>

  %vector_a = xegpu.load_nd %tdesc1:
     tensor_desc<8x16xbf16, #sg_map_a> into vector<8x1xbf16>

  xegpu.prefetch_nd %tdesc1: tensor_desc<8x16xbf16, #sg_map_a>

  %vector_b = xegpu.load_nd %tdesc1:
     tensor_desc<16x16xbf16, #sg_map_b> into vector<8x2xbf16>

  %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_a #sg_map_b_reg #sg_map_c} :vector<8x1xbf16>, vector<8x1x2xbf16> into vector<8x1xfloat>

  xegpu.store_nd %vector_c, %tdesc2:
          vector<8x1xfloat>, tensor_desc<8x16xfloat, #sg_map_c>

  %tdesc_updated = xegpu.update_nd_offset %tdesc, %offsets:2 :
  	  tensor_desc<8x16xbf16, #sg_map_a>, index, index into tensor_desc<8x16xbf16, #sg_map_a>

```

## sg_map use case - regular load:
An example on how to perform transpose using load_gather with chunk_size_per_lane in SIMT flavor.

```mlir

  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>
  #scatter_attr = !xegpu.tdesc_attr< memory_space=slm, scattered=true>
  %scatter_tdesc_chunk = xegpu.create_tdesc, %src_addr, %offsets
		{chunk_size_per_lane=4} :
		uint64, vector<16xindex> into tensor_desc<16x4xfp32, #scatter_attr, #sg_map_t>

  %result = xegpu.load_gather %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose=[1,0]} :
          tensor_desc<16x4xfp32, #tdesc_attr, #sg_map_t>, vector<16xi1> -> vector<4x1xfp32>

  #sg_map = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #tdesc_attr = !xegpu.tdesc_attr< memory_space=slm, boundary_check=false>
  %tdesc2 = xegpu.create_nd_tdesc %dest_addr, %offset:
		uint64, index into tensor_desc<64xfp32, #tdesc_attr>
  xegpu.store_nd %value, %tdesc2:
                vector<4xfp32>, tensor_desc<64xfp32, #tdesc_attr>

```

## Notes

Currently, there is no lower-level dialect for the Intel GPU compiler toolchain to represent GPU ops with values based on LLVM data types such as NVVM
dialect for the Nvidia GPU compiler toolchain. XeGPU dialect uses LLVM or SPIR-V intrinsic to access advanced intel GPU instructions. When the lower-level
software changes, we expect XeGPU lowering passes to change accordingly.
