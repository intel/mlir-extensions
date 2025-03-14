# RFC for XeGPU Dialect

## Summary
The XeGPU dialect provides an abstraction that closely models Xe instructions to support high-performance GEMM code generation.
The XeGPU operations are designed to support tile based programming. The same set of operations work at multiple levels, including workgroup, subgroup, and work item.
The workgroup level operation can be decomposed and unrolled to multiple XeGPU operations at subgroup level, which can be further decomposed to work item level.
Along the way, the tensor size is partitioned to smaller size, and the subgroup and work item level XeGPU operations exactly match the hardware instructions’ semantics including the matrix sizes.
The lowering and optimizations built on top of the XeGPU dialect are target-specific.

## Proposal
XeGPU dialect models a subset of Xe GPU’s ISA. This is the counterpart of NVGPU and AMDGPU dialects, which provide a bridge dialect
in the MLIR gradual lowering. XeGPU dialect works with MLIR memref and vector type and complements Arith, Math, Vector, and Memref dialects.
XeGPU operations are introduced when there is a special Xe instruction not modeled by LLVM/SPIR-V dialect, for example, DPAS and 2D block
load and store. In some cases, one XeGPU op may lower to a sequence of instructions for a dedicated and performance-critical function.
For example, create_tdesc is mapped to a fixed sequence of instructions to create an address description.

Below is a summary of operation definition. The operation definition is general and works for workgroup, subgroup, or work item level.

| Ops	| Syntax	| Example |
| :---   | :----   | :--- |
|create_tdesc	| operation ::= xegpu.create_tdesc $base_addr, $offset attr-dict : type($base_addr), type($offset) -> type($tdesc)	| %scatter_tdesc = xegpu.create_tdesc %mem_addr, %offset: int64, Vector<16 x index> -> tensor_desc<16 x bf16, #xegpu.scatter_tdesc_attr<memory_space=slm>> |
|load	| operation ::= xegpu.load $tdesc, $mask attr-dict : type($tdesc), type($mask) -> type($res)	| %result = xegpu.load %scatter_tdesc, %mask {L1 = cached, L2 = uncached, transpose} : tensor_desc<16x8xbf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1> -> vector<8x16xbf16> |
|store	| operation ::= xegpu.store $value, $tdesc, $mask attr-dict : type($value), type($tdesc), type($mask)	| xegpu.store %value, %scatter_tdesc, %mask {L1 = cached, L2 = uncached} : vector<16xbf16>, tensor_desc<16xbf16, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> |
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
  #tdesc_attr2 = !xegpu.block_tdesc_attr<memory_space=slm, boundary_check=false>
  %tdesc2 = xegpu.create_nd_tdesc %mem_addr, %offset :
		uint64, index into tensor_desc<16xbf16, #tdesc_attr2>

  #tdesc_attr2 = !xegpu.block_tdesc_attr<memory_space=slm, boundary_check=false>
  %tdesc2 = xegpu.create_nd_tdesc %mref, %offset :
		memref<1024xbf16>, index into tensor_desc<16xbf16, #tdesc_attr2>
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

`dpas` does the matrix multiplication on the 2D matrix represented as 2D. This is the official representation regardless the hardware requires VNNI layout for the B matrix or not in register.

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

`load` moves data from memory to register per each work item. The output vector size is consistent with the subgroup size,
as the output describes the data being loaded at the subgroup level.

```mlir
  %result0 = xegpu.load %scatter_tdesc0, %mask {L1_hint = cached, L2_hint = uncached} :
        	  tensor_desc<16xuint8, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> into vector<16xuint8>
```

When loading a tensor_desc with chunk_size attribute, the output vector must be a 2D vector with the shape of [chunk_size, subgroup_size].
The transpose attribute must be present to explicitly describe the transpose effect.

```mlir
  #tdesc_attr = #xegpu.scatter_tdesc_attr<memory_space=slm, chunk_size=8>
  %result = xegpu.load %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose} :
          tensor_desc<16x8xbf16, #tdesc_attr>, vector<16xi1> -> vector<8x16xbf16>
```
The mask operand masks simd lanes (a.k.a offsets) by setting corresponding position to 0, so that it is safe to pass out-of-boundary
addresses/offsets as long as they are masked. There is no modification to the result vector registers for the masked SIMD lanes. For
tensor_desc with `chunk_size` attribute, the mask applies to the first dimension in memory and not the second dimension (Chunk Size).

`load` is a slightly higher level operation than native hardware instruction. When the hardware performs a load, it may load
each low-precision element to a uint32. In this case, the lowering uses an additional instruction to further gather the value from the
registers to fully-packed vectors. `Load` returns a vector of uint8 fully packed. The data type being loaded could be uint8, uint16,
uint32, uint64.

`store` moves data from register to the memory specified by tensor_desc.
```mlir
xegpu.store %value, %scatter_tdesc1, %mask :
     	 vector<16xuint16>, vector<16xi1>, tensor_desc<16xuint16, #xegpu.scatter_tdesc_attr<>>
```
Attributes `L1_hint`, `L2_hint`, `L3_hint`, and `memory_space` can be applied to `store`. Similar to `load`,
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

**Attribute xegpu.sg_map**

xegpu.sg_map specifies how a 2D tensor (defined by the tensor descriptor) is partitioned among work items (WIs) within a subgroup. sg_map consists of two parameters:
  * wi_layout: Defines the 2D arrangement of WIs within the subgroup.
  * wi_data: Specifies the shape of the tensor fragment that each WI loads or stores as a single packed data unit (16/32-bit).

`sg_map` defines a single, minimal distribution unit, where each work item in a subgroup is assigned its own data fragment. Processing the entire tensor may require one or more distribution units.

When a sg_map attribute is attached to a tensor descriptor, load/store/dpas will operate in SIMT flavor. The sg_map attribute must be specified when creating the tensor descriptor.

**Constraints**

Given these definitions:
```mlir
wi_data_size = wi_data[0] × wi_data[1]
subgroup_size = wi_layout[0] × wi_layout[1]
distribution_unit_size = subgroup_size × wi_data_size
tensor_size = tensor_desc[0] × tensor_desc[1]
n_distribution_units = tensor_size / distribution_unit_size
```

the following conditions must hold:

* subgroup_size must represent the number of WIs in a subgroup for a kernel.
* tensor_desc[0] must be evenly divisible by wi_layout[0] × wi_data[0].
* tensor_desc[1] must be evenly divisible by wi_layout[1] × wi_data[1].

As a result, tensor_size will be evenly divisible by distribution_unit_size (i.e., tensor_size % distribution_unit_size == 0), and each work item will recieve the distribution unit multiple times, with each unit having wi_data_size.
Note: When wi_data describes multiple elements, they must all come from a single, contiguous dimension.

Conceptually, the work item (WI) distribution process can be broken down into two steps. The first step divides the 2D tensor data according to `wi_layout` to obtain a 2D subtensor. The second step extracts the elements to be packed from the dimension indicated by `wi_data`, treating it as the innermost dimension, and then linearizes the remaining elements in the 2D subtensor as the outermost dimension.

**Resulting WI Data Fragment**

Each work item’s fragment of the distributed tensor is represented by a 2D vector (e.g., a SPIR-V or LLVM vector) with the shape [n_distribution_units, wi_data_size]. The result 2D vector will be further lowered to a 1D “SIMT-flavored” vector, such as a SPIR-V vector or LLVM vector, as the elements in the inner dimension being packed to a single packed data unit.

**Examples of WI distribution with sg_map**

In the example below, the subgroup has 16 work items in wi_layout=[1, 16], each accessing 1 element as specified by wi_data=[1,1]. So, wi_data_size is 1, distribution_unit_size is 16, tensor_size is 128.

```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %tdesc_a = xegpu.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16, #sg_map_a>
```

With `sg_map` attribute attached to tensor_desc, xegpu.load_nd operates in SIMT flavor and returns back a fragment associated with individual work item. The tensor_desc in the first example below specifies a tensor of 8x16 elements, which is distributed 8 times so each work item gets <8x1xbf16>. The second example shows the each work item gets <8x2xint8> with 2 int8 elements packed as one unit.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %vector_a = xegpu.load_nd %tdesc_a:
     tensor_desc<8x16xbf16, #sg_map_a> into vector<8x1xbf16>

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>
  %vector_a = xegpu.load_nd %tdesc_a:
     tensor_desc<8x32xint8, #sg_map_a> into vector<8x2xint8>
```
The example below shows a larger 2D tensor being distributed using sg_map.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %vector_a = xegpu.load_nd %tdesc_a:
     tensor_desc<12x32xbf16, #sg_map_a> into vector<24x1xbf16>

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>
  %vector_a = xegpu.load_nd %tdesc_a:
     tensor_desc<12x32xbf16, #sg_map_a> into vector<12x2xbf16>
```

The example below shows the wi_data contains 2 elements for the first dimension. The result vector takes wi_data_size as inner dimension size, the data fragement <16x1xbf16> is loaded and packed as <8x2xbf16>, a process also known as "VNNI" transformation.

```mlir
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  %vector_b = xegpu.load_nd %tdesc1:
     tensor_desc<16x16xbf16, #sg_map_b> into vector<8x2xbf16>
```

For load_nd with `transpose` attribute, wi_layout is transposed to match with the tensor dimension swap. The tensor is distributed 8 times, each time get one f32 elements, so each WI get <8x1xf32>.

```mlir
  #sg_map_tdesc = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>
  // the sg_map of the result vector after transpose is xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %at = xegpu.load_nd %tdesc1 {transpose = [1,0]} :
     tensor_desc<16x8xf32, #sg_map_at> into vector<8x1xf32>
```

The examples below demonstrate how wi_data can be used to model the transpose_bit_width. When wi_data is [1, 2], the transpose treats the matrix as consisting of 32-bit data elements. In this case, each work item receives 8x2 bf16 elements, rather than 16x1 bf16.
```mlir
  #sg_map_tdesc = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>
  // the sg_map of the result vector after transpose is xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  %at = xegpu.load_nd %tdesc1 {transpose = [1,0]}:
     tensor_desc<16x16xfp16> into vector<16x1xfp16>

  #sg_map_tdesc = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 2]>
  // the sg_map of the result vector after transpose is xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  %at = xegpu.load_nd %tdesc1 {transpose = [1,0]}:
     tensor_desc<16x16xbf16, #sg_map_at> into vector<8x2xbf16>
```

`xegpu.sg_map` is also applied to 1D vector load for WI data distribution. When the tensor_desc only specify 1D tensor, `sg_map.wi_layout[0]` and `sg_map.wi_data[0]` must be 1, and they are ignored in the WI distribution. Note that after the distribution, the output vector is shown as a 2d vector, with the inner-dimension used for packing.

```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>
  #tdesc_attr1 = !xegpu.block_tdesc_attr<memory_space=slm, boundary_check=false, sg= #sg_map_a>
  %tdesc1 = xegpu.create_nd_tdesc %mem_addr, %offset :
		uint64, index into tensor_desc<16xbf16, #tdesc_attr1>

  %vector_a = xegpu.load_nd %tdesc_1:
     tensor_desc<16xbf16, #sg_map_a> into vector<1x2xbf16>
```
`xegpu.sg_map` also applies to 3D vector, which represents the result of 2D block load with array_length.
```mlir
#sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
%tdesc2 = xegpu.create_nd_tdesc %mem_addr, %offsets:2, %base_shape:2,%base_stride:2
		: uint64, index, index, index, index, index, index
     	into tensor_desc<8x16xbf16, #xegpu.block_tdesc_attr<array_length=2>, #sg_map_a>

  %result = xegpu.load_nd %tdesc2 {L1_hint = uncached, L3_hint = uncached}:
          tensor_desc<8x16xbf16, #xegpu.block_tdesc_attr<array_length=2>> into vector<16x1xbf16>
```  

`xegpu.sg_map` is also used to describe the WI data distribution for regular loads. The example below shows that each work item (WI) loads one fp32 data element. The resulting vector <16 x fp32> is loaded and distributed to each WI as <1 x fp32>.
```mlir
  #sg_map_t = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #scatter_attr = !xegpu.tdesc_attr< memory_space=slm, scattered=true>
  %scatter_tdesc = xegpu.create_tdesc, %src_addr, %offsets:
		uint64, vector<16xindex> into tensor_desc<16xfp32, #scatter_attr, #sg_map_t>

  %result = xegpu.load %scatter_tdesc, %mask {L1 = cached, L2 = uncached} :
          tensor_desc<16xfp32, #tdesc_attr, #sg_map_t>, vector<1xi1> -> vector<1xfp32>
```

The example below illustrates how each work item loads 4 fp32 data elements with the chunk_size. This loading process, combined with the chunk_size, effectively loads a 2D tensor and performs a transpose, resulting in the transposition of the wi_layout.
```mlir
  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>
  #scatter_attr = !xegpu.tdesc_attr< memory_space=slm, scattered=true>
  %scatter_tdesc_chunk = xegpu.create_tdesc, %src_addr, %offsets
		{chunk_size=4} :
		uint64, vector<16xindex> into tensor_desc<16x4xfp32, #scatter_attr, #sg_map_t>

  %result = xegpu.load %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose=[1,0]} :
          tensor_desc<16x4xfp32, #tdesc_attr, #sg_map_t>, vector<1xi1> -> vector<4x1xfp32>
```

The load with chunk_size pack the low-precision data to 32-bit data using wi_data = [1, 2].
```mlir
  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 2]>
  #scatter_attr = !xegpu.tdesc_attr< memory_space=slm, scattered=true>
  %scatter_tdesc_chunk = xegpu.create_tdesc, %src_addr, %offsets
		{chunk_size=4} :
		uint64, vector<16xindex> into tensor_desc<16x8xbf16, #scatter_attr, #sg_map_t>

  %result = xegpu.load %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose=[1,0]} :
          tensor_desc<16x8xbf16, #tdesc_attr, #sg_map_t>, vector<1xi1> -> vector<4x2xbf16>
```

User must use legal sg_map value for the WI data distribution for certain operations on PVC and ARC. It includes load_nd/store_nd, load/store with chunk_size, and DPAS.

**Rules of sg_map setting for load and store on PVC and ARC**
The WI data distribution requires the following sg_map for the 2D block load and store to work with DPAS on PVC. Not using the sg_map value defined here leads to undefined behavior.
```mlir
# assert (wi_layout[0] x wi_layout[1] == subgroup_size) // PVC subgroup_size = 16
For matrix A load
#sg_map_a_bf16 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>    // WI data distribute from [8, 16] to [8, 1]
#sg_map_a_f16  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>    // WI data distribute from [8, 16] to [8, 1]
#sg_map_a_tf32 = xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 1]>     // WI data distribute from [8, 8] to [4, 1]
#sg_map_a_ui8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>    // WI data distribute from [8, 32] to [8, 2]
#sg_map_a_si8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>    // WI data distribute from [8, 32] to [8, 2]
For matrix B load
#sg_map_b_bf16 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>   // WI data distribute from [16, 16] to [8, 2]
#sg_map_b_f16  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>   // WI data distribute from [16, 16] to [8, 2]
#sg_map_b_tf32 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>   // WI data distribute from [8, 16] to [8, 1]
#sg_map_b_ui8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [4, 1]>   // WI data distribute from [32, 16] to [8, 4]
#sg_map_b_si8  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [4, 1]>   // WI data distribute from [32, 16] to [8, 4]
For matrix C load
#sg_map_c_f32  = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
#sg_map_c_si32 = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
For matrix load/store of any type
#sg_map_anytype = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
#sg_map_anytype = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // WI data distribute from [8, 16] to [8, 1]
For matrix load with transpose for A or B
#sg_map_at_tf32 = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>   // WI data distribute from [16, 8] to [8, 1]
```

The WI data distribution requires the following sg_map for the 2D block load and store to work with DPAS on ARC.
```mlir
# assert (wi_layout[0] x wi_layout[1] == subgroup_size) // ARC subgroup_size = 8
For matrix A load
#sg_map_a_bf16 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>   // WI data distribute from [8, 16] to [8, 2]
#sg_map_a_f16  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>   // WI data distribute from [8, 16] to [8, 2]
#sg_map_a_tf32 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>   // WI data distribute from [8, 8] to [8, 1]
#sg_map_a_ui8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>   // WI data distribute from [8, 32] to [8, 4]
#sg_map_a_si8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>   // WI data distribute from [8, 32] to [8, 4]
For matrix B load
#sg_map_b_bf16 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [2, 1]>    // WI data distribute from [16, 8] to [8, 2]
#sg_map_b_f16  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [2, 1]>    // WI data distribute from [16, 8] to [8, 2]
#sg_map_b_tf32 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>    // WI data distribute from [8, 8] to [8, 1]
#sg_map_b_ui8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [4, 1]>    // WI data distribute from [32, 8] to [8, 4]
#sg_map_b_si8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [4, 1]>    // WI data distribute from [32, 8] to [8, 4]
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
Reference of related spirv extension: [SPV_INTEL_2d_block_io](https://github.com/KhronosGroup/SPIRV-Registry/pull/305), [add SPV_INTEL_subgroup_matrix_multiply_accumulate](https://github.com/KhronosGroup/SPIRV-Registry/pull/306)

The sg_map required by DPAS operation can be propogated to 2D block load operations at subgroup level. During the propogation, the sg_map may be temporarily associated with vector/arith/math operations for the output vectors and removed after it is propogated to tensor_desc. The sg_map does not describe the data fragments after the tensor being loaded and packed to registers. Instead, It describes the data fragments and packing built upon the plain tensor layout.

Below are a few examples of DPAS's sg_map.
```mlir
  PVC BF16 example
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  #sg_map_c = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>

  // Before WI distribution: %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_c} :vector<8x16xbf16>, vector<16x16xbf16> into vector<8x16xfloat>
  %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_c} :vector<8x1xbf16>, vector<8x2xbf16> into vector<8x1xfloat>

  ARC int8 example
  #sg_map_a_ui8  = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>
  #sg_map_b_ui8 = xegpu.sg_map<wi_layout = [1, 8], wi_data = [4, 1]>
  #sg_map_c = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>

  // Before WI distribution: %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_c} :vector<8x32xui8>, vector<32x16xui8> into vector<8x16xi32>
  %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_c} :vector<8x4xui8>, vector<8x4xui8> into vector<8x1xi32>
```

The sg_map propagation process may encounter operations that require changing the mapping between the input and output operands. Specifically, the transpose operation swaps the wi_layout and wi_data to correctly track the data fragments affected by the transpose, and the bitcast operation adjusts the wi_data to reflect the correct number of data elements being packed.

```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 2]> //the producer op of vector_a uses #sg_map_a
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  //Before WI distribution:  %vector_b = vector.transpose [1, 0] %vector_a {#sg_map_b} :vector<16x16xbf16> into vector<16x16xbf16>
  %vector_b = vector.transpose %vector_a {#sg_map_b} :vector<8x2xbf16> into vector<8x2xbf16>

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]> //the producer op of vector_a uses #sg_map_a
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>
  //Before WI distribution:  %vector_b = vector.bitcast %vector_a {#sg_map_b} :vector<8x16xi16> into vector<8x32xi8>
  %vector_b = vector.bitcast %vector_a :vector<8x1xi16> into vector<8x2xi8>
```

Operations such as reduction and broadcast are exception to WI distribution rule. The 1d reduced vector doesn't participate in the WI distribution so that the dimension size doesn't change. The wi_layout must be [1, %subgroup_size], and wi_data must be [1, 1]. The SIMT distribution needs to use different dialect operations since the vector dialect can't express cross-lane semantics.  
```mlir

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]> //the producer op of vector_a uses #sg_map_a
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  //Before WI distribution:  %vector_b = vector.reduction [1] %vector_a {#sg_map_b} :vector<16x16xbf16> into vector<16xbf16>
  %vector_a' = reshape %vector_a :vector<16x1xbf16> into vector<16xbf16>
  %vector_b' = gpu.subgroup_reduce %vector_a' :vector<16xbf16> into vector<16xbf16>
  %vector_b = reshape %vector_b' :vector<16xbf16> into vector<16x1xbf16>

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]> //the producer op of vector_a uses #sg_map_a
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  //Before WI distribution:  %vector_b = vector.broadcast %vector_a {#sg_map_b} :vector<16xbf16> into vector<16x16xbf16>
  %vector_a' = reshape %vector_a :vector<16x1xbf16> into vector<16xbf16>
  %cst0 = arith.constant 0 : i32
  %vector_b', %p = gpu.shuffle idx %vector_a', %cst0, %subgroup_size: vector<16xbf16> into vector<16xbf16>
  %vector_b = reshape %vector_b' :vector<16xbf16> into vector<16x1xbf16>
```

Extract, insert and regular element-wise operations do not modify the sg_map. However, the result of SIMT distribution may have to use different operations.
```mlir
  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]> //the producer op of vector_a uses #sg_map_a
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  // Before WI distribution:  %vector_b = vector.extract %vector_a [0] {#sg_map_b} :vector<32x16xbf16> from vector<2x32x16xbf16>
  %vector_b = vector.extract_strided_slice  %vector_a { offsets = [0], sizes = [16], strides = [1]}: vector<16x2xbf16> from vector<32x2xbf16>

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]> //the producer op of vector_a uses #sg_map_a
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>
  //Before WI distribution:  %vector_b = vector.extract_strided_slice  %vector_a { offsets = [0], sizes = [32], strides = [1]}:  vector<32x16xbf16> from vector<64x16xbf16>
  %vector_b = vector.extract_strided_slice  %vector_a { offsets = [0], sizes = [16], strides = [1]}: vector<16x2xf16> from vector<32x2xbf16>

  #sg_map_a = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]> //the producer op of vector_a uses #sg_map_a
  #sg_map_b = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>
  #sg_map_c = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>
  //Before WI distribution:  %vector_c = arith.addf  %vector_a, %vector_b {#sg_map_c} :vector<24x32xbf16>
  %vec_a1, %vec_a2 = vector.deinterleave %vector_a : vector<24x2xbf16> to vector<24x1xbf16>
  %vec_b1, %vec_b2 = vector.deinterleave %vector_b : vector<24x2xbf16> to vector<24x1xbf16>
  %vec_c1 = vector.add  %vec_a1, %vec_b1 : vector<24x1xbf16>
  %vec_c2 = vector.add  %vec_a2, %vec_b2 : vector<24x1xbf16>
  %vector_c = vector.interleave %vec_c1, %vec_c2 :vector<24x1xbf16> to vector(24x2xbf16>
```

users must use for the WI data distribution of 1D block load and regular load with chunk_size on PVC and ARC. The rule of thumb is that the wi_data size must be 32bit, regardless on PVC or ARC.  Not using this sg_map defined here leads to undefined behavior.
```mlir
  For 1D block load
  # assert (wi_layout[0] x wi_layout[1] == subgroup_size) // PVC subgroup_size = 16
  #sg_map = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>  // for 32-bit data element
  #sg_map_t = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 2]>  // for 16-bit data element like bf16, f16
  #sg_map_t = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 4]>  // for 8-bit data element like uint8, sint8

  For regular load with chunk_size  // PVC subgroup_size = 16
  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]> // for 32-bit data element
  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 2]>  // for 16-bit data element like bf16, f16
  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 4]>  // for 8-bit data element like uint8, sint8

  For 1D block load
  # assert (wi_layout[0] x wi_layout[1] == subgroup_size) // ARC subgroup_size = 8
  #sg_map = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>  // for 32-bit data element
  #sg_map_t = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>  // for 16-bit data element like bf16, f16
  #sg_map_t = xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 4]>  // for 8-bit data element like uint8, sint8

  For regular load with chunk_size // ARC subgroup_size = 8
  #sg_map_t = xegpu.sg_map<wi_layout = [8, 1], wi_data = [1, 1]> // for 32-bit data element
  #sg_map_t = xegpu.sg_map<wi_layout = [8, 1], wi_data = [1, 2]>  // for 16-bit data element like bf16, f16
  #sg_map_t = xegpu.sg_map<wi_layout = [8, 1], wi_data = [1, 4]>  // for 8-bit data element like uint8, sint8
```

**sg_map use case - 2D load**

An example on how to load a 2D block, perform dpas, and store back to memory.

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

  %vector_c = xegpu.dpas %vector_a, %vector_b {#sg_map_a #sg_map_b_reg #sg_map_c} :vector<8x1xbf16>, vector<8x2xbf16> into vector<8x1xfloat>

  xegpu.store_nd %vector_c, %tdesc2:
          vector<8x1xfloat>, tensor_desc<8x16xfloat, #sg_map_c>

  %tdesc_updated = xegpu.update_nd_offset %tdesc, %offsets:2 :
  	  tensor_desc<8x16xbf16, #sg_map_a>, index, index into tensor_desc<8x16xbf16, #sg_map_a>

```

**sg_map use case - regular load**
An example on how to perform transpose using load with chunk_size in SIMT flavor.

```mlir

  #sg_map_t = xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>
  #scatter_attr = !xegpu.tdesc_attr< memory_space=slm, scattered=true>
  %scatter_tdesc_chunk = xegpu.create_tdesc, %src_addr, %offsets
		{chunk_size=4} :
		uint64, vector<16xindex> into tensor_desc<16x4xfp32, #scatter_attr, #sg_map_t>

  %result = xegpu.load %scatter_tdesc_chunk, %mask {L1 = cached, L2 = uncached, transpose=[1,0]} :
          tensor_desc<16x4xfp32, #tdesc_attr, #sg_map_t>, vector<16xi1> -> vector<4x1xfp32>

  #sg_map = xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>
  #tdesc_attr = !xegpu.tdesc_attr< memory_space=slm, boundary_check=false>
  %tdesc2 = xegpu.create_nd_tdesc %dest_addr, %offset:
		uint64, index into tensor_desc<64xfp32, #tdesc_attr>
  xegpu.store_nd %value, %tdesc2:
                vector<4xfp32>, tensor_desc<64xfp32, #tdesc_attr>

```

## extending sg_map attributes to support work group level semantic

By allowing XeGPU operating on workgroup level data size, it provides a concise IR for tensor compiler instead of multiple level nested loop IR for subgroup and work item level operation. To enable XeGPU operate the workgroup level, we introduce `sg_layout` and `sg_data` parameters into `sg_map` attribute to specify how the data is distributed across subgroups. With this extension, `sg_map` enables tensor compiler to express the cooperative operation among subgroups by specifying a `sg_map` to partition data among subgroups without manipulating a nested IR representation. The attribute allows tensor compiler to control the block size for both the workgroup and subgroup and perform autotuning as the number of subgroups, layout, and tensor size per subgroups are critical performance knobs.

we also propose to extend the sg_map to cover multiple dimension instead of 2D only and add `order` attribute. 

**Extended xegpu.sg_map**

The extended `sg_map` specifies how a n-d tensor (defined by the tensor descriptor) is partitioned among subgroup within a workgroup. sg_map consists of four parameters:
  * sg_layout: Defines the n-d arrangement of subgroups within the workgroup. The dimensions can be a 3d array as [dim_0, dim_1, dim_2]. 
  * sg_data: Specifies the shape of the tensor size for each subgroup after decomposition.
  * wi_layout: Defines the 2D arrangement of WIs within the subgroup.
  * wi_data: Specifies the shape of the tensor fragment that each WI loads or stores as a single packed data unit (16/32-bit).
  * order: The dimension order used to linearize n-d subgroup ids to 1-d. The first dimension in the order list is the fastest-changing dimension.

Example of linerized subgourp id regarding order[1, 0] vs. order [0, 1]. 
| sg_layout[4, 4] | order[1, 0] | order[0, 1]
| :----   | :----   |  :----   |
| [0, 0], [0, 1], [0, 2], [0, 3] | 0, 1, 2, 3 | 0, 4, 8, 12 |
| [1, 0], [1, 1], [1, 2], [1, 3] | 4, 5 , 6, 7| 1, 5, 9, 13 |
| [2, 0], [2, 1], [2, 2], [2, 3] | 8, 9, 10 , 11 | 2, 6, 10, 14 |
| [3, 0], [3, 1], [3, 2], [3, 3] | 12, 13, 14, 15 | 3, 7, 11, 15 |

For a subgroup in 3-d sg_layout [dim_0, dim_1, dim_2], order[2, 1, 0] maps a subgroup with 3-d index [x, y, z] to a linear subgroup index [z + dim_2 * y + dim_2 * dim_1 * x ], order[1, 2, 0] maps to [y + dim_2 * z + dim_2 * dim_1 * x].

When a sg_map attribute is attached to a tensor descriptor, load/store/dpas will operate at the workgroup level. The sg_map attribute must be specified when creating the tensor descriptor.

**Constraints**

Given these definitions:
```mlir
sg_data_size = sg_data[0] × sg_data[1]
workgroup_size = sg_layout[0] × sg_layout[1]
tensor_size = tensor_desc[0] × tensor_desc[1]
```

The following conditions must hold:

* workgroup_size must represent the number of subgroups in a workgroup for a kernel.
* tensor_desc[0] must be either evenly divisible by sg_layout[0] × sg_data[0], or vice versa.
* tensor_desc[1] must be either evenly divisible by sg_layout[1] × sg_data[1], or vice versa.

**distribution rule**

The tensor_desc is distributed to sg_data x sg_layout along each dimension in a round-robin fashion. If sg_data[i] x sg_layout[i] < tensor_desc[i], there is data left after all subgroups are assigned for the first round, the rest data will wrap around and be assigned to the first subgroup until the data is completely assigned. If sg_data[i] x sg_layout[i] > tensor_desc[i], the data may be used up before all subgroups are assigned. In this case, we broadcast the tensor data to multiple subgroups by repeating the data assignment to the rest subgroups along that dimension until the all subgroups get data.

**Resulting subgroup Data size**

The distributed tensor for each subgroup has the same dimension as the work group level tensor. 

**Examples of workgroup distribution with extended sg_map**

The workgroup creates a tensor descriptor [128, 128] and distributes to 4 subgroups with `sg_layout` [2,2], and each subgroup gets `sg_data` [32,128]. The first dimension is split and distributed to subgroups in two rounds, and the second dimension is assigned as whole to multiple subgroups.

```mlir
   #sg_map_a = #xegpu.sg_map<sg_layout = [2, 2], sg_data = [32, 128], wi_layout=[1,16], wi_data = [1, 1], order = [1, 0]>
   %wg_tdesc = xegpu.create_nd_tdesc %A[%m, %c0] : memref<1024x1024xf16> -> tensor_desc<16x128xf16, #sg_map_a>
```
The table below shows the result tensor for each subgroup and its linear subgroup id.

| subgroup tensor	| 2D subgroup id	| Linearized subgroup id
| :---   | :----   | :----   |
| [ 0:31, 0:127] | [0, 0], [0, 1] | 0 , 1 |
| [ 32:63, 0:127] | [1, 0], [1, 1] | 2 , 3 |
| [ 64:95, 0:127] | [0, 0], [0, 1] | 0 , 1 |
| [ 96:127, 0:127] | [1, 0], [1, 1] | 2 , 3 |

The `sg_map` attribute propagates from the matrix multiplication ops to other ops. Since we can't attatch the `sg_map` attribute to MLIR vector data type, we attach the attribute to vector type-based operations temporarily within the workgroup distribution pass. The `sg_map` attribute propagation can be performed from output to input, or the other direction. We describes below the propagation rules from output to input for typical operations including dpas, reduction, broadcast, shape_cast, and transpose.

For `dpas`, the `sg_map` attribute of input operands must have the same `sg_layout`, and `sg_data` for m and n dimension as output, and `sg_data` for k dimension must be same as operand A and B. `order` must be same as output.
```mlir
   #sg_map_d = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [32, 64], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]>
   %vector_d = xegpu.dpas %vector_a, %vector_b, %vector_c {#sg_map_d}:
     vector<256x256xfloat>, vector<256x32xbf16>, vector<32x256xbf16>
	   into vector<256x256xfloat>
   //derived sg_map for input operands
   #sg_map_a = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [32, 32], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]> //sg_map for %vector_a
   #sg_map_b = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [32, 64], wi_layout=[1,16], wi_data = [2, 1], order=[1, 0]> //sg_map for %vector_b
   #sg_map_c = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [32, 64], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]> //sg_map for %vector_c
```

For `reduction`,  `sg_map` of the input operand has an additional dimension to represent the dimension being reduced.  `sg_layout` must be the same and the new dimension as `1`. The new dimension of `sg_data` must be the same as the input tensor size, and the other dimension must be the same as the output's `sg_map`. The new dimension of `order` should not change the existing ordering specified by the output's `sg_map`.

```mlir
   #sg_map_a = #xegpu.sg_map<sg_layout = [32], sg_data = [8], wi_layout=[16], wi_data = [1], order=[0]>
   %vector_a = vector.multi_reduction <add> %vector_b, %cst_0 [1] {#sg_map_a}: vector<256x128xfloat> into vector<256xfloat>
   
   //derived sg_map for input operand
   #sg_map_b = #xegpu.sg_map<sg_layout = [32, 1], sg_data = [8, 128], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]>
```

The rule also applies to reduction from 3d to 2d.
```mlir
   #sg_map_a = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [1, 32], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]>
   %%vector_a = vector.multi_reduction <add>, %vector_b, %cst_0 [1] {#sg_map_a}: vector<8x32x128xf32> to vector<8x128xf32>

   //derived sg_map for input operand
   #sg_map_b = #xegpu.sg_map<sg_layout = [8, 1, 4], sg_data = [1, 32, 32], wi_layout=[1, 1, 16], wi_data = [1, 1, 1], order=[2, 1, 0]>
```

For `shape_cast`, it first determines the dimensions being reduced or expanded. The input's `sg_map` needs to expand or reduce the value accordingly for related dimension in `sg_layout` and `sg_data`. `order` should be consistent between input and output.
```mlir
   sg_map_a = #xegpu.sg_map<sg_layout = [8, 1, 4], sg_data = [1, 32, 32], wi_layout=[1, 1, 16], wi_data = [1, 1, 1], order=[2, 1, 0] >
   %vector_a = vector.shape_cast %vector_b {#sg_map_a} : vector<256x128xf32> to vector<8x32x128xf32>

   //derived sg_map for input operand
   #sg_map_b = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [32, 32], wi_layout=[1, 16], wi_data = [1, 1], order=[1, 0]>
```

For `broadcast`, `sg_map` of the input operand has one less dimension for the broadcast dimension. `sg_layout` for that dimension must be `1` in the ouptut sg_map and must be removed for the input operand. The corresponding dimension in `sg_data` and `order` must be removed also.

```mlir
   #sg_map_a = #xegpu.sg_map<sg_layout = [16, 1], sg_data = [16, 256], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]>
   %vector_a = vector.broadcast %vector_b [1] {#sg_map_a}: vector<256xfloat> into vector<256x256xfloat>

   //derived sg_map for input operand
   #sg_map_b = #xegpu.sg_map<sg_layout = [16], sg_data = [16], wi_layout=[16], wi_data = [1], order=[1, 0]>
```

For `transpose`, the values in `sg_map` must be swapped for the two dimensions being transposed, including `sg_layout`, `sg_data`, and `order`.
```mlir
   #sg_map_a = #xegpu.sg_map<sg_layout = [4, 8], sg_data = [32, 64], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]>
   %vector_a = vector.transpose %vector_b {#sg_map_a}: vector<512x128xfloat> into vector<128x512xfloat>

   //derived sg_map for input operand
   #sg_map_b = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [64, 32], wi_layout=[16, 1], wi_data = [1, 1], order=[0, 1]>
```

`sg_map` may be assinged for certain operation before the workgroup layout propagation, for example, the cooperative load pass may specify `sg_map` for certain load to be cooperated. In this case, the propagation may insert an operation to express the conversion of one `sg_map` to the other.

`convert_layout` is introduced to convert two inconsistent `sg_map`. 

```mlir
   #sg_map_b = #xegpu.sg_map<sg_layout = [8, 4], sg_data = [32, 64], wi_layout=[16, 1], wi_data = [1, 1], order = [1, 0]>  // used for cooperative load/prefetch
   #sg_map_a = #xegpu.sg_map<sg_layout = [32, 1], sg_data = [8, 256], wi_layout=[1,16], wi_data = [1, 1], order=[1, 0]> // used as mma's input matrix A
   %vector_a = xegpu.convert_layout %vector_b {#sg_map_a #sg_map_b}: vector<256x256xfloat> into vector<256x256xfloat>
```
The `sg_map` conversion can be lowered to storing and loading from the shared local memory. It can be conceptually viewed as a composition of two operations: 1) store the vector to shared local memory with the #sg_map_b and 2) use sg_map_a mapping to load the data from shared local memory.

## Appendix 1 - Code examples for work group level XeGPU using sg_map attribute

## Appendix 1.1 Simple Gemm with prefetch
The first example shows a simple gemm. It demonstrates the different sg_map we used for prefetch and load. The sg_map doesn't show wi_layout and wi_data for simplicity.
```mlir
Pseudo code for simple gemm
C[4096, 4096] = matmul (A[4096, 4096], B[4096, 4096])
```

```mlir
#mp_a     = #sg_map<sg_layout=[8,4], sg_data=[32,32], order=[1,0]>
#mp_a_pfh = #sg_map<sg_layout=[32,1], sg_data=[8,32], order=[1,0]>
#mp_b     = #sg_map<sg_layout=[8,4], sg_data=[32,64], order=[1,0]
#mp_b_pfh = #sg_map<sg_layout=[4,8], sg_data=[8,32], order=[1,0]>
#mp_c     = #sg_map<sg_layout=[8,4], sg_data=[32,64], order=[1,0]>

func.func @test_gemm(%a : memref<4096x4096xf16>,
       %b: memref<4096x4096xf16>,
       %c: memref<4096xf32> ) {
  scf.for %i = %c0 to %c4096 step %c256 {
    scf.for %j = %c0 to %c4096 step %c256 {
       %1 = create_nd_tdesc %a[%i, %c0] : memref<4096x4096xf16> -> tensor_desc<256x32xf16, #mp_a>   // sg_layout=[8,4], sg_data=[32,32]
       %2 = create_nd_tdesc %b[%c0, %j] : memref<4096x4096xf16> -> tensor_desc<32x256xf16, #mp_b> // sg_layout=[8,4], sg_data=[32,64]
       %1p = create_nd_tdesc %a[%i, %c96] : memref<4096x4096xf16> -> tensor_desc<256x32xf16, #mp_a_pfh]>  // sg_layout=[32,1]
       %2p = create_nd_tdesc %b[%c96, %j] : memref<4096x4096xf16> -> tensor_desc<32x256xf16, #mp_b_pfh> // sg_layout=[4,8]

       %3 = create_nd_tdesc %c[%i, %j] : memref<4096x4096xf32> -> tensor_desc<256x256xf32, #mp_c>           // sg_layout=[32, 1]

       scf.for %k= %c0 to %c4096 step %c32 {
           %4  = load_nd %1 : tensor_desc<256x32xf16  #mp_a > -> vector<256x32xf16>	             // sg_layout=[8,4], sg_data=[32,32]
           %10 = load_nd %2  : tensor_desc<32x256xf16 #mp_b> -> vector<32x256xf16>                // sg_layout=[8,4], sg_data=[32,64]
          
           prefetch_nd %1 : tensor_desc<256x32xf16, #mp_a_pfh>             			      // sg_layout=[32,1]
           prefetch_nd %2  : tensor_desc<32x256xf16, #mp_a_pfh>                                    // sg_layout=[4,8]
           %6 = dpas %4, %10 {#mp_a #mp_b #mp_c} : (vector<256x32xf16>, vector<32x256xf16>) -> vector<256x256xf32> //sg_layout=[8,4]
           %1 = update_nd_offset%1, %c0, %c32 :  tensor_desc<256x32xf16, #mp_a>
           %2 = update_nd_offset%2, %c32, %c0 :  tensor_desc<32x256xf16, #mp_b>
           %1p = update_nd_offset%1p, %c0, %c32 :  tensor_desc<256x32xf16, #mp_a_pft>
           %2p = update_nd_offset%2p, %c32, %c0 :  tensor_desc<32x256xf16, #mp_b_pft>
         } 
         store_nd %3, %6: (tensor_desc<256x256xf32, #mp_c>, vector<256x256xf32>)                    // sg_layout=[8, 4]
    } 
  }
```
## Appendix 1.2 Gemm with transpose, broadcast, and reduction
The second example contains transpose, broadcast, and reduction. The sg_map doesn't show wi_layout and wi_data for simplicity.
```mlir
Pseduo code for the original problem.
C[4096, 4096] = matmul (A[4096, 4096], BT[4096, 4096]) + broad_cast(bcast[4096], dim=0)
Reduce[4096] = reduce_add(C[4096, 4096], dim=1)
```

```mlir
#mp_a     = #sg_map<sg_layout=[8,4], sg_data=[32,32], order=[1,0]>
#mp_a_pfh = #sg_map<sg_layout=[32,1], sg_data=[8,32], order=[1,0]>
#mp_b     = #sg_map<sg_layout=[8,4], sg_data=[32,64], order=[1,0]>
#mp_bt    = #sg_map<sg_layout=[4,8], sg_data=[64,32], order=[0,1]>
#mp_bt_pfh = #sg_map<sg_layout=[32,1], sg_data=[8,32], order=[1,0]>
#mp_c     = #sg_map<sg_layout=[8,4], sg_data=[32,64], order=[1,0]>

#mp_bcast = #sg_map<sg_layout=[8, 4], sg_data=[1,64], order=[1,0]>
#mp_bcast2 = #sg_map<sg_layout=[32], sg_data=[8], order=[0]>
#mp_reduce= #sg_map<sg_layout=[32], sg_data=[8], order=[0]>
#mp_reduce2= #sg_map<sg_layout=[32, 1], sg_data=[8, 256], order=[1,0]>

func.func @test_gemm(%a : memref<4096x4096xf16>,
       %b: memref<4096x4096xf16>,
       %bcast: memref<4096xf32>
       %res: memref<4096xf32> ) {
  scf.for %i = %c0 to %c4096 step %c256 {
    scf.for %j = %c0 to %c4096 step %c256 {
       %1 = create_nd_tdesc %a[%i, %c0] : memref<4096x4096xf16> -> tensor_desc<256x32xf16, #mp_a>   // sg_layout=[8,4], sg_data=[32,32]
       %2 = create_nd_tdesc %bt[%j, %c0] : memref<4096x4096xf16> -> tensor_desc<256x32xf16, #mp_bt> // sg_layout=[4,8], sg_data=[64,32]
       %1p = create_nd_tdesc %a[%i, %c192] : memref<4096x4096xf16> -> tensor_desc<256x32xf16, #mp_a_pfh]>  // sg_layout=[32,1]
       %2p = create_nd_tdesc %bt[%j, %c192] : memref<4096x4096xf16> -> tensor_desc<256x32xf16, #mp_bt_pfh> // sg_layout=[32,1]

       %bcast'= memref.cast %bcast: memref<4096xf32> -> memref<1x4096xf32>
       %7 = create_nd_tdesc %bcast'[%j] : memref<1x4096xf32> -> tensor_desc<1x256xf32, #mp_bast>           // sg_layout=[4, 8], sg_data=[1,32]

       %res'= memref.cast %res: memref<4096xf32> -> memref<4096x1xf32>
       %3 = create_nd_tdesc %res'[%i] : memref<4096x1xf32> -> tensor_desc<256x1xf32, #mp_reduce>           // sg_layout=[32, 1]

       scf.for %k= %c0 to %c4096 step %c32 {
           %4  = load_nd %1 : tensor_desc<256x32xf16  #mp_a > -> vector<256x32xf16>	             // sg_layout=[8,4], sg_data=[32,32]
           %10 = load_nd %2  : tensor_desc<256x32xf16 #mp_bt> -> vector<256x32xf16>               // sg_layout=[4,8], sg_data=[64,32]
           %5  = vector.transpose %10 {#mp_bt #mp_b}: vector<256x32xf16> -> vector<32x256xf16>   // sg_layout=[4,8] -> sg_layout=[8,4]

           prefetch_nd %1 : tensor_desc<256x32xf16, #mp_a_pfh>             			      // sg_layout=[32,1]
           prefetch_nd %2  : tensor_desc<256x32xf16, #mp_a_pfh>                                    // sg_layout=[32,1]
           %6 = dpas %4, %5 {#mp_a #mp_b #mp_c} : (vector<256x32xf16>, vector<32x256xf16>) -> vector<256x256xf32> //sg_layout=[8,4]
           %1 = update_nd_offset%1, %c0, %c32 :  tensor_desc<256x32xf16, #mp_a>
           %2 = update_nd_offset%2, %c0, %c32 :  tensor_desc<256x32xf16, #mp_bt>
           %1p = update_nd_offset%1p, %c0, %c32 :  tensor_desc<256x32xf16, #mp_a_pft>
           %2p = update_nd_offset%2p, %c32, %c0 :  tensor_desc<256x32xf16, #mp_bt_pft>
         } 

         %12  = load_nd %7  : tensor_desc<256xf32, #mp_bcast2> -> vector<256xf16>                // sg_layout=[32], sg_data=[8]
         %12' = convert_layout {#mp_bcast2 #mp_bcast} %12 :  vector<256x256xf32>		 // sg_layout=[32] -> sg_layout=[8, 4]
         %13 = vector.broadcast {#mp_c} %12' [0]: vector<256xf32> => vector<256x256xf32>   	 // sg_layout=[8, 4], sg_data=[32,64]
         %14 = add %6, %13 : vector<256x256xf32>
         %14' = convert_layout {#mp_c #mp_reduce2} %14 :  vector<256x256xf32>				   // sg_layout=[8, 4] -> sg_layout=[32]
         %16 = vector.reduction {#mp_reduce2 #mp_reduce} <add> %14' [1]: vector<256x256xf32> => vector<256xf32>  // sg_layout=[32]
         store_nd %3, %7: (tensor_desc<256xf32, #mp_reduce>, vector<256xf32>)                               // sg_layout=[32]
    } 
  }
```

## Appendix 1.3 Gemm implementation with two cache levels
For GPU support high-performance prefetch through two level of caches.
```mlir
#mp_a = #sg_map<sg_layout=[8,4], sg_data=[64,32], order=[1,0]>
#mp_b = #sg_map<sg_layout=[8,4], sg_data=[32,64], order=[1,0]>
#mp_c = #sg_map<sg_layout=[8,4], sg_data=[64,64], order=[1,0]>

#mp_a_copl2 = #sg_map<sg_layout=[32,1], sg_data=[16,128], order=[1,0]>
#mp_b_copl2 = #sg_map< sg_layout=[16,2], sg_data=[8,128], order=[1,0]>

#mp_a_copl1 = #sg_map<sg_layout=[32,1], sg_data=[16,32], order=[1,0]>
#mp_b_copl1 = #sg_map< sg_layout=[4, 8], sg_data=[8,32], order=[1,0]>

func.func @test_gemm(%a : memref<4096x4096xf16>,
       %b: memref<4096x4096xf16>,
       %c: memref<4096xf32> ) {
   scf.for %i = %c0 to %c4096 step %c256 {
     scf.for %j = %c0 to %c4096 step %c256 {
        %a1_l2 = create_nd_tdesc %a[%i, %c0] : memref<4096x4096xf16> -> tensor_desc<512x128xf16, #mp_a_copl2>
        %b1_l2 = create_nd_tdesc %b[%c0, %j] : memref<4096x4096xf16> -> tensor_desc<128x256xf16, #mp_b_copl2>
        %a2_l2 = create_nd_tdesc %a[%i, %c256] : memref<4096x4096xf16> -> tensor_desc<512x128xf16, #mp_a_copl2>
        %b2_l2 = create_nd_tdesc %b[%c256, %j] : memref<4096x4096xf16> -> tensor_desc<128x256xf16, #mp_b_copl2>

        prefetch_nd %a1_l2 locality<2>: tensor_desc<512x128xf16, #mp_a_copl2>
        prefetch_nd %b1_l2 locality<2>: tensor_desc<128x256xf16, #mp_b_copl2>
	prefetch_nd %a2_l2 locality<2>: tensor_desc<512x128xf16, #mp_a_copl2>
        prefetch_nd %b2_l2 locality<2>: tensor_desc<128x256xf16, #mp_b_copl2>
        %a2_l2’ = update_nd_offset%a2_l2, %c0, %c32 :  tensor_desc<512x128xf16, #mp_b_copl2>
        %b2_l2’ = update_nd_offset%b2_l2, %c32, %c0 :  tensor_desc<128x256xf16, #mp_b_copl2>

        %a1_l1 = create_nd_tdesc %a[%i, %c0] : memref<4096x4096xf16> -> tensor_desc<512x32xf16, #mp_a_copl1>
        %b1_l1 = create_nd_tdesc %b[%c0, %j] : memref<4096x4096xf16> -> tensor_desc<32x256xf16, #mp_b_copl1>
        %a2_l1 = create_nd_tdesc %a[%i, %c32] : memref<4096x4096xf16> -> tensor_desc<512x32xf16, #mp_a_copl1>
        %b2_l1 = create_nd_tdesc %b[%c32, %j] : memref<4096x4096xf16> -> tensor_desc<32x256xf16, #mp_b_copl1>
        %a3_l1 = create_nd_tdesc %a[%i, %c64] : memref<4096x4096xf16> -> tensor_desc<512x32xf16, #mp_a_copl1>
        %b3_l1 = create_nd_tdesc %b[%c64, %j] : memref<4096x4096xf16> -> tensor_desc<32x256xf16, #mp_b_copl1>
        %a4_l1 = create_nd_tdesc %a[%i, %c96] : memref<4096x4096xf16> -> tensor_desc<512x32xf16, #mp_a_copl1>
        %b4_l1 = create_nd_tdesc %b[%c96, %j] : memref<4096x4096xf16> -> tensor_desc<32x256xf16, #mp_b_copl1>

        prefetch_nd %a1_l1 locality<3>: tensor_desc<512x32xf16, #mp_a_copl1>
        prefetch_nd %b1_l1 locality<3>: tensor_desc<32x256xf16, #mp_b_copl1>
        prefetch_nd %a2_l1 locality<3>: tensor_desc<512x32xf16, #mp_a_copl1>
        prefetch_nd %b2_l1 locality<3>: tensor_desc<32x256xf16, #mp_b_copl1>
        prefetch_nd %a3_l1 locality<3>: tensor_desc<512x32xf16, #mp_a_copl1>
        prefetch_nd %b3_l1 locality<3>: tensor_desc<32x256xf16, #mp_b_copl1>
        prefetch_nd %a4_l1 locality<3>: tensor_desc<512x32xf16, #mp_a_copl1>
        prefetch_nd %b4_l1 locality<3>: tensor_desc<32x256xf16, #mp_b_copl1>
        %a4_l1’ = update_nd_offset% a4_l1, %c0, %c128 :  tensor_desc<512x32xf16, #mp_a_copl1>
        %b4_l1’ = update_nd_offset% b4_l1, %c128, %c0 :  tensor_desc<32x256xf16, #mp_b_copl1>

        %a1_load = create_nd_tdesc %a[%i, %c0] : memref<4096x4096xf16> -> tensor_desc<512x32xf16, #mp_a>
        %b1_load = create_nd_tdesc %b[%c0, %j] : memref<4096x4096xf16> -> tensor_desc<32x256xf16, #mp_b>

        %c_tile = create_nd_tdesc %c[%i, %j] : memref<4096x4096xf32> -> tensor_desc<512x256xf32, #mp_c>

        scf.for %k= %c0 to %c4096 step %c32 {
            %a1_r = load_nd %a1_load : tensor_desc<256x32xf16  #mp_a > -> vector<512x32xf16>
            %b1_r = load_nd %b1_load  : tensor_desc<32x256xf16 #mp_b> -> vector<32x256xf16>

            Scf.if (%k %4 == 0) {
                gpu.barrier
                prefetch_nd %a2_l2’ locality<2>: tensor_desc<512x128xf16, #mp_a_copl2>
                prefetch_nd %b2_l2’ locality<2>: tensor_desc<128x256xf16, #mp_b_copl2>
                %a2_l2’ = update_nd_offset%a2_l2’, %c0, %c128 :  tensor_desc<512x128xf16, #mp_a_copl2>
                %b2_l2’ = update_nd_offset%b2_l2’, %c128, %c0 :  tensor_desc<128x256xf16, #mp_b_copl2>
            }
            prefetch_nd %a4_l1’ locality<3>: tensor_desc<512x32xf16, #mp_a_copl1>
            prefetch_nd %b4_l1’ locality<3>: tensor_desc<32x256xf16, #mp_b_copl1>
            %a4_l1’ = update_nd_offset%a4_l1’, %c0, %c32 :  tensor_desc<512x32xf16, #mp_a_copl1>
            %b4_l1’ = update_nd_offset%b4_l1’, %c32, %c0 :  tensor_desc<32x256xf16, #mp_b_copl1>

            %a1_load = update_nd_offset%a1_load, %c0, %c32 :  tensor_desc<512x32xf16, #mp_a>
            %a2_load = update_nd_offset%b1_load, %c32, %c0 :  tensor_desc<32x256xf16, #mp_b>

            %6 = dpas %a1_r, %b1_r {#mp_a #mp_b #mp_c} : (vector<512x32xf16>, vector<32x256xf16>) -> vector<512x256xf32>
        }
       store_nd %c_tile, %6: (tensor_desc<512x256xf32, #mp_c>, vector<512x256xf32>)
     }
   }
}
```

## Notes

Currently, there is no lower-level dialect for the Intel GPU compiler toolchain to represent GPU ops with values based on LLVM data types such as NVVM
dialect for the Nvidia GPU compiler toolchain. XeGPU dialect uses LLVM or SPIR-V intrinsic to access advanced intel GPU instructions. When the lower-level
software changes, we expect XeGPU lowering passes to change accordingly.
