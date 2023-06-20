# RFC for XeTile and XeGPU Dialect

## Summary
Lowering GEMM (General matrix multiplication) to an efficient nested loop is a complicated task, with multiple factors determining the efficiency. After decomposing the task into workgroup kernels and further down to subgroup kernels, each subgroup kernel executes a GEMM operation on submatrices. Generating efficient code for GEMM requires a good decomposition that creates enough subtasks to drive high core utilization and large enough subgroup-level submatrix size for code efficiency. On top of this, each hardware has its own recipe for the best code sequence for sub-group level GEMM, which contains both common target-independent techniques and target-specific optimizations. 

This RFC propose XeTile and XeGPU Dialect to support effecitient code generation for Xe GPU. 

## Motivation

To facilitate efficient code generation for GEMM, we introduce two new dialects, XeTile and XeGPU dialects. XeTile dialect supports the tile-based programming model and decomposes the GEMM kernel to a large enough tile size at the subgroup level.  Users can use the XeTile dialect to build a subgroup-level microkernel that implements batch-reduced gemm, using the best-known recipe for specific hardware. The recipe at this level includes target-independent optimizations like cooperative prefetch, cooperative load, K-slicing, and software pipelining. Users can further perform optimization like fusing with neighbor operations.  

The XeTile dialect works as the lowest abstraction layer which hides the matrix hardware difference between different GPU micro-architectures by working at tiles with larger size than the underneath hardware support. With the XeTile dialect, the lowering pass can set up hardware 2D block loader to autopad the out-of-boundary access.  XeTile dialect also abstracts out the hard limitations so that it can support arbitrary input matrix sizes.  When the input matrix sizes don’t meet 2D block load requirements, the lowering pass implements 2D tile load using 1d load and scalar load with target-specific recipes. 

The XeGPU dialect provides 1:1 mapping to match Xe instructions like DPAS and 2D block load. The matrix size being processed at this level exactly match the hardware instructions or the intrinsic supported by the lower-level GPU compiler.  All the optimizations built on top of XeGPU dialect is target specific. One optimization is to decompose a large contiguous 2D tile to several smaller tiles, to help lower-level compiler to provide better register allocation. After lowering to XeGPU dialect, user could estimate the total register size used by the sub-group level GEMM assuming the lower-level compiler does proper register allocation to facilitate the GEMM decomposition. 

## High level Design Consideration

The XeTile dialect is designed to support the lowering from workgroup level device function to subgroup (warp counterpart) level device function. It supports cooperative prefetching, loading the 2D tile to registers, doing matrix multiplication, and storing it back.  Users may use XeTile to implement a higher-level microkernel like BRGemm. It works on a 2D tile (like 64x64) so that it could be lowered to the best-known code sequence like a lowest-level microkernel. 

The XeTile dialect is designed to complement the MLIR Vector dialect. It is target-independent and could be assimilated into the Vector dialect in the future. The vector dialect is used to support similar tasks but covers a broad range of vectors like 1-D and n_D. The XeTile dialect limits the vector to be 2-D and also extends the Vector dialect semantics. It introduces a Tile data type, which combines the base tile info with memory description, and supports functionality like load/store/prefetch Tiles to vectors. 

The XeTile dialect design is strongly influenced by the XeTLA::tile API, and the lowering to XeGPU dialect should closely follow XeTLA::tile implementation.  It supports the efficient mapping of tile-based programming model to PVC’s 2D block loader. The implementation should recommend subgroup-level tile sizes for best performance and optionally code efficiency for the subgroup tile size (could be at the BRGEMM level).  The XeTile supports arbitrary input matrix sizes, so that user don’t need to deal with out-of-boundary cases and hardware limitations.  The support details are summarized below. 
1.	Support the input matrix size well divisible by the subgroup level tile size. There is no out-of-boundary cases. 
2.	Support the input matrix size not divisible by the subgroup level tile size, but the matrix size still meets the 2D block load requirements.  The compiler-generated code set up the 2D block loader hardware to detect the out-of-boundary and perform padding automatically.
3.	Support the row dimension of the subgroup tile for matrix A to be not multiple of DPAS HW size (say 8 on PVC).  This gives some flexibility in subgroup-level tile size so gives more flexibility to GEMM decomposition. It handles the remaining rows automatically, if it is not a multiple of 8 in the memory, it is loaded to registers as 8 rows (or multiple of 8 rows). 
4.	Support the input matrix with an unaligned shape. For example, in case the size (in byte) of N, K dims are not divided by 8, then the micro-kernel uses 1d load and scalar load and does the software padding. 
5.	Optimize the preparation of the 2D block load address (AKA payload. The initial address setup is more expensive since it involves the base tile (surface) sizes, tile offsets and sizes. The initial address preparations could be hoisted out of loop, so only offset update is performed within the loop. 
6.	Tile prefetch support cooperative prefetch. for example, if total subgraph thread number is 8, then 8 subgroup threads to cooperatively fetch the tile A. Subgroup id and total thread number is passed to tile prefetch to do the split. 

The XeTile dialect hides the VNNI layout and exposes the register block layout to the user. This diverges from the XeTLA::tile API design. For a DPAS instruction with input matrix_A[8, 16] and matrix_B[16, 16], the input matrix need to re-layout to be matrix_A[8, 16/2, 2] and matrix _B[16/2, 16, 2].  This causes a real physical layout difference for the matrix_B. However, since this layout is very short-lived and only affects the loading of matrix_B and DPAS instruction, the XeTile hides the VNNI layout from user and expects the vector re-layout operations are inserted automatically in the lowering process. The XeGPU dialect supports load instruction with VNNI transformation capability so a target-specific optimization pass could merge the re-layout to the 2D load operation. 

The XeTile dialect requires user to specify a register block size for tile load operation. The tile is loaded to a high dimension vector with blocked layout, which represents how the tile is loaded to registers. Each XeGPU register can be very large so it can hold the entire input matrix to a DPAS instruction.  So conceptually the 2D tile is loaded into a 2D array of registers, each with the size of register block.  For example, the tile for submatrix A[32, 64] is loaded to a register region with a blocked layout [32/8, 64/16, 8, 16].  Exposing the inner-block allows gradual lowering. Since the pre-processing of matrix A/B and post-processing of matrix C needs to represented on the vectors in blocked lyout vectors, so the XeTile-based program is closer to lowered form and help understand the lowering. 

The XeGPU dialect models the Xe ISA semantics but works at vector and tile data type, so it works like a bridge dialect before it is lowering to LLVM/SPIRV which works on llvm.struct or spriv.vector. The scope of XeGPU dialect includes the following. 
1.	DPAS  
2.	2D block load/store/prefetch
3.	1D load/store/prefetch, scalar load/store/prefetch

Since there are existing dialects operating on vector data type, including vector, arith, and math. XeGPU dialect doesn’t intend to include full Xe ISA instructions but to include all the target-specific operations beyond the scope of existing dialects. Together with vector/arith/math dialects, it supports the following features: 
1.	Register region access, to access register subblock from a larger register block 
2.	Barrier, fence, atomic operations
3.	Data type conversion, to support mixed data type 
4.	Math operations 


## Proposal
### XeTile Dialect

The XeTile dialect exploits the hardware features to do the auto-padding, which provides a simple and efficient generated code than the software padding. Using the tile dialect, user don’t need to detect the out-of-boundary case, and also the dialect takes care of unaligned shape, so the same code runs for the unaligned use case.  Users can focus on high-level optimization like software pipelining, cooperative prefetch, and K-slicing. 

To create a 2D Tile memory descriptor, user needs to set up a tile with memref and its “base tile” information. The memref and its “base tile” must be 2D and must be contiguous. With the tile abstraction, XeTile does load_tile, prefetch_tile, and store_tile.  The XeTile only takes the base memref, offsets, sizes, and stride. Offsets and sizes describe the tile along the row and column, in the number of elements. Stride describe the number of elements between the two elements along the leading dimension.  The innermost dimension must be contiguous. The current version only supports the 2D memref has row-major layout.  

```mlir 
  %tile = XeTile.init_tile %base_memref, %offset:2, %sizes:2, %stride
     memref<64x64xbf16>, index, index, index, index, index
     into tile<64x64xbf16, index, index, index, index, index>
```
init_tile is similar to memref.subview in terms of setting up a memory region out of a larger memory region. However, the subveiw leaves the handling of out-of-bounds access to the user to handle, but the tile hides the details from the user. The lowering pass could lower the subview to init_tile using the base tile info extracted “extract_strided_metadata”. 
Init_ccop_tile splits a tile into multiple smaller tile so that the current thread can cooperatively work on the tile with the neighbour subgroup thread within the same workgroup. This can be used by cooperative prefetch and load on matrix A and B, where subgroup shares the memory access with its peer subgroups.  

```mlir 
  %tile = XeTile.init_coop_tile %tile, %coop_id, %coop_size
     tile<64x64xbf16, index, index, index, index, index>, uint_32, uint_32
     into tile<8x8xbf16, index, index, index, index, index>
```

load_tile loads a tile to register region. Optionally with blocked layout, which was represented with the high dimension vector.  The blocking don’t changes the order of the outer dimension, so for vector [m, n] with blockfactor [MB, NB], the register region has layout as [m/MB, n/NB, MB, NB].  

```mlir 
  %vector_a = XeTile.load_tile %tile_a   inner_blocks = [8, 16]
     tile<64x64xbf16, index, index, index, index, index> 
		into vector <8x4x8x16xb16>
```

load_tile supports transpose. The inner_blocks describe the memory layout before the transpose. On the backward path, one of input matrices needs to be transposed for matmul operation. 
```mlir 
  %vector_a = XeTile.load_tile %tile_a   inner_blocks = [8, 16] Transpose = TRUE
     tile<64x64xbf16, index, index, index, index, index> 
		into vector <4x8x16x8xbf16>
```

These load_tile variants need to be used together with the tile_mma.  The VNNI layout transformation is not exposed to tile dialect users.  A lowering pass will add the VNNI transformation at the XeGPU dialect. 
store_tile stores the blocked register region back to memory in plain layout. VNNI_transform and transpose are not supported. 
```mlir 
  XeTile.store_tile %tile_a  %vector_a  inner_blocks = [8, 16]  
	   vector <8x4x8x16> 
     into tile<64x64xbf16, index, index, index, index, index> 
```

coop_prefetch_tile prefetches the tile to cache.  

```mlir 
  XeTile.prefetch_tile %coop_tile_a  
     tile<8x8xbf16, index, index, index, index, index> 
```

load_tile can be used to load the tile to registers in a cooperative flavor. The coop_tile is created out of tile and is a tile with a smaller size. So the load operation is the same.  

```mlir 
  %a = XeTile.load_tile %coop_tile_a  
     tile<16x16xbf16, index, index, index, index, index>  
	   into Vector <16x16xbf16>
```

Once with the block-layout register region, tile_mma represents the matrix multiplication on 4 D vectors. The semantics can be represented by vector.contract, so tile_mma works like a syntax sugar. This also means that the code can be mapped to HW without DPAS support nicely.  

```mlir 
  %vector_c = XeTile.tile_mma %vector_a, %vector_b   
     vector <8x4x8x8xbf16>, vector<4x8x8x16xbf16>
	   into vector <8x8x8x16float>  
```

Tile can be updated using the offset_x and offset_y.  These operations are used when one tile is being processed and moved to a new tile. Usually only one value is needed to update since the tile is only moving along the K dimension. 

```mlir
  XeTile.update_tile_offset %tile, %offset_x, offset_y
		tile<64x64xbf16, index, index, index, index, index>, index
```

### XeGPU dialect
The XeGPU dialect models Xe GPU’s ISA but works on MLIR vector type and memref-based tile type as a bridge dialect.  XeGPU operations are only introduced only when vector-based dialects (vector/math/arith) can’t express the operation semantics. This is consistent with the design NV and AMD does with NVGPU ( alink) and AMDGPU (alink) dialects, which work with vector and memref type.  

The XeGPU dialect supports XeTile dialects lowering, so the tile-based XeTile operation can be further decomposed to many XeGPU ops. Besides tile-based operation, it also includes 1D load and scatter load.  For tile-based operation, the XeGPU dialect reuses the tile data type so it looks similar the to XeTile dialect. Compared with the XeTile dialect, the XeGPU dialect works at smaller tile size and lower-dimension vectors and expects each operation to be mapped to one instruction underneath. 

Instead of load, store, and prefetch tile, the XeGPU dialect offers load_2D, store_2D, prefetch_2D operation on a 2D block. We reuse tile to describe the 2D block, since it is essentially a tile but with a smaller size to fit exactly one Xe ISA instruction. 

```mlir
  %block = XeGPU.init_tile %base_memref, %offset:2, %sizes:2, %strides
     memref<8x16xbf16>, index, index, index, index, index
     into tile<8x16xbf16, index, index, index, index, index>
```

load_2D loads a 2D block from global memory, represented by tile, to registers, represented by a vector. The vector data used in the XeGPU is expected to exactly describe the data layout and establish exact mapping to physical registers. The vector data used in the XeTile is a conceptual vector, which doesn’t reflect the exact data layout in the physical registers. For example, the vector A[4, 8, 16, 8] used in the XeTile would be lowered to 32 separate smaller vectors with a VNNI data layout, like a_vnni[8, 8, 2]. The a_vnni would be mapped to registers and used exactly as is to feed to DPAS instruction. 

```mlir 
  %a = XeGPU.load_2D %block 
     tile<8x16xbf16, index, index, index, index, index> 
		into vector<8x16xbf16>
```

load_2D supports VNNI transform for low-precision data type like fp16, bf16, int8, and int4. VNNI transformation takes a number of low-precision data and fits them into 32-bit data. For example, it takes 2 bf16 or 4 int8 values from matrix A’s row dimension, or B’s column dimension and puts them into the innermost dimension.  The VNNI transformation doesn’t change the overall size but splits the dimension to 3. 
Load_2D doesn’t support converting the VNNI layout back to a non-VNNI layout. The VNNI layout is supposed to be applied to the weight matrix only for the DPAS operation, and this is only the use and no need to convert the layout back.  

```mlir
  %bt = XeGPU.load_2D %block_b   VNNI_AXIS = 1 
     tile<16x16xbf16, index, index, index, index, index> 
		into vector <8x16x2xbf16>

  %at = XeGPU.load_2D %block_a   VNNI_AXIS = 0 (default)
     tile<8x16xbf16, index, index, index, index, index> 
		into vector <8x8x2xbf16>
```

The variant of VNNI transformation has no impact on the physical data layout, so it can be optimized away from code sequence. However, it does contribute to the code readability, since it is much easier to understand that A[8, 8, 2]  x B[8, 16, 2], vs. A[8, 16] x B[8, 16, 2]. 
load_2D supports transpose. This is to support tile transposition. The operation definition supports all data types, but hardware may have limitations. For example, PVC only supports data types with 4-byte (DW) and 8-byte (DQ). 
There is a use case to transpose a low-precision matrix on the backward path. In this case, the lowering path on PVC would need to use an alternative XeGPU code sequence to emulate the effects. 

```mlir
  %at = XeGPU.load_2D %block_a   Transpose = TRUE
     tile<8x16xf32, index, index, index, index, index> 
		into vector <16x8xf32>
```

load_2d supports VNNI transformation and transpose combined for low-precision data type like fp16, bf16, int8, and int4. The operation definition supports VNNI_AXIS to be both row and column, but it is only supported when VNNI_AXIS = 1 along column dimension on PVC. The example below shows that a bf16 matrix [8row, 16col] is transposed to [16col, 8row], and then VNNI transform to [8col, 8row, 2col]. In PVC hardware, it is fulfilled by a hardware transpose on DW data type, the bf16 [8row, 16col] is viewed as DW [8row, 8col], and transpose to DW[8col, 8row], and then can be viewed as [8col, 8row, 2col]. 

```mlir
  %at = XeGPU.load_2D %block_a   Transpose = TRUE   VNNI_AXIS = 1
     tile<8x16xbf16, index, index, index, index, index> 
		into vector <8x8x2bf16>
```

Dpas operation. The dimension of the vector is reduced to 3 dimensions and fits the hardware directly.  When the input matrix is lowa -precision data type (lower than 32bit), the matrix B must be in VNNI layout, meaning the reduction axis needs to be split into 2 axis and the inner dimension has multiple data elements fitting the 32bit. 

```mlir
  %vector_c = XeGPU.dpas %vector_a, %vector_b   
     vector <8x8x2xbf16>, vector<8x16x2xbf16>
	   into vector <8x16xfloat>   
```

store_2D stores the blocked register region back to memory in plain layout. VNNI_transform and transpose are not supported. 

```mlir
  XeGPU.store_2D %tile_a  %vector_a 
	   vector <8x16> 
     into tile<8x16xbf16, index, index, index, index, index> 
```

prefetch_2D prefetches a 2D block to cache.  

```mlir
  XeTile.prefetch_2D %block_a  
     tile<64x64xbf16, index, index, index, index, index> 
```

Due to hardware limitation, XeGPU’s load_2D operation can’t fully support XeTile to load arbitrary 2D matrix size, the lowering pass maps the XeTile’s load_tile to load_1d and load_scalar. 
Load_1d, store_1d load a 1d vector using a memory pointer and size.  User can use load_tile to load a vector with shape [1, x]. However, on PVC, this gives lower efficiency than 1d block load since the maximum of x is limited to 64 bytes. 

```mlir
 %result = XeGPU.load_1d %base, %offset: uint64, uint64 into vector<8xf32>
 XeGPU.store_1d %value, %base, %offset: vector<8xf32>, uint64, uint64
```

There is a counterpart in vector dialect like 1d vector load and store, which can be lowered to XeGPU’s load_1d and store_1d. 

```mlir
 %result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<8xf32>
 vector.store %value, %memref[%i, %j]: vector<8xf32>, memref<200x100xf32>
```

load_scalar, store_scalar
scatter loader is the most flexible load. It can be used to load unaligned cases. Each WI thread can read multiple data elements.  On PVC, it is up to 4 or 8 DW. Scatter load allows each WI thread to use different addresses, or addresses with strides. However, it is not common that deep learning use case requires strided load at the innermost dimension. 

```mlir
%result = XeGPU.load_scalar %base, %offset, %mask, %mask_val: 
                             uint64, uint64, i1, any, into any
XeGPU.store_scalar %value, %base, %offset, %mask: 
                         any, uint64, uint64, i1
```

There is a counterpart in vector dialect like 1d vector.gather and vector.scatter, which can be lowered to XeGPU’s load_scalar and store_scalar. 

```mlir
%0 = vector.gather %base[%c0][%v], %mask, %pass_thru
   : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>

vector.scatter %base[%c0][%v], %mask, %value
    : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
```


## Alternative

Alternative design is to not introduce XeTile dialect. Instead we just provide XeGPU dialect. The benefit of XeTile dialect support gradual lowering and allow some ciritical optimization. For example, the XeTile dialect could be used to support microkenrel level interface like BRGEMM to choose the best the subgroup tile size. This XeTile dialect suggests a list of subgroup tile size with good performance, and it also allows more choices of decomposition by support row number not divisble by DPAS size (8). It can assist the microkenrel to estimate the register use and code efficiency at the subgroup level, which can eventually contribute to searchign for an optimized GEMM decomposition.  The other example is that the 2d load block size could be larger than the DPAS size for best code efficiency. Having XeTile working at the same size and allowing the lowering pass to handle the size differnce works better than leaving all these level of details to the lowering from vector dialect at subgroup level.  The VNNI layout transform is another example which XeTile helps to hide the low level detail.  

## Questions

Currently there is no NVVM counterpart. XeGPU dialect uses SPIRV Intel extension to access joint-matrix or SPRIV external function to access intel GPU VC intrinsics. This may change in the future, so we expect XeGPU dialect may have changes. 
