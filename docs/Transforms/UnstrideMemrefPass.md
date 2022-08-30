# UnstrideMemref Pass


This pass is used for handling memref with ranks greater than 1.
In MLIR, Memrefs are lowered in various paths. Two of them are of importance to us at this point: 
Lowering in CPU (via LLVM dialect)
Lowering in GPU (via spv dialect)

In the CPU path, LLVM dialect lowers the memref types to MemrefDescriptor, which keep track of the base pointer allocation along with other information such as shapes and strides. 

In the GPU path, spv dialect didn’t choose this path due to complexities in SPIRV specification and necessity to target mobile platforms (performance issues). A detailed description for this choice is given here (https://mlir.llvm.org/docs/Dialects/SPIR-V/#lowering-memrefs-to-spvarray-and-spvrtarray). In spv dialect, the meref types are converted to a !spv.ptr<!spv.array<nelts x elem_type>> when the memref is statically shaped, and !spv.ptr<!spv.rtarray<elem_type>> when the memref is dynamically shaped.


Therefore, to enable this process, in this pass we convert memrefs with more than rank of 1 (ranks > 1) to flat pointers (1-D ranked spv.ptr) and keep other information separately in other variables. So that we can do the right index calculations. Since, we lose the information about both ranks and sizes in each ranks, we need to keep the information about both the shape and the strides to calculate the right address. 

This pass supports both statically and dynamically shaped memrefs.

# Example

## Before The pass:

```
 // -----// IR Dump Before {anonymous}::UnstrideMemrefsPass //----- //
func.func @addt(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>) -> memref<2x5xf32> {
  %c5 = arith.constant 5 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %memref = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
  memref.copy %arg1, %memref : memref<2x5xf32> to memref<2x5xf32>
  %memref_0 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
  memref.copy %arg0, %memref_0 : memref<2x5xf32> to memref<2x5xf32>
  %memref_1 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c2, %arg9 = %c5, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    %0 = memref.load %memref_0[%arg2, %arg3] : memref<2x5xf32>
    %1 = memref.load %memref[%arg2, %arg3] : memref<2x5xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %memref_1[%arg2, %arg3] : memref<2x5xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  gpu.dealloc  %memref_0 : memref<2x5xf32>
  gpu.dealloc  %memref : memref<2x5xf32>
  return %memref_1 : memref<2x5xf32>
}
```

## After the Pass:

```
// -----// IR Dump After {anonymous}::UnstrideMemrefsPass //----- //
func.func @addt(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>) -> memref<2x5xf32> {
  %c5 = arith.constant 5 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %memref = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
  %0 = "util.undef"() : () -> index
  %1 = memref.reinterpret_cast %memref to offset: [0], sizes: [%0], strides: [1] : memref<2x5xf32> to memref<?xf32>
  memref.copy %arg1, %memref : memref<2x5xf32> to memref<2x5xf32>
  %memref_0 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
  %2 = "util.undef"() : () -> index
  %3 = memref.reinterpret_cast %memref_0 to offset: [0], sizes: [%2], strides: [1] : memref<2x5xf32> to memref<?xf32>
  memref.copy %arg0, %memref_0 : memref<2x5xf32> to memref<2x5xf32>
  %memref_1 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
  %4 = "util.undef"() : () -> index
  %5 = memref.reinterpret_cast %memref_1 to offset: [0], sizes: [%4], strides: [1] : memref<2x5xf32> to memref<?xf32>
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c2, %arg9 = %c5, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    %6 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 5 + d1)>(%arg2, %arg3)[%c5]
    %7 = memref.load %3[%6] : memref<?xf32>
    %8 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 5 + d1)>(%arg2, %arg3)[%c5]
    %9 = memref.load %1[%8] : memref<?xf32>
    %10 = arith.addf %7, %9 : f32
    %11 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 5 + d1)>(%arg2, %arg3)[%c5]
    memref.store %10, %5[%11] : memref<?xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  gpu.dealloc  %memref_0 : memref<2x5xf32>
  gpu.dealloc  %memref : memref<2x5xf32>
  return %memref_1 : memref<2x5xf32>
}
```

As shown in the example above, the index calculation is done using affine apply and the memrefs are 1D.
 
## Reason for this Custom Pass:

Upstream does not have a pass which does these conversions. Our goal is to add this pass to upstream which we think will be useful to the MLIR community.
