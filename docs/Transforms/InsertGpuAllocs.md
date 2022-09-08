# InsertGpuAllocs Pass


The InsertGpuAllocs pass, as the name suggests, inserts the gpu allocs in the IR. Memref alloc is an operation in the memref dialect that can be used to allocate the memory on the host side and or on the device side. The MLIR IR is a mix of host and device code.
To distinguish between host side memory allocation and device side memory allocation, we convert all the memref.allocs that refer to device (gpu) side memory allocations and references, into gpu.alloc, which is an operation of the upstream GPU dialect. This distinction helps in lowering to llvm and calling the appropriate memory allocation operation at runtime.
The pass traverses all the memref (load/store) operations inside the gpu launch op in the IR and checks for its aliases and its defining op. If the defining op is a memref.alloc op it replaces that op in the IR with gpu.alloc op, because all the operations under the gpu.launch op are device side computations and will execute on the device.

# Example

```
// -----// IR Dump Before {anonymous}::InsertGPUAllocs //----- //
func.func @main() {
  %0 = memref.alloc() : memref<8xf32>
  %1 = memref.alloc() : memref<8xf32>
  %2 = memref.alloc() : memref<8xf32>
  .
  .
  .
  gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c8, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) {
    %7 = gpu.block_id  x
    %8 = memref.load %0[%7] : memref<8xf32>
    %9 = memref.load %1[%7] : memref<8xf32>
    %10 = arith.addf %8, %9 : f32
    memref.store %10, %2[%7] : memref<8xf32>
    gpu.terminator
  }
  %6 = memref.cast %2 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()
  return
}
```

The Pass will change the IR to:

```
// -----// IR Dump After {anonymous}::InsertGPUAllocs //----- //
func.func @main() {
  %memref = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
  %memref_2 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
  %memref_3 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
  .
  .
  .
  gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c8, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) {
    %4 = gpu.block_id  x
    %5 = memref.load %memref[%4] : memref<8xf32>
    %6 = memref.load %memref_2[%4] : memref<8xf32>
    %7 = arith.addf %5, %6 : f32
    memref.store %7, %memref_3[%4] : memref<8xf32>
    gpu.terminator
  }
  %3 = memref.cast %memref_3 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  return
}
```


As shown in the example above, the memref.allocs in the IR are referring to device buffer allocation and hence they are replaced with gpu.alloc from the gpu dialect.

## Limitations of this pass.

1. This pass only supports only memref::AllocOp and not its variants like memref::AllocaOp, memref::AllocaScopeOp & AllocaScopeReturnOp.
2. This pass needs to be run before the GpuKernelOutlining pass since it operates on gpu.launch blocks and not on gpu.launch_func.
3. This pass only covers static shapes and shapes with unknown dims and known rank.

Note: We plan to add support for these limitations in incremental future PR's.

## Reason for this Custom Pass:

Upstream does not have a pass which does these conversions. Our goal is to add this pass to upstream which we think will be useful to the MLIR community.
