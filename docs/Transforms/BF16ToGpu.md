# BF16ToGPU Pass


BF16ToGPU pass transforms gpu dialect with bf16 dtype to a form that can be lowered to
spirv and execute on Intel GPUs.
Since bf16 is not a type directly supported by spirv, bf16 type is passes as an i16 type to
spirv functions. This requires changing gpu.launch_func (caller) and gpu.func (callee).

* Caller side changes are as follows:  
MLIR does not support direct casting of non scalar type, so caller side casting from
bf16 type to i16 type bf16 is done indirectly by using memref.view
gpu.alloc of bf16 type is replaced with gpu.alloc of i8 type. Then one view of bf16 and
another view of i16 is created. Host side code that used original gpu.alloc is updated to
use the bf16 view. And the i16 view is passed to gpu.launch_func, replacing old arguments.

* Callee side changes are as follows:  
gpu.func's bf16 arguments replaced with i16 arguments. bf16 type usage inside gpu kernel body
can be divided into two different types. First, operations that interpret bits according to
bf16 specification. Second, operations that only care about the size of the types.
First type of operations include most of Arithmetic dialect and Math dialect operations
expect for bit cast. The operation's bf16 operands and results types are replaced with f32.
bf16 operands are replaced with a seqeuence i16 to bf16 bitcast followed by bf16 to f32 extf.
bf16 results are replaced with a sequence of f32 to bf16 truncf followed by bf16 to i16 bitcast.
The resulting code, in summary, has two parts.
    1) bf16 operations emulated by f32 operations with the help of widening and truncating.
    2) bitcast operations between bf16 and i16 type.

    Second type of operations get bf16 operands and results type replaced with i16.


## Example

```
  func.func @test(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>) -> memref<10x20xbf16> {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<10x20xbf16>
    memref.copy %arg1, %memref : memref<10x20xbf16> to memref<10x20xbf16>
    %memref_0 = gpu.alloc  host_shared () : memref<10x20xbf16>
    memref.copy %arg0, %memref_0 : memref<10x20xbf16> to memref<10x20xbf16>
    %memref_1 = gpu.alloc  host_shared () : memref<10x20xbf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<10x20xbf16>, %memref : memref<10x20xbf16>, %memref_1 : memref<10x20xbf16>)
    gpu.dealloc  %memref_0 : memref<10x20xbf16>
    gpu.dealloc  %memref : memref<10x20xbf16>
    return %memref_1 : memref<10x20xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>, %arg2: memref<10x20xbf16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<10x20xbf16>
      %3 = memref.load %arg1[%0, %1] : memref<10x20xbf16>
      %4 = arith.addf %2, %3 : bf16
      memref.store %4, %arg2[%0, %1] : memref<10x20xbf16>
      gpu.return
    }
  }
```

The Pass will change the IR to:

```
  func.func @test(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>) -> memref<10x20xbf16> {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<400xi8>
    %view = memref.view %memref[%c0][] : memref<400xi8> to memref<10x20xbf16>
    %view_0 = memref.view %memref[%c0][] : memref<400xi8> to memref<10x20xi16>
    memref.copy %arg1, %view : memref<10x20xbf16> to memref<10x20xbf16>
    %c0_1 = arith.constant 0 : index
    %memref_2 = gpu.alloc  host_shared () : memref<400xi8>
    %view_3 = memref.view %memref_2[%c0_1][] : memref<400xi8> to memref<10x20xbf16>
    %view_4 = memref.view %memref_2[%c0_1][] : memref<400xi8> to memref<10x20xi16>
    memref.copy %arg0, %view_3 : memref<10x20xbf16> to memref<10x20xbf16>
    %c0_5 = arith.constant 0 : index
    %memref_6 = gpu.alloc  host_shared () : memref<400xi8>
    %view_7 = memref.view %memref_6[%c0_5][] : memref<400xi8> to memref<10x20xbf16>
    %view_8 = memref.view %memref_6[%c0_5][] : memref<400xi8> to memref<10x20xi16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%view_4 : memref<10x20xi16>, %view_0 : memref<10x20xi16>, %view_8 : memref<10x20xi16>)
    gpu.dealloc  %memref_2 : memref<400xi8>
    gpu.dealloc  %memref : memref<400xi8>
    return %view_7 : memref<10x20xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xi16>, %arg1: memref<10x20xi16>, %arg2: memref<10x20xi16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<10x20xi16>
      %3 = memref.load %arg1[%0, %1] : memref<10x20xi16>
      %4 = arith.bitcast %2 : i16 to bf16
      %5 = arith.extf %4 : bf16 to f32
      %6 = arith.bitcast %3 : i16 to bf16
      %7 = arith.extf %6 : bf16 to f32
      %8 = arith.addf %5, %7 : f32
      %9 = arith.truncf %8 : f32 to bf16
      %10 = arith.bitcast %9 : bf16 to i16
      memref.store %10, %arg2[%0, %1] : memref<10x20xi16>
      gpu.return
    }
  }
```


As shown in the example above, the memref.allocs in the IR are referring to device buffer allocation and hence they are replaced with gpu.alloc from the gpu dialect.

## Limitations of this pass.


1. This pass only covers static shapes.
2. This pass only supports scalar operations in kernel body.
