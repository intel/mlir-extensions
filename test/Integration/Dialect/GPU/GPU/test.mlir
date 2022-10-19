// RUN: %{python_executable} %{imex_tools_dir}/imex-runner.py -i %s -a -b --pass-pipeline-file=%p/gpu-to-spirv.pp -n | FileCheck %s

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>} {
  memref.global "private" constant @__constant_2x5xf32_0 : memref<2x5xf32> = dense<[[1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]>
  memref.global "private" constant @__constant_2x5xf32 : memref<2x5xf32> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]]>
  func.func @addt(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>) -> memref<2x5xf32> {
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
    %0 = memref.reinterpret_cast %memref to offset: [0], sizes: [%c10], strides: [1] : memref<2x5xf32> to memref<?xf32>
    memref.copy %arg1, %memref : memref<2x5xf32> to memref<2x5xf32>
    %memref_0 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
    %1 = memref.reinterpret_cast %memref_0 to offset: [0], sizes: [%c10], strides: [1] : memref<2x5xf32> to memref<?xf32>
    memref.copy %arg0, %memref_0 : memref<2x5xf32> to memref<2x5xf32>
    %memref_1 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5xf32>
    %2 = memref.reinterpret_cast %memref_1 to offset: [0], sizes: [%c10], strides: [1] : memref<2x5xf32> to memref<?xf32>
    gpu.launch_func  @addt_kernel::@addt_kernel blocks in (%c2, %c5, %c1) threads in (%c1, %c1, %c1) args(%1 : memref<?xf32>, %0 : memref<?xf32>, %2 : memref<?xf32>)
    gpu.dealloc  %memref_1 : memref<2x5xf32>
    gpu.dealloc  %memref_0 : memref<2x5xf32>
    gpu.dealloc  %memref : memref<2x5xf32>
    return %memref_1 : memref<2x5xf32>
  }
  gpu.module @addt_kernel {
    gpu.func @addt_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c5 = arith.constant 5 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = arith.muli %0, %c5 : index
      %3 = arith.addi %2, %1 : index
      %4 = memref.load %arg0[%3] : memref<?xf32>
      %5 = memref.load %arg1[%3] : memref<?xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg2[%3] : memref<?xf32>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_2x5xf32 : memref<2x5xf32>
    %1 = memref.get_global @__constant_2x5xf32_0 : memref<2x5xf32>
    %2 = call @addt(%0, %1) : (memref<2x5xf32>, memref<2x5xf32>) -> memref<2x5xf32>
    %3 = memref.cast %2 : memref<2x5xf32> to memref<*xf32>
    call @printMemrefF32(%3) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
}
