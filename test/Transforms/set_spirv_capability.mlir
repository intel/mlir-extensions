// RUN: imex-opt --set-spirv-capablilities %s | FileCheck %s

module attributes {gpu.container_module} {

// CHECK: module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spv.resource_limits<>>} {

  func.func @main() {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.200000e+00 : f32
    %cst_0 = arith.constant 1.100000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %memref = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %memref_2 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %memref_3 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %0 = memref.cast %memref : memref<8xf32> to memref<?xf32>
    %1 = memref.cast %memref_2 : memref<8xf32> to memref<?xf32>
    %2 = memref.cast %memref_3 : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%0, %cst_0) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%1, %cst) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%2, %cst_1) : (memref<?xf32>, f32) -> ()
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8xf32>, %memref_2 : memref<8xf32>, %memref_3 : memref<8xf32>)
    %3 = memref.cast %memref_3 : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%3) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>} {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<8xf32>
      %2 = memref.load %arg1[%0] : memref<8xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<8xf32>
      gpu.return
    }
  }
  func.func private @fillResource1DFloat(memref<?xf32>, f32)
  func.func private @printMemrefF32(memref<*xf32>)
}