// RUN: imex-opt --set-spirv-capablilities %s | FileCheck %s

module attributes {gpu.container_module} {

// CHECK: module attributes {gpu.container_module} {
// CHECK: gpu.module @main_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Linkage, Kernel, Vector16, Float16Buffer, Float16, Float64, Int64, Groups, Int16, GenericPointer, Int8, ExpectAssumeKHR, AtomicFloat32AddEXT], [SPV_KHR_expect_assume, SPV_EXT_shader_atomic_float_add, GroupNonUniformArithmetic]>, #spirv.resource_limits<>>} {

  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
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
}
