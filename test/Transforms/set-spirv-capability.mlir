// RUN: imex-opt --set-spirv-capabilities='client-api=opencl' %s | FileCheck %s --check-prefix=OPENCL
// RUN: imex-opt --set-spirv-capabilities='client-api=vulkan' %s | FileCheck %s --check-prefix=VULKAN

module attributes {gpu.container_module} {

// OPENCL: module attributes {gpu.container_module} {
// OPENCL: gpu.module @main_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>} {
// VULKAN: module attributes {gpu.container_module} {
// VULKAN: gpu.module @main_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
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
