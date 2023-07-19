// RUN: imex-opt --set-spirv-abi-attrs='client-api=opencl' %s | FileCheck %s --check-prefix=OPENCL
// RUN: imex-opt --set-spirv-abi-attrs='client-api=vulkan' %s | FileCheck %s --check-prefix=VULKAN

gpu.module @main_kernel {
  gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel {

  // OPENCL: gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
  // VULKAN: gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {

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
