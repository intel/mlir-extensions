// RUN: imex-opt --split-input-file --set-spirv-abi-attrs='client-api=opencl' %s | FileCheck %s --check-prefix=OPENCL
// RUN: imex-opt --split-input-file --set-spirv-abi-attrs='client-api=vulkan' %s | FileCheck %s --check-prefix=VULKAN

gpu.module @main_kernel {
  gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel {
  // OPENCL-LABEL: gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes
  // OPENCL-SAME:    {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
  // VULKAN-LABEL: gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes
  // VULKAN-SAME:    {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
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

// -----

module {
  module attributes {gpu.container_module} {
    func.func @run(%arg0: memref<4096x4096xf16>) -> memref<4096x4096xf16> attributes {llvm.emit_c_interface} {
      %c1 = arith.constant 1 : index
      gpu.launch_func  @run_kernel::@run_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4096x4096xf16>)
      return %arg0 : memref<4096x4096xf16>
    }
    gpu.module @run_kernel {
      // OPENCL-LABEL: gpu.func @run_kernel(%arg0: memref<4096x4096xf16>) kernel attributes
      // OPENCL-SAME:    {VectorComputeFunctionINTEL, known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 1, 1, 1>,
      // OPENCL-SAME:     spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // VULKAN-LABEL: gpu.func @run_kernel(%arg0: memref<4096x4096xf16>) kernel attributes
      // VULKAN-SAME:    {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 1, 1, 1>,
      // VULKAN-SAME:     spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      gpu.func @run_kernel(%arg0: memref<4096x4096xf16>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
      }
    }
  }
}
