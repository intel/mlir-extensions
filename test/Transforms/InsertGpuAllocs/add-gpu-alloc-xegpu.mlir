// RUN: imex-opt --insert-gpu-allocs='client-api=opencl' %s | FileCheck %s --check-prefix=OPENCL
// RUN: imex-opt --insert-gpu-allocs='client-api=vulkan' %s | FileCheck %s --check-prefix=VULKAN

func.func @addt(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>) -> memref<2x5xf32> {
  %c1 = arith.constant 1 : index
  // OPENCL: %[[MEMREF0:.*]] = gpu.alloc () : memref<2x5xf32>
  // OPENCL: %[[MEMREF1:.*]] = gpu.alloc host_shared () : memref<2x5xf32>
  // VULKAN: %[[MEMREF0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>
  // VULKAN: %[[MEMREF1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>

  %0 = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>
  %1 = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>

  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    %c0 = arith.constant 0 : index
    %src_tile = xegpu.create_nd_tdesc %0[%c0, %c0] : memref<2x5xf32> -> !xegpu.tensor_desc<2x5xf32>
    %src_value = xegpu.load_nd %src_tile  : !xegpu.tensor_desc<2x5xf32> -> vector<2x5xf32>
    %res_tile = xegpu.create_nd_tdesc %1[%c0, %c0] : memref<2x5xf32> -> !xegpu.tensor_desc<2x5xf32>
    xegpu.store_nd %src_value, %res_tile: vector<2x5xf32>, !xegpu.tensor_desc<2x5xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  return %1 : memref<2x5xf32>
}
