// RUN: imex-opt --insert-gpu-allocs='client-api=opencl' %s | FileCheck %s --check-prefix=OPENCL
// RUN: imex-opt --insert-gpu-allocs='client-api=vulkan' %s | FileCheck %s --check-prefix=VULKAN

func.func @addt(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>) -> memref<2x5xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  // OPENCL: %[[MEMREF0:.*]] = gpu.alloc host_shared () : memref<2x5xf32>
  // OPENCL: memref.copy %arg1, %[[MEMREF0]] : memref<2x5xf32> to memref<2x5xf32>
  // OPENCL: %[[MEMREF1:.*]] = gpu.alloc host_shared () : memref<2x5xf32>
  // OPENCL: memref.copy %arg0, %[[MEMREF1]] : memref<2x5xf32> to memref<2x5xf32>
  // VULKAN: %[[MEMREF0:.*]] = memref.alloc() : memref<2x5xf32>
  // VULKAN: memref.copy %arg1, %[[MEMREF0]] : memref<2x5xf32> to memref<2x5xf32>
  // VULKAN: %[[MEMREF1:.*]] = memref.alloc() : memref<2x5xf32>
  // VULKAN: memref.copy %arg0, %[[MEMREF1]] : memref<2x5xf32> to memref<2x5xf32>

  %0 = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>
  // OPENCL:  %[[MEMREF2:.*]] = gpu.alloc host_shared () : memref<2x5xf32>

  %c1_0 = arith.constant 1 : index
  %1 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c2)[%c0, %c1]
  %2 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c5)[%c0, %c1]
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %1, %arg9 = %2, %arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_0, %arg12 = %c1_0, %arg13 = %c1_0) {
    %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg2)[%c1, %c0]
    %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg3)[%c1, %c0]
    %5 = memref.load %arg0[%3, %4] : memref<2x5xf32>
    %6 = memref.load %arg1[%3, %4] : memref<2x5xf32>
    %7 = arith.addf %5, %6 : f32
    memref.store %7, %0[%3, %4] : memref<2x5xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  return %0 : memref<2x5xf32>
}
