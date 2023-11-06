// RUN: imex-opt --insert-gpu-allocs='client-api=opencl' %s | FileCheck %s --check-prefix=OPENCL
// RUN: imex-opt --insert-gpu-allocs='client-api=vulkan' %s | FileCheck %s --check-prefix=VULKAN

#map0 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module {
  memref.global "private" constant @__constant_2x5xf32_0 : memref<2x5xf32> = dense<[[1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]>
  memref.global "private" constant @__constant_2x5xf32 : memref<2x5xf32> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]]>
func.func @addt(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>) -> memref<2x5xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %0 = memref.get_global @__constant_2x5xf32 : memref<2x5xf32>
  %1 = memref.get_global @__constant_2x5xf32_0 : memref<2x5xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>

  // OPENCL: [[VAR0:.*]] = memref.get_global @__constant_2x5xf32 : memref<2x5xf32>
  // OPENCL: %[[MEMREF0:.*]] = gpu.alloc host_shared () : memref<2x5xf32>
  // OPENCL: memref.copy [[VAR0]], %[[MEMREF0]] : memref<2x5xf32> to memref<2x5xf32>
  // OPENCL: [[VAR1:.*]] = memref.get_global @__constant_2x5xf32_0 : memref<2x5xf32>
  // OPENCL: %[[MEMREF1:.*]] = gpu.alloc host_shared () : memref<2x5xf32>
  // OPENCL: memref.copy [[VAR1]], %[[MEMREF1]] : memref<2x5xf32> to memref<2x5xf32>
  // OPENCL: %[[MEMREF2:.*]] = gpu.alloc host_shared () : memref<2x5xf32>
  // VULKAN: [[VAR0:.*]] = memref.get_global @__constant_2x5xf32 : memref<2x5xf32>
  // VULKAN: %[[MEMREF0:.*]] = memref.alloc() : memref<2x5xf32>
  // VULKAN: memref.copy [[VAR0]], %[[MEMREF0]] : memref<2x5xf32> to memref<2x5xf32>
  // VULKAN: [[VAR1:.*]] = memref.get_global @__constant_2x5xf32_0 : memref<2x5xf32>
  // VULKAN: %[[MEMREF1:.*]] = memref.alloc() : memref<2x5xf32>
  // VULKAN: memref.copy [[VAR1]], %[[MEMREF1]] : memref<2x5xf32> to memref<2x5xf32>

  %c1_0 = arith.constant 1 : index
  %3 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c2)[%c0, %c1]
  %4 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c5)[%c0, %c1]
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %3, %arg9 = %4, %arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_0, %arg12 = %c1_0, %arg13 = %c1_0) {
    // OPENCL: gpu.launch {{.*}}
    // VULKAN: gpu.launch {{.*}}

    %5 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg2)[%c1, %c0]
    %6 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg3)[%c1, %c0]

    %7 = memref.load %0[%5, %6] : memref<2x5xf32>
    %8 = memref.load %1[%5, %6] : memref<2x5xf32>

    // OPENCL: [[VAR2:.*]] = memref.load %[[MEMREF0]][%4, %5] : memref<2x5xf32>
    // OPENCL: [[VAR3:.*]] = memref.load %[[MEMREF1]][%4, %5] : memref<2x5xf32>
    // VULKAN: [[VAR2:.*]] = memref.load %[[MEMREF0]][%4, %5] : memref<2x5xf32>
    // VULKAN: [[VAR3:.*]] = memref.load %[[MEMREF1]][%4, %5] : memref<2x5xf32>

    %9 = arith.addf %7, %8 : f32
    memref.store %9, %2[%5, %6] : memref<2x5xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  return %2 : memref<2x5xf32>
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
