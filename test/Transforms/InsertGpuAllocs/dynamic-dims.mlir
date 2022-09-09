// RUN: imex-opt --insert-gpu-alloc %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @addt(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) -> memref<2x5xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: [[VAR0:.*]] = memref.dim %arg1, %c0 : memref<?x?xf32>
  // CHECK: [[VAR1:.*]] = memref.dim %arg1, %c1 : memref<?x?xf32>
  // CHECK: %[[MEMREF0:.*]] = gpu.alloc ([[VAR0:.*]], [[VAR1:.*]]) {gpu.alloc_shared} : memref<?x?xf32>
  %0 = memref.alloc() {alignment = 128 : i64} : memref<2x5xf32>
  %1 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %2 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %c1_0 = arith.constant 1 : index
  %3 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%1)[%c0, %c1]
  %4 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%2)[%c0, %c1]
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %3, %arg9 = %4, %arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_0, %arg12 = %c1_0, %arg13 = %c1_0) {
    %5 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg2)[%c1, %c0]
    %6 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg3)[%c1, %c0]
    %7 = memref.load %arg0[%5, %6] : memref<?x?xf32>
    // CHECK: [[VAR1:.*]] = memref.load %[[MEMREF0:.*]][%8, %9] : memref<?x?xf32>
    %8 = memref.load %arg1[%5, %6] : memref<?x?xf32>
    %9 = arith.addf %7, %8 : f32
    memref.store %9, %0[%5, %6] : memref<2x5xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  return %0 : memref<2x5xf32>
}
