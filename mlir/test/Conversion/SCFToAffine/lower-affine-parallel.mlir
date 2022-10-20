// RUN: imex-opt %s -lower-affine -split-input-file| FileCheck %s

// CHECK-LABLEL: copy_affine
// CHECK: scf.parallel
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.store
func.func @copy_affine(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf64>
  affine.parallel (%arg3) = (0) to (symbol(%0)) {
    %1 = affine.load %arg0[%arg3] : memref<?xf64>
    affine.store %1, %arg1[%arg3] : memref<?xf64>
  }
  return
}

// -----

// CHECK-LABLEL: copy_affine_with_strides
// CHECK: scf.parallel
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.store
#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
func.func @copy_affine_with_strides(%arg0: memref<?xf64, #map>, %arg1: memref<?xf64, #map>) {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf64, #map>
  affine.parallel (%arg2) = (0) to (symbol(%0)) {
    %1 = affine.load %arg0[%arg2] : memref<?xf64, #map>
    affine.store %1, %arg1[%arg2] : memref<?xf64, #map>
  }
  return
}
