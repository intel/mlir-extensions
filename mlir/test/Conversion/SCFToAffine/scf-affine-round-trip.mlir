// RUN: imex-opt %s --scf-to-affine -lower-affine -split-input-file| FileCheck %s

// CHECK-LABLEL: copy_affine
// CHECK: scf.parallel
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.store
func.func @copy_affine(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf64>
  scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
    %2 = memref.load %arg0[%arg4] : memref<?xf64>
    memref.store %2, %arg1[%arg4] : memref<?xf64>
    scf.yield
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
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf64, #map>
  scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
    %2 = memref.load %arg0[%arg4] : memref<?xf64, #map>
    memref.store %2, %arg1[%arg4] : memref<?xf64, #map>
    scf.yield
  }
  return
}
