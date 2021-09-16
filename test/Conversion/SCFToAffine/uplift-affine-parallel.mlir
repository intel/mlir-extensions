// RUN: dpcomp-opt %s --scf-to-affine -allow-unregistered-dialect -split-input-file| FileCheck %s

// CHECK_LABLEL: empty_affine
// CHECK: affine.parallel
// CHECK-NEXT: affine.load
// CHECK-NEXT: affine.store
func @copy_affine(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf64>
  scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
    %2 = memref.load %arg0[%arg4] : memref<?xf64>
    memref.store %2, %arg1[%arg4] : memref<?xf64>
    scf.yield
  }
  return
}

// -----

// CHECK_LABLEL: copy_affine_with_strides
// CHECK: affine.parallel
// CHECK-NEXT: affine.load
// CHECK-NEXT: affine.store
#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
func @copy_affine_with_strides(%arg0: memref<?xf64, #map>, %arg1: memref<?xf64, #map>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf64, #map>
  scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
    %2 = memref.load %arg0[%arg4] : memref<?xf64, #map>
    memref.store %2, %arg1[%arg4] : memref<?xf64, #map>
    scf.yield
  }
  return
}
