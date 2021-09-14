// RUN: dpcomp-opt %s -lower-affine -allow-unregistered-dialect -split-input-file| FileCheck %s

// CHECK_LABLEL: empty_affine
// CHECK: scf.parallel
func @empty_affine(%arg0: index) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  affine.parallel (%arg1) = (%c0) to (%arg0) step (1) {
    "test.foo"(%arg1) : (index) -> ()
    affine.yield
  }
  return
}
