// RUN: dpcomp-opt %s --scf-to-affine -allow-unregistered-dialect -split-input-file| FileCheck %s

// CHECK_LABLEL: empty_affine
// CHECK: affine.parallel
func @empty_affine(%arg0: index) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  scf.parallel (%arg1) = (%c0) to (%arg0) step (%c1) {
    "test.foo"(%arg1) : (index) -> ()
    scf.yield
  }
  return
}
