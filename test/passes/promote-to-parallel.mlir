// RUN: dpcomp-opt %s --dpcomp-promote-to-parallel | FileCheck %s

// CHECK: promote
// CHECK: scf.parallel
// CHECK: scf.reduce
func @promote(%arg0: i64) -> i64 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c0_i64 = constant 0 : i64
  %0 = index_cast %arg0 : i64 to index
  %1 = scf.for %arg1 = %c0 to %0 step %c1 iter_args(%arg2 = %c0_i64) -> (i64) {
    %2 = index_cast %arg1 : index to i64
    %3 = addi %arg2, %2 : i64
    scf.yield %3 : i64
  }
  return %1 : i64
}
