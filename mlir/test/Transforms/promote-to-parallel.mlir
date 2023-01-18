// RUN: imex-opt -allow-unregistered-dialect --imex-promote-to-parallel --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[INIT:.*]]: index)
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[C1:.*]] = arith.constant 1 : index
//       CHECK:  %[[C10:.*]] = arith.constant 10 : index
//       CHECK:  %[[RES:.*]] = scf.parallel (%[[ARG0:.*]]) = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) init (%[[INIT]]) -> index {
//       CHECK:  scf.reduce(%[[ARG0]]) : index {
//       CHECK:  ^bb0(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index):
//       CHECK:  %[[R:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : index
//       CHECK:  scf.reduce.return %[[R]] : index
//       CHECK:  }
//       CHECK:  scf.yield
//       CHECK:  }
//       CHECK:  return %[[RES]] : index
func.func @test(%arg: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = scf.for %i0 = %c0 to %c10 step %c1 iter_args(%arg1 = %arg) -> (index) {
    %1 = arith.addi %arg1, %i0 : index
    scf.yield %1 : index
  }
  return %0 : index
}
