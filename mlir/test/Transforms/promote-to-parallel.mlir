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

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[INIT:.*]]: i32, %[[VAL:.*]]: i32)
//       CHECK:  %[[RES:.*]] = scf.parallel (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) init (%[[INIT]]) -> i32 {
//       CHECK:  scf.reduce(%[[VAL]]) : i32 {
//       CHECK:  ^bb0(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32):
//       CHECK:  %[[R:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
//       CHECK:  scf.reduce.return %[[R]] : i32
//       CHECK:  }
//       CHECK:  scf.yield
//       CHECK:  }
//       CHECK:  return %[[RES]] : i32
func.func @test(%arg: i32, %val: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = scf.for %i0 = %c0 to %c10 step %c1 iter_args(%arg1 = %arg) -> (i32) {
    %1 = arith.addi %arg1, %val : i32
    scf.yield %1 : i32
  }
  return %0 : i32
}
