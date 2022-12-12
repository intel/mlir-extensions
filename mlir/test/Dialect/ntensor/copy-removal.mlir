// RUN: imex-opt -pass-pipeline='builtin.module(func.func(ntensor-copy-removal))' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT: ntensor.copy %[[ARG1]], %[[ARG2]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT: return
func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.ntensor<?xf32>) {
  ntensor.copy %arg1, %arg2 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  ntensor.copy %arg1, %arg2 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  return
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//       CHECK: %[[RES1:.*]] = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>
//       CHECK: ntensor.copy %[[RES1]], %[[ARG]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//       CHECK: %[[RES2:.*]] = ntensor.primitive "bar" (%[[RES1]]) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
//       CHECK: return %[[RES2]]
func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>
  ntensor.copy %0, %arg1 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  %1 = ntensor.primitive "bar" (%arg1) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}
