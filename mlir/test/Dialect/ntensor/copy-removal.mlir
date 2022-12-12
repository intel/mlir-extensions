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
