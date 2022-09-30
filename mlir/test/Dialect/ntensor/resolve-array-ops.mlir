// RUN: imex-opt --ntensor-resolve-array-ops --split-input-file %s | FileCheck %s

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index) -> f32 {
  %0 = ntensor.getitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : index] -> f32
  return %0 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : index] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   ntensor.store %[[ARG3]], %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return
