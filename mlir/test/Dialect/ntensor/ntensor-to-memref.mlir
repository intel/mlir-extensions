// RUN: imex-opt %s -ntensor-to-memref --split-input-file | FileCheck %s

func.func @test(%arg: !ntensor.ntensor<?x?xf32>) -> !ntensor.ntensor<?x?xf32> {
  return %arg : !ntensor.ntensor<?x?xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?x?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : memref<?x?xf32>

// -----

func.func @test(%arg: !ntensor.ntensor<?x?xf32>) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.dim %arg, %0 : !ntensor.ntensor<?x?xf32>
  return %1 : index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?x?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = memref.dim %[[ARG]], %[[C0]] : memref<?x?xf32>
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg: !ntensor.ntensor<?x?xf32, "test">) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.dim %arg, %0 : !ntensor.ntensor<?x?xf32, "test">
  return %1 : index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?x?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> index {
//  CHECK-NEXT:   %[[RES1:.*]] = memref.dim %[[ARG]], %[[C0]] : memref<?x?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index
