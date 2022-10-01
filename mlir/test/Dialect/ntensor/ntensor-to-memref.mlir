// RUN: imex-opt %s -ntensor-to-memref --split-input-file | FileCheck %s

func.func @test(%arg: !ntensor.ntensor<?x?xf32>) -> !ntensor.ntensor<?x?xf32> {
  return %arg : !ntensor.ntensor<?x?xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?x?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : memref<?x?xf32>
