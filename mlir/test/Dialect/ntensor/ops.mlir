// RUN: imex-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s -split-input-file | imex-opt | FileCheck %s

func.func @test(%arg: !ntensor.ntensor<2x3x4xf32>) -> !ntensor.ntensor<2x3x4xf32> {
  return %arg : !ntensor.ntensor<2x3x4xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<2x3x4xf32>)
//  CHECK-NEXT:   return %[[ARG]] : !ntensor.ntensor<2x3x4xf32>

// -----

func.func @test(%arg: !ntensor.ntensor<2x3x4xf32, "test">) -> !ntensor.ntensor<2x3x4xf32, "test"> {
  return %arg : !ntensor.ntensor<2x3x4xf32, "test">
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<2x3x4xf32, "test">)
//  CHECK-NEXT:   return %[[ARG]] : !ntensor.ntensor<2x3x4xf32, "test">

// -----

func.func @test(%arg: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.unary "-" (%arg : !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.unary "-" (%[[ARG]] : !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>
