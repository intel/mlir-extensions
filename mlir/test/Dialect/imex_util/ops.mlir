// RUN: imex-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s -split-input-file | imex-opt | FileCheck %s

func.func @test() -> tuple<> {
  %0 = imex_util.build_tuple tuple<>
  return %0 : tuple<>
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.build_tuple tuple<>
//  CHECK-NEXT:   return %[[RES]] : tuple<>

// -----

func.func @test(%arg1: index, %arg2: i64) -> tuple<index, i64> {
  %0 = imex_util.build_tuple %arg1, %arg2: index, i64 -> tuple<index, i64>
  return %0 : tuple<index, i64>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.build_tuple %[[ARG1]], %[[ARG2]] : index, i64 -> tuple<index, i64>
//  CHECK-NEXT:   return %[[RES]] : tuple<index, i64>

// -----

func.func @test(%arg1: tuple<index, i64>, %arg2: index) -> i64 {
  %0 = imex_util.tuple_extract %arg1 : tuple<index, i64>, %arg2 -> i64
  return %0 : i64
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tuple<index, i64>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.tuple_extract %[[ARG1]] : tuple<index, i64>, %[[ARG2]] -> i64
//  CHECK-NEXT:   return %[[RES]] : i64
