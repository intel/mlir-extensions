// RUN: imex-opt -allow-unregistered-dialect --expand-tuple --canonicalize --split-input-file %s | FileCheck %s

func.func @test(%arg1: tuple<index, i64>) -> tuple<index, i64> {
  return %arg1 : tuple<index, i64>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   return %[[ARG1]], %[[ARG2]] : index, i64

// -----

func.func @test(%arg1: tuple<i64>) -> tuple<i64> {
  return %arg1 : tuple<i64>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: i64)
//  CHECK-NEXT:   return %[[ARG]] : i64

// -----

func.func @test() -> tuple<index, i64> {
  %0 = imex_util.env_region "test" -> tuple<index, i64> {
    %1 = "test.test"() : () -> tuple<index, i64>
    imex_util.env_region_yield %1: tuple<index, i64>
  }
  return %0 : tuple<index, i64>
}

// CHECK-LABEL: func @test
//       CHECK: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[RES:.*]]:2 = imex_util.env_region "test" -> index, i64 {
//       CHECK: %[[TUP:.*]] = "test.test"() : () -> tuple<index, i64>
//       CHECK: %[[E1:.*]] = imex_util.tuple_extract %[[TUP]] : tuple<index, i64>, %[[C0]] -> index
//       CHECK: %[[E2:.*]] = imex_util.tuple_extract %[[TUP]] : tuple<index, i64>, %[[C1]] -> i64
//       CHECK: imex_util.env_region_yield %[[E1]], %[[E2]] : index, i64
//       CHECK: }
//       CHECK: return %[[RES]]#0, %[[RES]]#1 : index, i64
