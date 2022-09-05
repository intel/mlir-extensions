// RUN: imex-opt %s -canonicalize --split-input-file | FileCheck %s

func.func @test(%arg1: index, %arg2: i64) -> i64 {
  %0 = imex_util.build_tuple %arg1, %arg2: index, i64 -> tuple<index, i64>
  %cst = arith.constant 1 : index
  %1 = imex_util.tuple_extract %0 : tuple<index, i64>, %cst -> i64
  return %1 : i64
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   return %[[ARG2]] : i64

// -----

func.func @remove_empty_region() {
  imex_util.env_region "test" {
  }
  return
}
// CHECK-LABEL: func @remove_empty_region
//   CHECK-NOT:   imex_util.env_region
//  CHECK-NEXT:   return

// -----

func.func @empty_region_out_value(%arg1: index) -> index {
  %0 = imex_util.env_region "test" -> index {
    imex_util.env_region_yield %arg1: index
  }
  return %0 : index
}
// CHECK-LABEL: func @empty_region_out_value
//  CHECK-SAME: (%[[ARG:.*]]: index)
//       CHECK: return %[[ARG]] : index
