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
