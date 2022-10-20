// RUN: imex-opt --expand-tuple --canonicalize --split-input-file %s | FileCheck %s

func.func @test(%arg1: tuple<index, i64>) -> tuple<index, i64> {
  return %arg1 : tuple<index, i64>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT: return %[[ARG1]], %[[ARG2]] : index, i64
