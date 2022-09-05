// RUN: imex-opt %s -allow-unregistered-dialect -canonicalize --split-input-file | FileCheck %s

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

// -----

func.func @merge_nested_region() {
  imex_util.env_region "test" {
    "test.test1"() : () -> ()
    imex_util.env_region "test" {
      "test.test2"() : () -> ()
    }
    "test.test3"() : () -> ()
  }
  return
}
// CHECK-LABEL: func @merge_nested_region
//  CHECK-NEXT:   imex_util.env_region
//  CHECK-NEXT:   "test.test1"() : () -> ()
//  CHECK-NEXT:   "test.test2"() : () -> ()
//  CHECK-NEXT:   "test.test3"() : () -> ()
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @nested_region_yield_args() {
  %0:4 = imex_util.env_region "test" -> index, i32, index, i64 {
    %1:3 = "test.test1"() : () -> (index, i32, i64)
    imex_util.env_region_yield %1#0, %1#1, %1#0, %1#2: index, i32, index, i64
  }
  "test.test2"(%0#0, %0#2, %0#3) : (index, index, i64) -> ()
  return
}
// CHECK-LABEL: func @nested_region_yield_args
//  CHECK-NEXT:   %[[RES:.*]]:2 = imex_util.env_region "test" -> index, i64 {
//  CHECK-NEXT:   %[[VAL:.*]]:3 = "test.test1"() : () -> (index, i32, i64)
//  CHECK-NEXT:   imex_util.env_region_yield %[[VAL]]#0, %[[VAL]]#2 : index, i64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   "test.test2"(%[[RES]]#0, %[[RES]]#0, %[[RES]]#1) : (index, index, i64) -> ()
//  CHECK-NEXT:   return
