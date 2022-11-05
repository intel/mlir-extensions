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

// -----

func.func @test() {
  imex_util.env_region "test" {
    imex_util.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   imex_util.env_region "test" {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test() {
  imex_util.env_region "test" {
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   imex_util.env_region "test" {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) {
  imex_util.env_region "test" %arg1 : index {
    imex_util.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   imex_util.env_region "test" %[[ARG1]] : index {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index, %arg2: i64) {
  imex_util.env_region "test" %arg1, %arg2 : index, i64 {
    imex_util.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   imex_util.env_region "test" %[[ARG1]], %[[ARG2]] : index, i64 {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) -> index {
  %0 = imex_util.env_region "test" -> index {
    imex_util.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> index {
//  CHECK-NEXT:     imex_util.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: index) -> index {
  %0 = imex_util.env_region "test" %arg1 : index -> index {
    imex_util.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" %[[ARG1]] : index -> index {
//  CHECK-NEXT:     imex_util.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.enforce_shape %[[ARG1]] : tensor<?xf32>(%[[ARG2]]) -> tensor<?xf32>
//  CHECK-NEXT:   return %[[RES]]
func.func @test(%arg1: tensor<?xf32>, %arg2: index) -> tensor<?xf32> {
  %0 = imex_util.enforce_shape %arg1 : tensor<?xf32>(%arg2) -> tensor<?xf32>
  return %0: tensor<?xf32>
}
