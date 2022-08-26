// RUN: imex-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s -split-input-file | imex-opt | FileCheck %s

func.func @test() {
  plier_util.env_region "test" {
    plier_util.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   plier_util.env_region "test" {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) {
  plier_util.env_region "test" %arg1 : index {
    plier_util.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   plier_util.env_region "test" %[[ARG1]] : index {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index, %arg2: i64) {
  plier_util.env_region "test" %arg1, %arg2 : index, i64 {
    plier_util.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   plier_util.env_region "test" %[[ARG1]], %[[ARG2]] : index, i64 {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) -> index {
  %0 = plier_util.env_region "test" -> index {
    plier_util.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = plier_util.env_region "test" -> index {
//  CHECK-NEXT:     plier_util.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: index) -> index {
  %0 = plier_util.env_region "test" %arg1 : index -> index {
    plier_util.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = plier_util.env_region "test" %[[ARG1]] : index -> index {
//  CHECK-NEXT:     plier_util.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index
