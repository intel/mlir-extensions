// RUN: imex-opt --split-input-file %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt --split-input-file %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt --split-input-file -mlir-print-op-generic %s | imex-opt | FileCheck %s

func.func @test() {
  region.env_region "test" {
    region.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   region.env_region "test" {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test() {
  region.env_region "test" {
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   region.env_region "test" {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) {
  region.env_region "test" %arg1 : index {
    region.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   region.env_region "test" %[[ARG1]] : index {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index, %arg2: i64) {
  region.env_region "test" %arg1, %arg2 : index, i64 {
    region.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   region.env_region "test" %[[ARG1]], %[[ARG2]] : index, i64 {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) -> index {
  %0 = region.env_region "test" -> index {
    region.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = region.env_region "test" -> index {
//  CHECK-NEXT:     region.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: index) -> index {
  %0 = region.env_region "test" %arg1 : index -> index {
    region.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = region.env_region "test" %[[ARG1]] : index -> index {
//  CHECK-NEXT:     region.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: index) -> index {
  %0 = region.env_region "test1" %arg1 : index -> index {
    %1 = region.env_region "test2" %arg1 : index -> index {
      region.env_region_yield %arg1: index
    }
    region.env_region_yield %1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = region.env_region "test1" %[[ARG1]] : index -> index {
//  CHECK-NEXT:     %[[RES1:.*]] = region.env_region "test2" %[[ARG1]] : index -> index {
//  CHECK-NEXT:       region.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:     }
//  CHECK-NEXT:   region.env_region_yield %[[RES1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index
