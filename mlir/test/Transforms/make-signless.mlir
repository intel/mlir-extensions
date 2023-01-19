// RUN: imex-opt --imex-make-signless --canonicalize --split-input-file %s | FileCheck %s

func.func @test(%arg: si32) -> si32 {
  return %arg : si32
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: i32)
//  CHECK-NEXT:   return %[[ARG]] : i32

// -----

func.func @test(%arg: memref<?xsi32>) -> memref<?xsi32> {
  return %arg : memref<?xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32>)
//  CHECK-NEXT:   return %[[ARG]] : memref<?xi32>

// -----

func.func @test(%arg: tensor<?xsi32>) -> tensor<?xsi32> {
  return %arg : tensor<?xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: tensor<?xi32>)
//  CHECK-NEXT:   return %[[ARG]] : tensor<?xi32>

// -----

func.func @test(%arg: index) -> memref<?xsi32> {
  %0 = memref.alloc(%arg) : memref<?xsi32>
  return %0 : memref<?xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = memref.alloc(%[[ARG]]) : memref<?xi32>
//  CHECK-NEXT:   return %[[RES]] : memref<?xi32>

// -----

func.func @test(%arg: index) -> memref<?xsi32> {
  %0 = memref.alloca(%arg) : memref<?xsi32>
  return %0 : memref<?xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = memref.alloca(%[[ARG]]) : memref<?xi32>
//  CHECK-NEXT:   return %[[RES]] : memref<?xi32>

// -----

func.func @test(%arg: memref<?xsi32>) {
  memref.dealloc %arg : memref<?xsi32>
  return
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32>)
//  CHECK-NEXT:   memref.dealloc %[[ARG]] : memref<?xi32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg: index) -> tensor<?xsi32> {
  %0 = tensor.empty(%arg) : tensor<?xsi32>
  return %0 : tensor<?xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = tensor.empty(%[[ARG]]) : tensor<?xi32>
//  CHECK-NEXT:   return %[[RES]] : tensor<?xi32>

// -----

func.func @test(%arg1: si32, %arg2: si32, %arg3: si32) -> tensor<3xsi32> {
  %0 = tensor.from_elements %arg1, %arg2, %arg3 : tensor<3xsi32>
  return %0 : tensor<3xsi32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32)
//  CHECK-NEXT:   %[[RES:.*]] = tensor.from_elements %[[ARG1]], %[[ARG2]], %[[ARG3]] : tensor<3xi32>
//  CHECK-NEXT:   return %[[RES]] : tensor<3xi32>

// -----

func.func @test(%arg1: tensor<?xsi32>, %arg2: si32) -> tensor<?xsi32> {
  %0 = linalg.fill ins(%arg2 : si32) outs(%arg1 : tensor<?xsi32>) -> tensor<?xsi32>
  return %0 : tensor<?xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tensor<?xi32>, %[[ARG2:.*]]: i32)
//  CHECK-NEXT:   %[[RES:.*]] = linalg.fill ins(%[[ARG2]] : i32) outs(%[[ARG1]] : tensor<?xi32>) -> tensor<?xi32>
//  CHECK-NEXT:   return %[[RES]] : tensor<?xi32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @test(%arg0: tensor<2x3xsi32>, %arg1 : tensor<2x3xsi32>) -> tensor<2x3xsi32> {
  %0 = tensor.empty() : tensor<2x3xsi32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2x3xsi32>, tensor<2x3xsi32>)
      outs(%0 : tensor<2x3xsi32>) {
    ^bb0(%arg3: si32, %arg4: si32, %arg5: si32):
      %2 = imex_util.sign_cast %arg3 : si32 to i32
      %3 = imex_util.sign_cast %arg4 : si32 to i32
      %4 = arith.addi %2, %3 : i32
      %5 = imex_util.sign_cast %4 : i32 to si32
      linalg.yield %5 : si32
  } -> tensor<2x3xsi32>
  return %1 : tensor<2x3xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tensor<2x3xi32>, %[[ARG2:.*]]: tensor<2x3xi32>)
//  CHECK-NEXT:   %[[RES:.*]] = tensor.empty() : tensor<2x3xi32>
//  CHECK-NEXT:   %[[RES1:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[ARG1]], %[[ARG2]] : tensor<2x3xi32>, tensor<2x3xi32>) outs(%[[RES]] : tensor<2x3xi32>) {
//  CHECK-NEXT:   ^bb0(%[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32):
//  CHECK-NEXT:   %[[RES2:.*]] = arith.addi %[[ARG3]], %[[ARG4]] : i32
//  CHECK-NEXT:   linalg.yield %[[RES2]] : i32
//  CHECK-NEXT:   } -> tensor<2x3xi32>
//  CHECK-NEXT:   return %[[RES1]] : tensor<2x3xi32>

// -----

func.func @test(%arg: memref<?xsi32, strided<[?], offset: ?>>) -> memref<si32> {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg : memref<?xsi32, strided<[?], offset: ?>> -> memref<si32>, index, index, index
  return %base_buffer : memref<si32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32, strided<[?], offset: ?>>)
//  CHECK-NEXT:   %[[BASE:.*]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[ARG]] : memref<?xi32, strided<[?], offset: ?>> -> memref<i32>, index, index, index
//  CHECK-NEXT:   return %[[BASE]] : memref<i32>
