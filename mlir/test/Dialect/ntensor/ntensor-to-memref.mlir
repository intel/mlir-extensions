// RUN: imex-opt %s -ntensor-to-memref --split-input-file | FileCheck %s

func.func @test(%arg: !ntensor.ntensor<?x?xf32>) -> !ntensor.ntensor<?x?xf32> {
  return %arg : !ntensor.ntensor<?x?xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?x?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : memref<?x?xf32>

// -----

func.func @test(%arg: !ntensor.ntensor<?x?xf32>) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.dim %arg, %0 : !ntensor.ntensor<?x?xf32>
  return %1 : index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?x?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = memref.dim %[[ARG]], %[[C0]] : memref<?x?xf32>
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg: !ntensor.ntensor<?x?xf32, "test">) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.dim %arg, %0 : !ntensor.ntensor<?x?xf32, "test">
  return %1 : index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?x?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> index {
//  CHECK-NEXT:   %[[RES1:.*]] = memref.dim %[[ARG]], %[[C0]] : memref<?x?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32>, %arg2: index, %arg3: index, %arg4: index) -> !ntensor.ntensor<?x?xf32> {
  %1 = ntensor.subview %arg1[1, %arg2][2, %arg3][3, %arg4] : !ntensor.ntensor<?x?xf32> to !ntensor.ntensor<?x?xf32>
  return %1 : !ntensor.ntensor<?x?xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = memref.subview %arg0[1, %[[ARG2]]] [2, %[[ARG3]]] [3, %[[ARG4]]] : memref<?x?xf32> to memref<2x?xf32, strided<[?, ?], offset: ?>>
//  CHECK-NEXT:   %[[RES2:.*]] = imex_util.change_layout %[[RES]] : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
//  CHECK-NEXT:   return %[[RES2]] : memref<?x?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32, "test">, %arg2: index, %arg3: index, %arg4: index) -> !ntensor.ntensor<?x?xf32, "test"> {
  %1 = ntensor.subview %arg1[1, %arg2][2, %arg3][3, %arg4] : !ntensor.ntensor<?x?xf32, "test"> to !ntensor.ntensor<?x?xf32, "test">
  return %1 : !ntensor.ntensor<?x?xf32, "test">
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> memref<?x?xf32> {
//  CHECK-NEXT:   %[[RES1:.*]] = memref.subview %arg0[1, %[[ARG2]]] [2, %[[ARG3]]] [3, %[[ARG4]]] : memref<?x?xf32> to memref<2x?xf32, strided<[?, ?], offset: ?>>
//  CHECK-NEXT:   %[[RES2:.*]] = imex_util.change_layout %[[RES1]] : memref<2x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES2]] : memref<?x?xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : memref<?x?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32>, %arg2: index, %arg3: index, %arg4: index) -> !ntensor.ntensor<?xf32> {
  %1 = ntensor.subview %arg1[1, %arg2][1, %arg3][3, %arg4] : !ntensor.ntensor<?x?xf32> to !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = memref.subview %arg0[1, %[[ARG2]]] [1, %[[ARG3]]] [3, %[[ARG4]]] : memref<?x?xf32> to memref<?xf32, strided<[?], offset: ?>>
//  CHECK-NEXT:   %[[RES2:.*]] = imex_util.change_layout %[[RES]] : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
//  CHECK-NEXT:   return %[[RES2]] : memref<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32, "test">, %arg2: index, %arg3: index, %arg4: index) -> !ntensor.ntensor<?xf32, "test"> {
  %1 = ntensor.subview %arg1[1, %arg2][1, %arg3][3, %arg4] : !ntensor.ntensor<?x?xf32, "test"> to !ntensor.ntensor<?xf32, "test">
  return %1 : !ntensor.ntensor<?xf32, "test">
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> memref<?xf32> {
//  CHECK-NEXT:   %[[RES1:.*]] = memref.subview %arg0[1, %[[ARG2]]] [1, %[[ARG3]]] [3, %[[ARG4]]] : memref<?x?xf32> to memref<?xf32, strided<[?], offset: ?>>
//  CHECK-NEXT:   %[[RES2:.*]] = imex_util.change_layout %[[RES1]] : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES2]] : memref<?xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : memref<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = ntensor.load %arg1[%0] : !ntensor.ntensor<?xf32>
  return %1 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = memref.load %[[ARG]][%[[IND]]] : memref<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">) -> f32 {
  %0 = arith.constant 0 : index
  %1 = ntensor.load %arg1[%0] : !ntensor.ntensor<?xf32, "test">
  return %1 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> f32 {
//  CHECK-NEXT:   %[[RES1:.*]] = memref.load %[[ARG]][%[[IND]]] : memref<?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES1]] : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: f32) {
  %0 = arith.constant 0 : index
  ntensor.store %arg2, %arg1[%0] : !ntensor.ntensor<?xf32>
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: f32)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   memref.store %[[ARG2]], %[[ARG1]][%[[IND]]] : memref<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">, %arg2: f32) {
  %0 = arith.constant 0 : index
  ntensor.store %arg2, %arg1[%0] : !ntensor.ntensor<?xf32, "test">
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: f32)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   imex_util.env_region "test" {
//  CHECK-NEXT:   memref.store %[[ARG2]], %[[ARG1]][%[[IND]]] : memref<?xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: tensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.from_tensor %arg1 : tensor<?xf32> to !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: tensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = bufferization.to_memref %[[ARG]] : memref<?xf32>
//  CHECK-NEXT:   return %[[RES]] : memref<?xf32>

// -----

func.func @test(%arg1: tensor<?xf32>) -> !ntensor.ntensor<?xf32, "test"> {
  %0 = ntensor.from_tensor %arg1 : tensor<?xf32> to !ntensor.ntensor<?xf32, "test">
  return %0 : !ntensor.ntensor<?xf32, "test">
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: tensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> memref<?xf32> {
//  CHECK-NEXT:   %[[RES1:.*]] = bufferization.to_memref %[[ARG]] : memref<?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES1]] : memref<?xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : memref<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> tensor<?xf32> {
  %0 = ntensor.to_tensor %arg1 : !ntensor.ntensor<?xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = bufferization.to_tensor %[[ARG]] : memref<?xf32>
//  CHECK-NEXT:   return %[[RES]] : tensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">) -> tensor<?xf32> {
  %0 = ntensor.to_tensor %arg1 : !ntensor.ntensor<?xf32, "test"> to tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> tensor<?xf32> {
//  CHECK-NEXT:   %[[RES1:.*]] = bufferization.to_tensor %[[ARG]] : memref<?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES1]] : tensor<?xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : tensor<?xf32>

// -----

func.func @test(%arg1: memref<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.from_memref %arg1 : memref<?xf32> to !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : memref<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> memref<?xf32> {
  %0 = ntensor.to_memref %arg1 : !ntensor.ntensor<?xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : memref<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<5xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<5xf32>
  return %0 : !ntensor.ntensor<5xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = memref.cast %[[ARG]] : memref<?xf32> to memref<5xf32>
//  CHECK-NEXT:   return %[[RES]] : memref<5xf32>
