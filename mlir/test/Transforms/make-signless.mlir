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

func.func @test(%arg1: tensor<?xsi32>, %arg2: si32) -> tensor<?xsi32> {
  %0 = linalg.fill ins(%arg2 : si32) outs(%arg1 : tensor<?xsi32>) -> tensor<?xsi32>
  return %0 : tensor<?xsi32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tensor<?xi32>, %[[ARG2:.*]]: i32)
//  CHECK-NEXT:   %[[RES:.*]] = linalg.fill ins(%[[ARG2]] : i32) outs(%[[ARG1]] : tensor<?xi32>) -> tensor<?xi32>
//  CHECK-NEXT:   return %[[RES]] : tensor<?xi32>
