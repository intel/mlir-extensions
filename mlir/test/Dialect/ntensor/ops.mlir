// RUN: imex-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s -split-input-file | imex-opt | FileCheck %s

func.func @test(%arg: !ntensor.ntensor<2x3x4xf32>) -> !ntensor.ntensor<2x3x4xf32> {
  return %arg : !ntensor.ntensor<2x3x4xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<2x3x4xf32>)
//  CHECK-NEXT:   return %[[ARG]] : !ntensor.ntensor<2x3x4xf32>

// -----

func.func @test(%arg: !ntensor.ntensor<2x3x4xf32, "test">) -> !ntensor.ntensor<2x3x4xf32, "test"> {
  return %arg : !ntensor.ntensor<2x3x4xf32, "test">
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<2x3x4xf32, "test">)
//  CHECK-NEXT:   return %[[ARG]] : !ntensor.ntensor<2x3x4xf32, "test">

// -----

func.func @test(%arg: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.unary "-" (%arg : !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.unary "-" (%[[ARG]] : !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.binary "-" (%arg1 : !ntensor.ntensor<?xf32>, %arg2 : !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.binary "-" (%[[ARG1]] : !ntensor.ntensor<?xf32>, %[[ARG2]] : !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : index] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   ntensor.setitem(%[[ARG1]] : !ntensor.ntensor<?xf32>) [%[[ARG2]] : index] = (%[[ARG3]] : f32)
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index) -> f32 {
  %0 = ntensor.getitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : index] -> f32
  return %0 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.getitem(%[[ARG1]] : !ntensor.ntensor<?xf32>) [%[[ARG2]] : index] -> f32
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test() {
  ntensor.call "foo" ()
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   ntensor.call "foo" ()
//  CHECK-NEXT:   return

// -----

func.func @test() -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.call "foo" () -> !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.call "foo" () -> !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test() -> (!ntensor.ntensor<?xf32>, f32) {
  %0:2 = ntensor.call "foo" () -> !ntensor.ntensor<?xf32>, f32
  return %0#0, %0#1 : !ntensor.ntensor<?xf32>, f32
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[RES:.*]]:2 = ntensor.call "foo" () -> !ntensor.ntensor<?xf32>, f32
//  CHECK-NEXT:   return %[[RES]]#0, %[[RES]]#1 : !ntensor.ntensor<?xf32>, f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) {
  ntensor.call "foo" (%arg1) : !ntensor.ntensor<?xf32>
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   ntensor.call "foo" (%[[ARG1]]) : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index) {
  ntensor.call "foo" (%arg1, %arg2) : !ntensor.ntensor<?xf32>, index
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   ntensor.call "foo" (%[[ARG1]], %[[ARG2]]) : !ntensor.ntensor<?xf32>, index
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) {
  ntensor.call "foo" (bar:%arg1) : !ntensor.ntensor<?xf32>
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   ntensor.call "foo" (bar:%[[ARG1]]) : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return


// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index) {
  ntensor.call "foo" (%arg1, bar:%arg2) : !ntensor.ntensor<?xf32>, index
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   ntensor.call "foo" (%[[ARG1]], bar:%[[ARG2]]) : !ntensor.ntensor<?xf32>, index
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index) {
  ntensor.call "foo" ("baz":%arg1, bar:%arg2) : !ntensor.ntensor<?xf32>, index
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   ntensor.call "foo" (baz:%[[ARG1]], bar:%[[ARG2]]) : !ntensor.ntensor<?xf32>, index
//  CHECK-NEXT:   return

// -----

func.func @test() {
  ntensor.primitive "foo" ()
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   ntensor.primitive "foo" ()
//  CHECK-NEXT:   return

// -----

func.func @test() -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test() -> (!ntensor.ntensor<?xf32>, f32) {
  %0:2 = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>, f32
  return %0#0, %0#1 : !ntensor.ntensor<?xf32>, f32
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[RES:.*]]:2 = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>, f32
//  CHECK-NEXT:   return %[[RES]]#0, %[[RES]]#1 : !ntensor.ntensor<?xf32>, f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) {
  ntensor.primitive "foo" (%arg1) : !ntensor.ntensor<?xf32>
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   ntensor.primitive "foo" (%[[ARG1]]) : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index) {
  ntensor.primitive "foo" (%arg1, %arg2) : !ntensor.ntensor<?xf32>, index
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   ntensor.primitive "foo" (%[[ARG1]], %[[ARG2]]) : !ntensor.ntensor<?xf32>, index
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.dim %arg1, %0 : !ntensor.ntensor<?xf32>
  return %1 : index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG]], %[[IND]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[DIM]] : index

// -----

func.func @test() -> (!ntensor.slice, !ntensor.slice, !ntensor.slice, !ntensor.slice) {
  %0 = arith.constant 10 : index
  %1 = arith.constant 20 : index
  %2 = arith.constant 3 : index
  %3 = ntensor.build_slice (::)
  %4 = ntensor.build_slice (:%1:)
  %5 = ntensor.build_slice (%0:%1:)
  %6 = ntensor.build_slice (%0:%1:%2)
  return %3, %4, %5, %6 : !ntensor.slice, !ntensor.slice, !ntensor.slice, !ntensor.slice
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[BEGIN:.*]] = arith.constant 10 : index
//  CHECK-NEXT:   %[[END:.*]] = arith.constant 20 : index
//  CHECK-NEXT:   %[[STEP:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[S1:.*]] = ntensor.build_slice( : : )
//  CHECK-NEXT:   %[[S2:.*]] = ntensor.build_slice( : %[[END]] : )
//  CHECK-NEXT:   %[[S3:.*]] = ntensor.build_slice(%[[BEGIN]] : %[[END]] : )
//  CHECK-NEXT:   %[[S4:.*]] = ntensor.build_slice(%[[BEGIN]] : %[[END]] : %[[STEP]])
//  CHECK-NEXT:   %[[S1]], %[[S2]], %[[S3]], %[[S4]] : !ntensor.slice, !ntensor.slice, !ntensor.slice, !ntensor.slice

// -----

func.func @test(%arg1: index, %arg2: !ntensor.slice) -> (index, index, index, index) {
  %0:4 = ntensor.resolve_slice %arg2, %arg1
  return %0#0, %0#1, %0#2, %0#3 : index, index, index, index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: !ntensor.slice)
//  CHECK-NEXT:   %[[BEGIN:.*]], %[[END:.*]], %[[STEP:.*]] = ntensor.resolve_slice %[[ARG2]], %[[ARG1]]
//  CHECK-NEXT:   return %[[BEGIN]], %[[END]], %[[STEP]] : index, index, index

// -----

func.func @test(%arg1: index, %arg2: index) -> index {
  %0 = ntensor.resolve_index %arg1, %arg2
  return %0 : index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.resolve_index %[[ARG1]], %[[ARG2]]
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32, "test1">, %arg2: !ntensor.ntensor<?xf32, "test2">) {
  ntensor.copy %arg1, %arg2 : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32, "test2">
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32, "test1">, %[[ARG2:.*]]: !ntensor.ntensor<?xf32, "test2">)
//  CHECK-NEXT:   ntensor.copy %[[ARG1]], %[[ARG2]] : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32, "test2">
//  CHECK-NEXT:   return

// -----

func.func @test() -> !ntensor.ntensor<?x?xf32> {
  %0 = arith.constant 2 : index
  %1 = arith.constant 3 : index
  %3 = ntensor.create(%0, %1) : !ntensor.ntensor<?x?xf32>
  return %3 : !ntensor.ntensor<?x?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[D1:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[D2:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.create(%[[D1]], %[[D2]]) : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   return %[[RES]]

// -----

func.func @test() -> !ntensor.ntensor<?x?xi32> {
  %0 = arith.constant 2 : index
  %1 = arith.constant 3 : index
  %2 = arith.constant 5 : i32
  %3 = ntensor.create(%0, %1) = (%2 : i32) : !ntensor.ntensor<?x?xi32>
  return %3 : !ntensor.ntensor<?x?xi32>
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[D1:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[D2:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[VAL:.*]] = arith.constant 5 : i32
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.create(%[[D1]], %[[D2]]) = (%[[VAL]] : i32) : !ntensor.ntensor<?x?xi32>
//  CHECK-NEXT:   return %[[RES]]

// -----

// CHECK-LABEL: func @test({{.*}}) {
func.func @test(%t: !ntensor.ntensor<8x16x4xf32>, %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: ntensor.subview
  // CHECK-SAME: !ntensor.ntensor<8x16x4xf32> to !ntensor.ntensor<?x?x?xf32>
  %1 = ntensor.subview %t[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : !ntensor.ntensor<8x16x4xf32> to !ntensor.ntensor<?x?x?xf32>

  // CHECK: ntensor.subview
  // CHECK-SAME: !ntensor.ntensor<8x16x4xf32> to !ntensor.ntensor<4x4x4xf32>
  %2 = ntensor.subview %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : !ntensor.ntensor<8x16x4xf32> to !ntensor.ntensor<4x4x4xf32>

  // CHECK: ntensor.subview
  // CHECK-SAME: !ntensor.ntensor<8x16x4xf32> to !ntensor.ntensor<4x4xf32>
  %3 = ntensor.subview %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : !ntensor.ntensor<8x16x4xf32> to !ntensor.ntensor<4x4xf32>

  return
}

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = ntensor.load %arg1[%0] : !ntensor.ntensor<?xf32>
  return %1 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[ARG]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: f32) {
  %0 = arith.constant 0 : index
  ntensor.store %arg2, %arg1[%0] : !ntensor.ntensor<?xf32>
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: f32)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   ntensor.store %[[ARG2]], %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: tensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.from_tensor %arg1 : tensor<?xf32> to !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: tensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.from_tensor %[[ARG]] : tensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> tensor<?xf32> {
  %0 = ntensor.to_tensor %arg1 : !ntensor.ntensor<?xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.to_tensor %[[ARG]] : !ntensor.ntensor<?xf32> to tensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : tensor<?xf32>

// -----

func.func @test(%arg1: memref<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.from_memref %arg1 : memref<?xf32> to !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.from_memref %[[ARG]] : memref<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> memref<?xf32> {
  %0 = ntensor.to_memref %arg1 : !ntensor.ntensor<?xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.to_memref %[[ARG]] : !ntensor.ntensor<?xf32> to memref<?xf32>
//  CHECK-NEXT:   return %[[RES]] : memref<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.elementwise %arg1 : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32> {
  ^bb0(%arg2: f32):
    ntensor.elementwise_yield %arg2 : f32
  }
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.elementwise %[[ARG]] : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32> {
//  CHECK-NEXT:   ^bb0(%[[ARG1:.*]]: f32):
//  CHECK-NEXT:   ntensor.elementwise_yield %[[ARG1]] : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<5xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<5xf32>
  return %0 : !ntensor.ntensor<5xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.cast %[[ARG]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<5xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<5xf32>
