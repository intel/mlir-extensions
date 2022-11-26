// RUN: imex-opt %s -allow-unregistered-dialect -canonicalize --split-input-file | FileCheck %s

func.func @resolve_slice_propagate() -> (index, index, index, index) {
  %0 = arith.constant 50 : index
  %1 = arith.constant 10 : index
  %2 = arith.constant 20 : index
  %3 = arith.constant 3 : index
  %4 = ntensor.build_slice (%1:%2:%3)
  %5:4 = ntensor.resolve_slice %4, %0
  return %5#0, %5#1, %5#2, %5#3 : index, index, index, index
}
// CHECK-LABEL: func @resolve_slice_propagate
//  CHECK-NEXT:   %[[COUNT:.*]] = arith.constant 4 : index
//  CHECK-NEXT:   %[[BEGIN:.*]] = arith.constant 10 : index
//  CHECK-NEXT:   %[[END:.*]] = arith.constant 20 : index
//  CHECK-NEXT:   %[[STEP:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   return %[[BEGIN]], %[[END]], %[[STEP]], %[[COUNT]] : index, index, index, index

// -----

func.func @resolve_slice_propagate() -> (index, index, index, index) {
  %0 = arith.constant 50 : index
  %1 = arith.constant 10 : index
  %2 = arith.constant 20 : index
  %4 = ntensor.build_slice (%1:%2:)
  %5:4 = ntensor.resolve_slice %4, %0
  return %5#0, %5#1, %5#2, %5#3 : index, index, index, index
}
// CHECK-LABEL: func @resolve_slice_propagate
//  CHECK-NEXT:   %[[COUNT:.*]] = arith.constant 10 : index
//  CHECK-NEXT:   %[[STEP:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[END:.*]] = arith.constant 20 : index
//  CHECK-NEXT:   return %[[COUNT]], %[[END]], %[[STEP]], %[[COUNT]] : index, index, index, index

// -----

func.func @resolve_slice_propagate() -> (index, index, index, index) {
  %0 = arith.constant 50 : index
  %2 = arith.constant 20 : index
  %4 = ntensor.build_slice (:%2:)
  %5:4 = ntensor.resolve_slice %4, %0
  return %5#0, %5#1, %5#2, %5#3 : index, index, index, index
}
// CHECK-LABEL: func @resolve_slice_propagate
//  CHECK-NEXT:   %[[BEGIN:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[STEP:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[END:.*]] = arith.constant 20 : index
//  CHECK-NEXT:   return %[[BEGIN]], %[[END]], %[[STEP]], %[[END]] : index, index, index, index

// -----

func.func @resolve_slice_propagate() -> (index, index, index, index) {
  %0 = arith.constant 50 : index
  %4 = ntensor.build_slice (::)
  %5:4 = ntensor.resolve_slice %4, %0
  return %5#0, %5#1, %5#2, %5#3 : index, index, index, index
}
// CHECK-LABEL: func @resolve_slice_propagate
//  CHECK-NEXT:   %[[BEGIN:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[STEP:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[END:.*]] = arith.constant 50 : index
//  CHECK-NEXT:   return %[[BEGIN]], %[[END]], %[[STEP]], %[[END]] : index, index, index, index

// -----

func.func @resolve_slice_propagate(%arg: index) -> (index, index, index, index) {
  %1 = arith.constant 10 : index
  %2 = arith.constant 20 : index
  %3 = arith.constant 3 : index
  %4 = ntensor.build_slice (%1:%2:%3)
  %5:4 = ntensor.resolve_slice %4, %arg
  return %5#0, %5#1, %5#2, %5#3 : index, index, index, index
}
// CHECK-LABEL: func @resolve_slice_propagate
//  CHECK-SAME:   (%[[ARG:.*]]: index)
//  CHECK-NEXT:   %[[C2:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[BEGIN:.*]] = arith.constant 10 : index
//  CHECK-NEXT:   %[[END:.*]] = arith.constant 20 : index
//  CHECK-NEXT:   %[[STEP:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[CMP1:.*]] = arith.cmpi sle, %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   %[[BEGIN2:.*]] = arith.select %[[CMP1]], %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   %[[CMP2:.*]] = arith.cmpi sle, %[[ARG]], %[[END]] : index
//  CHECK-NEXT:   %[[END2:.*]] = arith.select %[[CMP2]], %[[ARG]], %[[END]] : index
//  CHECK-NEXT:   %[[V4:.*]] = arith.subi %[[END2]], %1 : index
//  CHECK-NEXT:   %[[V5:.*]] = arith.addi %[[V4]], %[[C2]] : index
//  CHECK-NEXT:   %[[COUNT:.*]] = arith.divui %[[V5]], %[[STEP]] : index
//  CHECK-NEXT:   return %[[BEGIN2]], %[[END2]], %[[STEP]], %[[COUNT]] : index, index, index, index

// -----

func.func @resolve_index_propagate(%arg: index) -> index{
  %1 = arith.constant 10 : index
  %2 = ntensor.resolve_index %1, %arg
  return %2 : index
}
// CHECK-LABEL: func @resolve_index_propagate
//  CHECK-SAME:   (%[[ARG:.*]]: index)
//  CHECK-NEXT:   %[[BEGIN:.*]] = arith.constant 10 : index
//  CHECK-NEXT:   %[[CMP1:.*]] = arith.cmpi sle, %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   %[[BEGIN2:.*]] = arith.select %[[CMP1]], %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   return %[[BEGIN2]] : index

// -----

func.func @test(%arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = ntensor.from_tensor %arg1 : tensor<?xf32> to !ntensor.ntensor<?xf32>
  %1 = ntensor.to_tensor %0 : !ntensor.ntensor<?xf32> to tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: tensor<?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : tensor<?xf32>

// -----

func.func @test(%arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = ntensor.from_tensor %arg1 : tensor<?xf32> to !ntensor.ntensor<?xf32>
  "test.test5"(%0) : (!ntensor.ntensor<?xf32>) -> ()
  %1 = ntensor.to_tensor %0 : !ntensor.ntensor<?xf32> to tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: tensor<?xf32>)
//  CHECK-NEXT:   %[[TMP:.*]] = ntensor.from_tensor %[[ARG]] : tensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   "test.test5"(%[[TMP]]) : (!ntensor.ntensor<?xf32>) -> ()
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.to_tensor %[[TMP]] : !ntensor.ntensor<?xf32> to tensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : tensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.to_tensor %arg1 : !ntensor.ntensor<?xf32> to tensor<?xf32>
  %1 = ntensor.from_tensor %0 : tensor<?xf32> to !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: memref<?xf32>) -> memref<?xf32> {
  %0 = ntensor.from_memref %arg1 : memref<?xf32> to !ntensor.ntensor<?xf32>
  %1 = ntensor.to_memref %0 : !ntensor.ntensor<?xf32> to memref<?xf32>
  return %1 : memref<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : memref<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.to_memref %arg1 : !ntensor.ntensor<?xf32> to memref<?xf32>
  %1 = ntensor.from_memref %0 : memref<?xf32> to !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   return %[[ARG]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<5xf32>
  %1 = ntensor.cast %0 : !ntensor.ntensor<5xf32> to !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   return %[[ARG]]

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32, "test1">
  %1 = ntensor.cast %0 : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32, "test2">
  %2 = ntensor.cast %1 : !ntensor.ntensor<?xf32, "test2"> to !ntensor.ntensor<?xf32>
  return %2 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   return %[[ARG]]

// -----

func.func @test(%arg1: tensor<?x?xf32>) -> index {
  %0 = arith.constant 1 : index
  %1 = ntensor.from_tensor %arg1 : tensor<?x?xf32> to !ntensor.ntensor<?x?xf32>
  %2 = ntensor.dim %1, %0 : !ntensor.ntensor<?x?xf32>
  return %2 : index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: tensor<?x?xf32>)
//       CHECK:   %[[IND:.*]] = arith.constant 1 : index
//       CHECK:   %[[DIM:.*]] = tensor.dim %[[ARG]], %[[IND]] : tensor<?x?xf32>
//       CHECK:   return %[[DIM]] : index

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32>) -> index {
  %0 = arith.constant 1 : index
  %1 = ntensor.to_tensor %arg1 : !ntensor.ntensor<?x?xf32> to tensor<?x?xf32>
  %2 = tensor.dim %1, %0 : tensor<?x?xf32>
  return %2 : index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?x?xf32>)
//       CHECK:   %[[IND:.*]] = arith.constant 1 : index
//       CHECK:   %[[DIM:.*]] = ntensor.dim %[[ARG]], %[[IND]] : !ntensor.ntensor<?x?xf32>
//       CHECK:   return %[[DIM]] : index

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?x?xf32>)
//  CHECK-NEXT:   return %[[ARG1]] : !ntensor.ntensor<?x?xf32>
func.func @test(%arg1: !ntensor.ntensor<?x?xf32>) -> (!ntensor.ntensor<?x?xf32>) {
  %0 = ntensor.broadcast (%arg1) : !ntensor.ntensor<?x?xf32> -> !ntensor.ntensor<?x?xf32>
  return %0#0 : !ntensor.ntensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<2x3xf32>, %[[ARG2:.*]]: !ntensor.ntensor<2x3xf32>)
//  CHECK-NEXT:   %[[RES1:.*]] = ntensor.cast %[[ARG1]] : !ntensor.ntensor<2x3xf32> to !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   %[[RES2:.*]] = ntensor.cast %[[ARG2]] : !ntensor.ntensor<2x3xf32> to !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   return %[[RES1]], %[[RES2]] : !ntensor.ntensor<?x?xf32>, !ntensor.ntensor<?x?xf32>
func.func @test(%arg1: !ntensor.ntensor<2x3xf32>, %arg2: !ntensor.ntensor<2x3xf32>) -> (!ntensor.ntensor<?x?xf32>, !ntensor.ntensor<?x?xf32>) {
  %0:2 = ntensor.broadcast (%arg1, %arg2) : !ntensor.ntensor<2x3xf32>, !ntensor.ntensor<2x3xf32> -> !ntensor.ntensor<?x?xf32>, !ntensor.ntensor<?x?xf32>
  return %0#0, %0#1 : !ntensor.ntensor<?x?xf32>, !ntensor.ntensor<?x?xf32>
}

// -----

func.func @test(%arg1: !ntensor.ntensor<2xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = ntensor.cast %arg1 : !ntensor.ntensor<2xf32> to !ntensor.ntensor<?xf32>
  %2 = ntensor.load %1[%0] : !ntensor.ntensor<?xf32>
  return %2 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<2xf32>)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[ARG]][%[[IND]]] : !ntensor.ntensor<2xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<2xf32>, %arg2: f32) {
  %0 = arith.constant 0 : index
  %1 = ntensor.cast %arg1 : !ntensor.ntensor<2xf32> to !ntensor.ntensor<?xf32>
  ntensor.store %arg2, %1[%0] : !ntensor.ntensor<?xf32>
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<2xf32>, %[[ARG2:.*]]: f32)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   ntensor.store %[[ARG2:.*]] %[[ARG1]][%[[IND]]] : !ntensor.ntensor<2xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<2xf32>) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.dim %arg1, %0 : !ntensor.ntensor<2xf32>
  return %1 : index
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[RES:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: !ntensor.ntensor<2xf32>) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.cast %arg1 : !ntensor.ntensor<2xf32> to !ntensor.ntensor<?xf32>
  %2 = ntensor.dim %1, %0 : !ntensor.ntensor<?xf32>
  return %2 : index
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[RES:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   return %[[RES]] : index
