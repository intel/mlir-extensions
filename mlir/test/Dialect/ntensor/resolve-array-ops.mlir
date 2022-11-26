// RUN: imex-opt --ntensor-resolve-array-ops --split-input-file %s | FileCheck %s

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index) -> f32 {
  %0 = ntensor.getitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : index] -> f32
  return %0 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: i32) -> f32 {
  %0 = ntensor.getitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : i32] -> f32
  return %0 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: i32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[IDIM:.*]] = arith.index_cast %[[ARG2]] : i32 to index
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[IDIM]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: si32) -> f32 {
  %0 = ntensor.getitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : si32] -> f32
  return %0 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: si32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[SDIM:.*]] = imex_util.sign_cast %[[ARG2]] : si32 to i32
//  CHECK-NEXT:   %[[IDIM:.*]] = arith.index_cast %[[SDIM]] : i32 to index
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[IDIM]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: index, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : index] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   ntensor.store %[[ARG3]], %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: i32, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : i32] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[IDIM:.*]] = arith.index_cast %[[ARG2]] : i32 to index
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[IDIM]], %[[DIM]]
//  CHECK-NEXT:   ntensor.store %[[ARG3]], %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: si32, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : si32] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: si32, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[SDIM:.*]] = imex_util.sign_cast %[[ARG2]] : si32 to i32
//  CHECK-NEXT:   %[[IDIM:.*]] = arith.index_cast %[[SDIM]] : i32 to index
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[IDIM]], %[[DIM]]
//  CHECK-NEXT:   ntensor.store %[[ARG3]], %[[ARG1]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.slice) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.getitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : !ntensor.slice] -> !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.slice)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[BEGIN:.*]], %[[END:.*]], %[[STEP:.*]], %[[COUNT:.*]] = ntensor.resolve_slice %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.subview %[[ARG1]][%[[BEGIN]]] [%[[COUNT]]] [%[[STEP]]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.slice, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : !ntensor.slice] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.slice, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[BEGIN:.*]], %[[END:.*]], %[[STEP:.*]], %[[COUNT:.*]] = ntensor.resolve_slice %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.subview %[[ARG1]][%[[BEGIN]]] [%[[COUNT]]] [%[[STEP]]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[DIM2:.*]] = ntensor.dim %[[RES]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[RES2:.*]] = ntensor.create(%[[DIM2]]) = (%[[ARG3]] : f32) : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   ntensor.copy %[[RES2]], %[[RES]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32>, %arg2: tuple<index, index>) -> f32 {
  %0 = ntensor.getitem(%arg1 : !ntensor.ntensor<?x?xf32>) [%arg2 : tuple<index, index>] -> f32
  return %0 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?x?xf32>, %[[ARG2:.*]]: tuple<index, index>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[E1:.*]] = imex_util.tuple_extract %[[ARG2]] : tuple<index, index>, %[[C0]] -> index
//  CHECK-NEXT:   %[[DIM1:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   %[[IND1:.*]] = ntensor.resolve_index %[[E1]], %[[DIM1]]
//  CHECK-NEXT:   %[[E2:.*]] = imex_util.tuple_extract %[[ARG2]] : tuple<index, index>, %[[C1]] -> index
//  CHECK-NEXT:   %[[DIM2:.*]] = ntensor.dim %[[ARG1]], %[[C1]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   %[[IND2:.*]] = ntensor.resolve_index %[[E2]], %[[DIM2]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[ARG1]][%[[IND1]], %[[IND2]]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32>, %arg2: tuple<index, index>, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?x?xf32>) [%arg2 : tuple<index, index>] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?x?xf32>, %[[ARG2:.*]]: tuple<index, index>, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[E1:.*]] = imex_util.tuple_extract %[[ARG2]] : tuple<index, index>, %[[C0]] -> index
//  CHECK-NEXT:   %[[DIM1:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   %[[IND1:.*]] = ntensor.resolve_index %[[E1]], %[[DIM1]]
//  CHECK-NEXT:   %[[E2:.*]] = imex_util.tuple_extract %[[ARG2]] : tuple<index, index>, %[[C1]] -> index
//  CHECK-NEXT:   %[[DIM2:.*]] = ntensor.dim %[[ARG1]], %[[C1]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   %[[IND2:.*]] = ntensor.resolve_index %[[E2]], %[[DIM2]]
//  CHECK-NEXT:   ntensor.store %[[ARG3]], %[[ARG1]][%[[IND1]], %[[IND2]]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.slice, %arg3: !ntensor.ntensor<?xf32>) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : !ntensor.slice] = (%arg3 : !ntensor.ntensor<?xf32>)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.slice, %[[ARG3:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[BEGIN:.*]], %[[END:.*]], %[[STEP:.*]], %[[COUNT:.*]] = ntensor.resolve_slice %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.subview %[[ARG1]][%[[BEGIN]]] [%[[COUNT]]] [%[[STEP]]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   ntensor.copy %[[ARG3]], %[[RES]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32>, %arg2: tuple<!ntensor.slice, index>, %arg3: !ntensor.ntensor<?xf32>) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?x?xf32>) [%arg2 : tuple<!ntensor.slice, index>] = (%arg3 : !ntensor.ntensor<?xf32>)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?x?xf32>, %[[ARG2:.*]]: tuple<!ntensor.slice, index>, %[[ARG3:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[IDX1:.*]] = imex_util.tuple_extract %[[ARG2]] : tuple<!ntensor.slice, index>, %[[C0]] -> !ntensor.slice
//  CHECK-NEXT:   %[[DIM1:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   %[[BEGIN:.*]], %[[END:.*]], %[[STEP:.*]], %[[COUNT:.*]] = ntensor.resolve_slice %[[IDX1]], %[[DIM1]]
//  CHECK-NEXT:   %[[IDX2:.*]] = imex_util.tuple_extract %[[ARG2]] : tuple<!ntensor.slice, index>, %[[C1]] -> index
//  CHECK-NEXT:   %[[DIM2:.*]]  = ntensor.dim %[[ARG1]], %[[C1]] : !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   %[[IDX3:.*]] = ntensor.resolve_index %[[IDX2]], %[[DIM2]]
//  CHECK-NEXT:   %[[RES1:.*]] = ntensor.subview %[[ARG1]][%[[BEGIN]], %[[IDX3]]] [%[[COUNT]], 1] [%[[STEP]], 1] : !ntensor.ntensor<?x?xf32> to !ntensor.ntensor<?x1xf32>
//  CHECK-NEXT:   %[[RES2:.*]]  = ntensor.subview %[[RES1]][0, 0] [%[[COUNT]], 1] [1, 1] : !ntensor.ntensor<?x1xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   ntensor.copy %[[ARG3]], %[[RES2]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return


// -----

func.func @test(%arg1: tuple<f32, f32>, %arg2: index) -> f32 {
  %0 = ntensor.getitem(%arg1 : tuple<f32, f32>) [%arg2 : index] -> f32
  return %0 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tuple<f32, f32>, %[[ARG2:.*]]: index)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[C2:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[E1:.*]] = imex_util.tuple_extract %[[ARG1]] : tuple<f32, f32>, %[[C0]] -> f32
//  CHECK-NEXT:   %[[E2:.*]] = imex_util.tuple_extract %[[ARG1]] : tuple<f32, f32>, %[[C1]] -> f32
//  CHECK-NEXT:   %[[T1:.*]] = ntensor.from_elements %[[E1]], %[[E2]] : !ntensor.ntensor<2xf32>
//  CHECK-NEXT:   %[[T2:.*]] = ntensor.cast %[[T1]] : !ntensor.ntensor<2xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[IND:.*]] = ntensor.resolve_index %[[ARG2]], %[[C2]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.load %[[T2]][%[[IND]]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf64>, %arg2: !ntensor.slice, %arg3: !ntensor.ntensor<?xf32>) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf64>) [%arg2 : !ntensor.slice] = (%arg3 : !ntensor.ntensor<?xf32>)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf64>, %[[ARG2:.*]]: !ntensor.slice, %[[ARG3:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   %[[BEGIN:.*]], %[[END:.*]], %[[STEP:.*]], %[[COUNT:.*]] = ntensor.resolve_slice %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.subview %[[ARG1]][%[[BEGIN]]] [%[[COUNT]]] [%[[STEP]]] : !ntensor.ntensor<?xf64> to !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   %[[RES1:.*]]  = ntensor.elementwise %[[ARG3]] : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf64> {
//  CHECK-NEXT:   ^bb0(%[[ARG4:.*]]: f32):
//  CHECK-NEXT:   %[[V:.*]] = arith.extf %[[ARG4]] : f32 to f64
//  CHECK-NEXT:   ntensor.elementwise_yield %[[V]]  : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   ntensor.copy %[[RES1]], %[[RES]] : !ntensor.ntensor<?xf64> to !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf64>, %arg2: !ntensor.slice, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf64>) [%arg2 : !ntensor.slice] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf64>, %[[ARG2:.*]]: !ntensor.slice, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   %[[BEGIN:.*]], %[[END:.*]], %[[STEP:.*]], %[[COUNT:.*]] = ntensor.resolve_slice %[[ARG2]], %[[DIM]]
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.subview %[[ARG1]][%[[BEGIN]]] [%[[COUNT]]] [%[[STEP]]] : !ntensor.ntensor<?xf64> to !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   %[[DIM2:.*]] = ntensor.dim %[[RES]], %[[C0]] : !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   %[[CONV:.*]] = arith.extf %[[ARG3]] : f32 to f64
//  CHECK-NEXT:   %[[RES2:.*]] = ntensor.create(%[[DIM2]]) = (%[[CONV]] : f64) : !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   ntensor.copy %[[RES2]], %[[RES]] : !ntensor.ntensor<?xf64> to !ntensor.ntensor<?xf64>
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.ntensor<?xi1>, %arg3: f32) {
  ntensor.setitem(%arg1 : !ntensor.ntensor<?xf32>) [%arg2 : !ntensor.ntensor<?xi1>] = (%arg3 : f32)
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.ntensor<?xi1>, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[DIM:.*]] = ntensor.dim %[[ARG1]], %[[C0]] : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[VAL:.*]] = ntensor.create(%[[DIM]]) = (%[[ARG3]] : f32) : !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   %[[TMP:.*]] = ntensor.elementwise %[[ARG2]], %[[VAL]], %[[ARG1]] : !ntensor.ntensor<?xi1>, !ntensor.ntensor<?xf32>, !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32> {
//  CHECK-NEXT:   ^bb0(%[[ARG4:.*]]: i1, %[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32):
//  CHECK-NEXT:   %[[YIELD:.*]] = arith.select %[[ARG4]], %[[ARG5]], %[[ARG6]] : f32
//  CHECK-NEXT:   ntensor.elementwise_yield %[[YIELD]]  : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT:   ntensor.copy %[[TMP]], %[[ARG1]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT:   return
