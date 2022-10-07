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
//  CHECK-NEXT:   %[[CMP1:.*]] = arith.cmpi slt, %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   %[[BEGIN2:.*]] = arith.select %[[CMP1]], %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   %[[CMP2:.*]] = arith.cmpi slt, %[[ARG]], %[[END]] : index
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
//  CHECK-NEXT:   %[[CMP1:.*]] = arith.cmpi slt, %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   %[[BEGIN2:.*]] = arith.select %[[CMP1]], %[[ARG]], %[[BEGIN]] : index
//  CHECK-NEXT:   return %[[BEGIN2]] : index
