// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// FIXME sed above, for using 1 instead of true

// -----
func.func @test_subview(%arg0: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.subview %arg0[%c0][%c3][%c3] : tensor<?xi64> to tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_subview
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.subview %arg0[[[C0]]] [[[C1]]] [[[C1]]] : tensor<?xi64> to tensor<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<?xi64>

// -----
func.func @test_subview_const(%arg0: tensor<?xi64>) -> tensor<3xi64> {
    %0 = ndarray.subview %arg0[0][3][3] : tensor<?xi64> to tensor<3xi64>
    return %0 : tensor<3xi64>
}
// CHECK-LABEL: @test_subview_const
// CHECK-NEXT: [[V0:%.*]] = ndarray.subview %arg0[0] [3] [3] : tensor<?xi64> to tensor<3xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<3xi64>

// -----
func.func @test_extract_slice(%arg0: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.extract_slice %arg0[%c0][%c3][%c3] : tensor<?xi64> to tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_extract_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice %arg0[[[C0]]] [[[C1]]] [[[C1]]] : tensor<?xi64> to tensor<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<?xi64>

// -----
func.func @test_extract_slice_const(%arg0: tensor<?xi64>) -> tensor<3xi64> {
    %0 = ndarray.extract_slice %arg0[0][3][3] : tensor<?xi64> to tensor<3xi64>
    return %0 : tensor<3xi64>
}
// CHECK-LABEL: @test_extract_slice_const
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice %arg0[0] [3] [3] : tensor<?xi64> to tensor<3xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<3xi64>

// -----
func.func @test_insert_slice(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<?xi64>
    return %arg0 : tensor<?xi64>
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[[[C0]]] [[[C1]]] [[[C1]]] : tensor<?xi64> into tensor<?xi64>

// -----
func.func @test_insert_slice_const(%arg0: tensor<?xi64>, %arg1: tensor<3xi64>) -> tensor<?xi64> {
    ndarray.insert_slice %arg1 into %arg0[0] [3] [3] : tensor<3xi64> into tensor<?xi64>
    return %arg0 : tensor<?xi64>
}
// CHECK-LABEL: @test_insert_slice_const
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[0] [3] [3] : tensor<3xi64> into tensor<?xi64>

// -----
func.func @test_insert_slice_scalar(%arg0: tensor<?xi64>, %arg1: tensor<i64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%c0] [%c1] [%c3] : tensor<i64> into tensor<?xi64>
    return %arg0 : tensor<?xi64>
}
// CHECK-LABEL: @test_insert_slice_scalar
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[C3:%.*]] = arith.constant
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[[[C0]]] [[[C1]]] [[[C3]]] : tensor<i64> into tensor<?xi64>

// -----
func.func @test_immutable_insert_slice(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.immutable_insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<?xi64>
    return %arg0 : tensor<?xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice %arg1 into %arg0 [[[C0]]] [[[C1]]] [[[C1]]] : tensor<?xi64> into tensor<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<?xi64>

// -----
func.func @test_immutable_insert_slice_const(%arg0: tensor<?xi64>, %arg1: tensor<3xi64>) -> tensor<?xi64> {
    %0 = ndarray.immutable_insert_slice %arg1 into %arg0[0] [3] [3] : tensor<3xi64> into tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice_const
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice %arg1 into %arg0 [0] [3] [3] : tensor<3xi64> into tensor<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<?xi64>

// -----
func.func @test_linspace(%arg0: si64, %arg1: si64, %arg2: si64) -> tensor<?xi64> {
    %0 = ndarray.linspace %arg0 %arg1 %arg2 false : (si64, si64, si64) -> tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_linspace
// CHECK-NEXT: ndarray.linspace %arg0 %arg1 %arg2 false : (si64, si64, si64) -> tensor<?xi64>

// -----
func.func @test_reshape(%arg0: index) -> tensor<?x?xi64> {
    %0 = tensor.empty(%arg0) : tensor<?xi64>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = "ndarray.reshape"(%0, %c0, %c3) : (tensor<?xi64>, index, index) -> tensor<?x?xi64>
    return %1 : tensor<?x?xi64>
}
// CHECK-LABEL: @test_reshape
// CHECK: tensor.empty
// CHECK: ndarray.reshape
// CHECK-SAME: -> tensor<?x?xi64>

// -----
func.func @test_copy(%arg0: tensor<5xi64>) -> tensor<5xi64> {
    %0 = ndarray.copy %arg0 : tensor<5xi64> -> tensor<5xi64>
    return %0 : tensor<5xi64>
}
// CHECK-LABEL: func.func @test_copy
// CHECK: [[V0:%.*]] = ndarray.copy
// CHECK-NEXT: return [[V0]] : tensor<5xi64>

// -----
func.func @test_castelem(%arg0: tensor<5xi64>) -> tensor<5xi32> {
    %0 = ndarray.cast_elemtype %arg0 : tensor<5xi64> to tensor<5xi32>
    return %0 : tensor<5xi32>
}
// CHECK-LABEL: func.func @test_castelem
// CHECK: [[V0:%.*]] = ndarray.cast_elemtype
// CHECK-NEXT: return [[V0]] : tensor<5xi32>
