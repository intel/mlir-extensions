// RUN: imex-opt %s -canonicalize | FileCheck %s

func.func @test_subview(%arg0: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.subview %arg0[%c0][%c3][%c3] : tensor<?xi64> to tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_subview
// CHECK-NEXT: [[C0:%.*]] = ndarray.subview %arg0[0] [3] [3] : tensor<?xi64> to tensor<3xi64>
// CHECK-NEXT: [[C1:%.*]] = tensor.cast %0 : tensor<3xi64> to tensor<?xi64>

func.func @test_insert_slice(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<?xi64>
    return %arg0 : tensor<?xi64>
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[C0:%.*]] = tensor.cast %arg1 : tensor<?xi64> to tensor<3xi64>
// CHECK-NEXT: ndarray.insert_slice [[C0]] into %arg0[0] [3] [3] : tensor<3xi64> into tensor<?xi64>

func.func @test_insert_slice_cast(%arg0: tensor<5xi64>, %arg1: tensor<3xi64>) -> tensor<5xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = tensor.cast %arg0 : tensor<5xi64> to tensor<?xi64>
    %2 = tensor.cast %arg1 : tensor<3xi64> to tensor<?xi64>
    ndarray.insert_slice %2 into %1[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<?xi64>
    return %arg0 : tensor<5xi64>
}
// CHECK-LABEL: @test_insert_slice_cast
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[0] [3] [3] : tensor<3xi64> into tensor<5xi64>
// CHECK-NEXT: return %arg0 : tensor<5xi64>

func.func @test_linspace() -> tensor<?xi64> {
    %c4 = arith.constant 4 : index
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 8.000000e+00 : f64
    %0 = ndarray.linspace %cst_1 %cst_2 %c4 false : (f64, f64, index) -> tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: func.func @test_linspace() -> tensor<?xi64> {
// CHECK: ndarray.linspace
// CHECK-SAME: tensor<4xi64>
// CHECK-NEXT: tensor.cast
// CHECK-SAME: tensor<?xi64>
