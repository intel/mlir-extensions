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

func.func @test_extract_slice(%arg0: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.extract_slice %arg0[%c0][%c3][%c3] : tensor<?xi64> to tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_extract_slice
// CHECK-NEXT: [[C0:%.*]] = ndarray.extract_slice %arg0[0] [3] [3] : tensor<?xi64> to tensor<3xi64>
// CHECK-NEXT: [[C1:%.*]] = tensor.cast %0 : tensor<3xi64> to tensor<?xi64>

func.func @test_extract_slice_cast(%arg0: tensor<5xi64>) -> tensor<3xi64> {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.extract_slice %arg0[%c1] [%c3] [%c1] : tensor<5xi64> to tensor<?xi64>
    %1 = tensor.cast %0 : tensor<?xi64> to tensor<3xi64>
    return %1 : tensor<3xi64>
}
// CHECK-LABEL: @test_extract_slice_cast
// CHECK-NEXT: [[C0:%.*]] = ndarray.extract_slice %arg0[1] [3] [1] : tensor<5xi64> to tensor<3xi64>

func.func @test_extract_slice_cast2(%arg0: tensor<5xi64>) -> tensor<3xi64> {
    %0 = ndarray.extract_slice %arg0[0] [3] [1] : tensor<5xi64> to tensor<3xi64>
    %1 = tensor.cast %0 : tensor<3xi64> to tensor<?xi64>
    %2 = ndarray.extract_slice %1[0] [3] [1] : tensor<?xi64> to tensor<3xi64>
    return %2 : tensor<3xi64>
}
// CHECK-LABEL: @test_extract_slice_cast2
// CHECK-NEXT: [[C0:%.*]] = ndarray.extract_slice %arg0[0] [3] [1] : tensor<5xi64> to tensor<3xi64>
// CHECK-NEXT: [[C1:%.*]] = ndarray.extract_slice %0[0] [3] [1] : tensor<3xi64> to tensor<3xi64>
// CHECK-NEXT: return [[C1:%.*]] : tensor<3xi64>

func.func @test_extract_immutable_insert_slice(%arg0: tensor<16x16xi64>) -> tensor<3x3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.empty(%c3, %c3) : tensor<?x?xi64>
    %1 = tensor.empty(%c3, %c3) : tensor<?x?xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0, %c0] [%c3, %c3] [%c1, %c1] : tensor<?x?xi64> into tensor<16x16xi64>
    %3 = ndarray.immutable_insert_slice %1 into %2[%c0, %c3] [%c3, %c3] [%c1, %c1] : tensor<?x?xi64> into tensor<16x16xi64>
    %4 = ndarray.extract_slice %3[0, 0] [3, 3] [1, 1] : tensor<16x16xi64> to tensor<3x3xi64>
    return %4 : tensor<3x3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice
// CHECK-NEXT: [[V0:%.*]] = tensor.empty
// CHECK-NEXT: return [[V0]] : tensor<3x3xi64>

func.func @test_extract_immutable_insert_slice_overwrite(%arg0: tensor<16x16xi64>) -> tensor<3x3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.empty(%c3, %c3) : tensor<?x?xi64>
    %1 = tensor.empty(%c3, %c3) : tensor<?x?xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0, %c0] [%c3, %c3] [%c1, %c1] : tensor<?x?xi64> into tensor<16x16xi64>
    %3 = ndarray.immutable_insert_slice %1 into %2[%c0, %c1] [%c3, %c3] [%c1, %c1] : tensor<?x?xi64> into tensor<16x16xi64>
    %4 = ndarray.extract_slice %3[0, 0] [3, 3] [1, 1] : tensor<16x16xi64> to tensor<3x3xi64>
    return %4 : tensor<3x3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice_overwrite
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: ndarray.immutable_insert_slice
// CHECK-NEXT: ndarray.immutable_insert_slice
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice
// CHECK-NEXT: return [[V0]] : tensor<3x3xi64>

func.func @test_extract_immutable_insert_slice_strided(%arg0: tensor<16xi64>) -> tensor<3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.empty(%c3) : tensor<?xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<16xi64>
    %4 = ndarray.extract_slice %2[0] [3] [%c1] : tensor<16xi64> to tensor<3xi64>
    return %4 : tensor<3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice_strided
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: ndarray.immutable_insert_slice
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice
// CHECK-NEXT: return [[V0]] : tensor<3xi64>

func.func @test_extract_immutable_insert_slice_strided2(%arg0: tensor<16xi64>) -> tensor<3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.empty(%c3) : tensor<?xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<16xi64>
    %4 = ndarray.extract_slice %2[0] [3] [%c3] : tensor<16xi64> to tensor<3xi64>
    return %4 : tensor<3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice_strided2
// CHECK-NEXT: [[V0:%.*]] = tensor.empty
// CHECK-NEXT: return [[V0]] : tensor<3xi64>

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

func.func @test_immutable_insert_slice(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.immutable_insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice
// CHECK-NEXT: [[C0:%.*]] = tensor.cast %arg1 : tensor<?xi64> to tensor<3xi64>
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice [[C0]] into %arg0 [0] [3] [3] : tensor<3xi64> into tensor<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<?xi64>

func.func @test_immutable_insert_slice_cast(%arg0: tensor<5xi64>, %arg1: tensor<3xi64>) -> tensor<5xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = tensor.cast %arg0 : tensor<5xi64> to tensor<?xi64>
    %2 = tensor.cast %arg1 : tensor<3xi64> to tensor<?xi64>
    %3 = ndarray.immutable_insert_slice %2 into %1[%c0] [%c3] [%c3] : tensor<?xi64> into tensor<?xi64>
    %4 = tensor.cast %3 : tensor<?xi64> to tensor<5xi64>
    return %4 : tensor<5xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice_cast
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice %arg1 into %arg0 [0] [3] [3] : tensor<3xi64> into tensor<5xi64>
// CHECK-NEXT: return [[V0:%.*]] : tensor<5xi64>

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
