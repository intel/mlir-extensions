// RUN: imex-opt %s -canonicalize | FileCheck %s

func.func @test_subview(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.subview %arg0[%c0][%c3][%c3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_subview
// CHECK-NEXT: [[C0:%.*]] = ndarray.subview %arg0[0] [3] [3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: [[C1:%.*]] = ndarray.cast %0 : !ndarray.ndarray<3xi64> to !ndarray.ndarray<?xi64>

func.func @test_subview_cast(%arg0: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<3xi64> {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.subview %arg0[%c1] [%c3] [%c1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<?xi64>
    %1 = ndarray.cast %0 : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
    return %1 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: @test_subview_cast
// CHECK-NEXT: [[C0:%.*]] = ndarray.subview %arg0[1] [3] [1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<3xi64>

func.func @test_subview_cast2(%arg0: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<3xi64> {
    %0 = ndarray.subview %arg0[0] [3] [1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<3xi64>
    %1 = ndarray.cast %0 : !ndarray.ndarray<3xi64> to !ndarray.ndarray<?xi64>
    %2 = ndarray.subview %1[0] [3] [1] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
    return %2 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: @test_subview_cast2
// CHECK-NEXT: [[C0:%.*]] = ndarray.subview %arg0[0] [3] [1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: [[C1:%.*]] = ndarray.subview %0[0] [3] [1] : !ndarray.ndarray<3xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: return [[C1:%.*]] : !ndarray.ndarray<3xi64>

func.func @test_extract_slice(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.extract_slice %arg0[%c0][%c3][%c3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_extract_slice
// CHECK-NEXT: [[C0:%.*]] = ndarray.extract_slice %arg0[0] [3] [3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: [[C1:%.*]] = ndarray.cast %0 : !ndarray.ndarray<3xi64> to !ndarray.ndarray<?xi64>

func.func @test_extract_slice_cast(%arg0: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<3xi64> {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.extract_slice %arg0[%c1] [%c3] [%c1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<?xi64>
    %1 = ndarray.cast %0 : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
    return %1 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: @test_extract_slice_cast
// CHECK-NEXT: [[C0:%.*]] = ndarray.extract_slice %arg0[1] [3] [1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<3xi64>

func.func @test_extract_slice_cast2(%arg0: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<3xi64> {
    %0 = ndarray.extract_slice %arg0[0] [3] [1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<3xi64>
    %1 = ndarray.cast %0 : !ndarray.ndarray<3xi64> to !ndarray.ndarray<?xi64>
    %2 = ndarray.extract_slice %1[0] [3] [1] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
    return %2 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: @test_extract_slice_cast2
// CHECK-NEXT: [[C0:%.*]] = ndarray.extract_slice %arg0[0] [3] [1] : !ndarray.ndarray<5xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: [[C1:%.*]] = ndarray.extract_slice %0[0] [3] [1] : !ndarray.ndarray<3xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: return [[C1:%.*]] : !ndarray.ndarray<3xi64>

func.func @test_extract_immutable_insert_slice(%arg0: !ndarray.ndarray<16x16xi64>) -> !ndarray.ndarray<3x3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.create %c3, %c3 {dtype = 2 : i8} : (index, index) -> !ndarray.ndarray<3x3xi64>
    %1 = ndarray.create %c3, %c3 {dtype = 2 : i8} : (index, index) -> !ndarray.ndarray<3x3xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0, %c0] [%c3, %c3] [%c1, %c1] : !ndarray.ndarray<3x3xi64> into !ndarray.ndarray<16x16xi64>
    %3 = ndarray.immutable_insert_slice %1 into %2[%c0, %c3] [%c3, %c3] [%c1, %c1] : !ndarray.ndarray<3x3xi64> into !ndarray.ndarray<16x16xi64>
    %4 = ndarray.extract_slice %3[0, 0] [3, 3] [1, 1] : !ndarray.ndarray<16x16xi64> to !ndarray.ndarray<3x3xi64>
    return %4 : !ndarray.ndarray<3x3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice
// CHECK: arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.create
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<3x3xi64>

func.func @test_extract_immutable_insert_slice_overwrite(%arg0: !ndarray.ndarray<16x16xi64>) -> !ndarray.ndarray<3x3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.create %c3, %c3 {dtype = 2 : i8} : (index, index) -> !ndarray.ndarray<3x3xi64>
    %1 = ndarray.create %c3, %c3 {dtype = 2 : i8} : (index, index) -> !ndarray.ndarray<3x3xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0, %c0] [%c3, %c3] [%c1, %c1] : !ndarray.ndarray<3x3xi64> into !ndarray.ndarray<16x16xi64>
    %3 = ndarray.immutable_insert_slice %1 into %2[%c0, %c1] [%c3, %c3] [%c1, %c1] : !ndarray.ndarray<3x3xi64> into !ndarray.ndarray<16x16xi64>
    %4 = ndarray.extract_slice %3[0, 0] [3, 3] [1, 1] : !ndarray.ndarray<16x16xi64> to !ndarray.ndarray<3x3xi64>
    return %4 : !ndarray.ndarray<3x3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice_overwrite
// CHECK-NEXT: arith.constant
// CHECK-NEXT: ndarray.create
// CHECK-NEXT: ndarray.create
// CHECK-NEXT: ndarray.immutable_insert_slice
// CHECK-NEXT: ndarray.immutable_insert_slice
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<3x3xi64>

func.func @test_extract_immutable_insert_slice_strided(%arg0: !ndarray.ndarray<16xi64>) -> !ndarray.ndarray<3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.create %c3 {dtype = 2 : i8} : (index) -> !ndarray.ndarray<3xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<16xi64>
    %4 = ndarray.extract_slice %2[0] [3] [%c1] : !ndarray.ndarray<16xi64> to !ndarray.ndarray<3xi64>
    return %4 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice_strided
// CHECK-NEXT: arith.constant
// CHECK-NEXT: ndarray.create
// CHECK-NEXT: ndarray.immutable_insert_slice
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<3xi64>

func.func @test_extract_immutable_insert_slice_strided2(%arg0: !ndarray.ndarray<16xi64>) -> !ndarray.ndarray<3xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.create %c3 {dtype = 2 : i8} : (index) -> !ndarray.ndarray<3xi64>
    %2 = ndarray.immutable_insert_slice %0 into %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<16xi64>
    %4 = ndarray.extract_slice %2[0] [3] [%c3] : !ndarray.ndarray<16xi64> to !ndarray.ndarray<3xi64>
    return %4 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: func.func @test_extract_immutable_insert_slice_strided2
// CHECK-NEXT: arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.create
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<3xi64>

func.func @test_insert_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    return %arg0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[C0:%.*]] = ndarray.cast %arg1 : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: ndarray.insert_slice %0 into %arg0[0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<?xi64>

func.func @test_insert_slice_cast(%arg0: !ndarray.ndarray<5xi64>, %arg1: !ndarray.ndarray<3xi64>) -> !ndarray.ndarray<5xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = ndarray.cast %arg0 : !ndarray.ndarray<5xi64> to !ndarray.ndarray<?xi64>
    %2 = ndarray.cast %arg1 : !ndarray.ndarray<3xi64> to !ndarray.ndarray<?xi64>
    ndarray.insert_slice %2 into %1[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    return %arg0 : !ndarray.ndarray<5xi64>
}
// CHECK-LABEL: @test_insert_slice_cast
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<5xi64>
// CHECK-NEXT: return %arg0 : !ndarray.ndarray<5xi64>

func.func @test_immutable_insert_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.immutable_insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice
// CHECK-NEXT: [[C0:%.*]] = ndarray.cast %arg1 : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice %0 into %arg0 [0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<?xi64>

func.func @test_immutable_insert_slice_cast(%arg0: !ndarray.ndarray<5xi64>, %arg1: !ndarray.ndarray<3xi64>) -> !ndarray.ndarray<5xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = ndarray.cast %arg0 : !ndarray.ndarray<5xi64> to !ndarray.ndarray<?xi64>
    %2 = ndarray.cast %arg1 : !ndarray.ndarray<3xi64> to !ndarray.ndarray<?xi64>
    %3 = ndarray.immutable_insert_slice %2 into %1[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    %4 = ndarray.cast %3 : !ndarray.ndarray<?xi64> to !ndarray.ndarray<5xi64>
    return %4 : !ndarray.ndarray<5xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice_cast
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice %arg1 into %arg0 [0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<5xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<5xi64>

func.func @test_dim(%arg0: !ndarray.ndarray<5xi64>) -> index {
    %c0 = arith.constant 0 : index
    %1 = ndarray.dim %arg0 %c0 : !ndarray.ndarray<5xi64> -> index
    return %1 : index
}
// CHECK-LABEL: @test_dim
// CHECK-NEXT: [[C0:%.*]] = arith.constant 5 : index
// CHECK-NEXT: return [[C0:%.*]] : index

func.func @test_linspace() -> !ndarray.ndarray<?xi64> {
    %c4 = arith.constant 4 : index
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 8.000000e+00 : f64
    %0 = ndarray.linspace %cst_1 %cst_2 %c4 false : (f64, f64, index) -> !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: func.func @test_linspace() -> !ndarray.ndarray<?xi64> {
// CHECK: ndarray.linspace
// CHECK-SAME: !ndarray.ndarray<4xi64>
// CHECK-NEXT: ndarray.cast
// CHECK-SAME: !ndarray.ndarray<?xi64>

func.func @test_permute_dims_cast(%arg0: !ndarray.ndarray<3x4x5xi64>) -> !ndarray.ndarray<?x?x?xi64> {
    %0 = ndarray.permute_dims %arg0 [2, 1, 0] : !ndarray.ndarray<3x4x5xi64> -> !ndarray.ndarray<?x?x?xi64>
    return %0 : !ndarray.ndarray<?x?x?xi64>
}
// CHECK-LABEL: func.func @test_permute_dims_cast
// CHECK: ndarray.permute_dims
// CHECK-SAME: !ndarray.ndarray<3x4x5xi64> -> !ndarray.ndarray<5x4x3xi64>
// CHECK-NEXT: ndarray.cast

func.func @test_permute_dims_identity1(%arg0: !ndarray.ndarray<3x4x5xi64>) -> !ndarray.ndarray<?x?x?xi64> {
    %0 = ndarray.permute_dims %arg0 [0, 1, 2] : !ndarray.ndarray<3x4x5xi64> -> !ndarray.ndarray<?x?x?xi64>
    return %0 : !ndarray.ndarray<?x?x?xi64>
}
// CHECK-LABEL: func.func @test_permute_dims_identity1
// CHECK: ndarray.cast %arg0 : !ndarray.ndarray<3x4x5xi64> to !ndarray.ndarray<?x?x?xi64>
// CHECK-NOT: ndarray.permute_dims

func.func @test_permute_dims_identity2(%arg0: !ndarray.ndarray<3x4x5xi64>) -> !ndarray.ndarray<3x4x5xi64> {
    %0 = ndarray.permute_dims %arg0 [0, 1, 2] : !ndarray.ndarray<3x4x5xi64> -> !ndarray.ndarray<3x4x5xi64>
    return %0 : !ndarray.ndarray<3x4x5xi64>
}
// CHECK-LABEL: func.func @test_permute_dims_identity2
// CHECK-NOT: ndarray.permute_dims
