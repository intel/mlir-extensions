// RUN: imex-opt %s | sed s/true\>/1\>/g | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | sed s/true\>/1\>/g | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s |  sed s/true\>/1\>/g | imex-opt | FileCheck %s

// FIXME sed above, for using 1 instead of true

// -----
func.func @test_subview(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.subview %arg0[%c0][%c3][%c3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_subview
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.subview %arg0[[[C0]]] [[[C1]]] [[[C1]]] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<?xi64>

// -----
func.func @test_subview_const(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<3xi64> {
    %0 = ndarray.subview %arg0[0][3][3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
    return %0 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: @test_subview_const
// CHECK-NEXT: [[V0:%.*]] = ndarray.subview %arg0[0] [3] [3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<3xi64>

// -----
func.func @test_extract_slice(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.extract_slice %arg0[%c0][%c3][%c3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_extract_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice %arg0[[[C0]]] [[[C1]]] [[[C1]]] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<?xi64>

// -----
func.func @test_extract_slice_const(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<3xi64> {
    %0 = ndarray.extract_slice %arg0[0][3][3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
    return %0 : !ndarray.ndarray<3xi64>
}
// CHECK-LABEL: @test_extract_slice_const
// CHECK-NEXT: [[V0:%.*]] = ndarray.extract_slice %arg0[0] [3] [3] : !ndarray.ndarray<?xi64> to !ndarray.ndarray<3xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<3xi64>

// -----
func.func @test_insert_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    return %arg0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[[[C0]]] [[[C1]]] [[[C1]]] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>

// -----
func.func @test_insert_slice_const(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<3xi64>) -> !ndarray.ndarray<?xi64> {
    ndarray.insert_slice %arg1 into %arg0[0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<?xi64>
    return %arg0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_insert_slice_const
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<?xi64>

// -----
func.func @test_insert_slice_scalar(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<i64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%c0] [%c1] [%c3] : !ndarray.ndarray<i64> into !ndarray.ndarray<?xi64>
    return %arg0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_insert_slice_scalar
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[C3:%.*]] = arith.constant
// CHECK-NEXT: ndarray.insert_slice %arg1 into %arg0[[[C0]]] [[[C1]]] [[[C3]]] : !ndarray.ndarray<i64> into !ndarray.ndarray<?xi64>

// -----
func.func @test_immutable_insert_slice(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.immutable_insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
    return %arg0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice %arg1 into %arg0 [[[C0]]] [[[C1]]] [[[C1]]] : !ndarray.ndarray<?xi64> into !ndarray.ndarray<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<?xi64>

// -----
func.func @test_immutable_insert_slice_const(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<3xi64>) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.immutable_insert_slice %arg1 into %arg0[0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_immutable_insert_slice_const
// CHECK-NEXT: [[V0:%.*]] = ndarray.immutable_insert_slice %arg1 into %arg0 [0] [3] [3] : !ndarray.ndarray<3xi64> into !ndarray.ndarray<?xi64>
// CHECK-NEXT: return [[V0:%.*]] : !ndarray.ndarray<?xi64>

// -----
func.func @test_linspace(%arg0: si64, %arg1: si64, %arg2: si64) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.linspace %arg0 %arg1 %arg2 false : (si64, si64, si64) -> !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_linspace
// CHECK-NEXT: ndarray.linspace %arg0 %arg1 %arg2 false : (si64, si64, si64) -> !ndarray.ndarray<?xi64>

func.func @test_create(%arg0: index, %arg1: index, %arg2: index, %arg3: i64) -> !ndarray.ndarray<?x?x?xf64> {
    %0 = ndarray.create %arg0, %arg1, %arg2 {dtype = 0 : i8} : (index, index, index) -> !ndarray.ndarray<?x?x?xf64>
    return %0 : !ndarray.ndarray<?x?x?xf64>
}
// CHECK-LABEL: @test_create
// CHECK: %arg0, %arg1, %arg2 {dtype = 0 : i8} : (index, index, index) -> !ndarray.ndarray<?x?x?xf64>

func.func @test_create2(%arg0: index, %arg1: index, %arg2: index, %arg3: i64) -> !ndarray.ndarray<?x?x?xi64> {
    %0 = ndarray.create %arg0, %arg1, %arg2 value %arg3 {environment = 3 : i64, team = 3 : i64, dtype = 2 : i8} : (index, index, index, i64) -> !ndarray.ndarray<?x?x?xi64>
    return %0 : !ndarray.ndarray<?x?x?xi64>
}
// CHECK-LABEL: @test_create2
// CHECK: ndarray.create %arg0, %arg1, %arg2 value %arg3 {dtype = 2 : i8, environment = 3 : i64, team = 3 : i64} : (index, index, index, i64) -> !ndarray.ndarray<?x?x?xi64>

// -----
func.func @test_reshape(%arg0: index) -> !ndarray.ndarray<?x?xi64> {
    %0 = ndarray.create %arg0 {dtype = 2 : i8} : (index) -> !ndarray.ndarray<?xi64>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = "ndarray.reshape"(%0, %c0, %c3) : (!ndarray.ndarray<?xi64>, index, index) -> !ndarray.ndarray<?x?xi64>
    return %1 : !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: @test_reshape
// CHECK: ndarray.create
// CHECK: ndarray.reshape
// CHECK-SAME: -> !ndarray.ndarray<?x?xi64>

// -----
func.func @test_ewbin(%arg0: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.ewbin %arg0, %arg0 {op = 0 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_ewbin
// CHECK-NEXT: ndarray.ewbin %arg0, %arg0 {op = 0 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>

// -----
func.func @test_reduction(%arg0: !ndarray.ndarray<?xi64>) -> si64 {
    %0 = ndarray.reduction %arg0 {op = 4 : i32} : !ndarray.ndarray<?xi64> -> !ndarray.ndarray<si64>
    %1 = builtin.unrealized_conversion_cast %0 : !ndarray.ndarray<si64> to si64
    return %1 : si64
}
// CHECK-LABEL: @test_reduction
// CHECK-NEXT: ndarray.reduction %arg0 {op = 4 : i32} : !ndarray.ndarray<?xi64> -> !ndarray.ndarray<si64>

// -----
func.func @test_dim(%arg0: !ndarray.ndarray<?xi64>) -> index {
    %c0 = arith.constant 0 : index
    %1 = ndarray.dim %arg0 %c0 : !ndarray.ndarray<?xi64> -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_dim
// CHECK: [[V0:%.*]] = ndarray.dim
// CHECK-NEXT: return [[V0]] : index

// -----
func.func @test_cast(%arg0: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.cast %arg0 : !ndarray.ndarray<5xi64> to !ndarray.ndarray<?xi64>
    return %0 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: func.func @test_cast
// CHECK: [[V0:%.*]] = ndarray.cast
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<?xi64>

// -----
func.func @test_copy(%arg0: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<5xi64> {
    %0 = ndarray.copy %arg0 : !ndarray.ndarray<5xi64> -> !ndarray.ndarray<5xi64>
    return %0 : !ndarray.ndarray<5xi64>
}
// CHECK-LABEL: func.func @test_copy
// CHECK: [[V0:%.*]] = ndarray.copy
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<5xi64>

// -----
func.func @test_castelem(%arg0: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<5xi32> {
    %0 = ndarray.cast_elemtype %arg0 : !ndarray.ndarray<5xi64> to !ndarray.ndarray<5xi32>
    return %0 : !ndarray.ndarray<5xi32>
}
// CHECK-LABEL: func.func @test_castelem
// CHECK: [[V0:%.*]] = ndarray.cast_elemtype
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<5xi32>

// -----
func.func @test_from_memref(%arg0: memref<?xi32, strided<[?], offset: ?>>) -> !ndarray.ndarray<?xi32> {
    %0 = ndarray.from_memref %arg0 : memref<?xi32, strided<[?], offset: ?>> -> !ndarray.ndarray<?xi32>
    return %0 : !ndarray.ndarray<?xi32>
}
// CHECK-LABEL: func.func @test_from_memref
// CHECK: [[V0:%.*]] = ndarray.from_memref
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<?xi32>

// -----
func.func @test_permute_dims(%arg0: !ndarray.ndarray<?x?x?xi64>) -> !ndarray.ndarray<?x?x?xi64> {
    %0 = ndarray.permute_dims %arg0 [0, 1, 2] : !ndarray.ndarray<?x?x?xi64> -> !ndarray.ndarray<?x?x?xi64>
    return %0 : !ndarray.ndarray<?x?x?xi64>
}
// CHECK-LABEL: func.func @test_permute_dims
// CHECK: [[V0:%.*]] = ndarray.permute_dims
// CHECK-NEXT: return [[V0]] : !ndarray.ndarray<?x?x?xi64>
