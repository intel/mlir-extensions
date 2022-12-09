// RUN: imex-opt %s | sed s/true\>/1\>/g | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | sed s/true\>/1\>/g | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s |  sed s/true\>/1\>/g | imex-opt | FileCheck %s

// FIXME sed above, for using 1 instead of true

// -----
func.func @test_extract_slice(%arg0: !ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<1 x i64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ptensor.extract_slice %arg0[%c0][%c3][%c3] : !ptensor.ptensor<1 x i64> to !ptensor.ptensor<1 x i64>
    return %0 : !ptensor.ptensor<1 x i64>
}
// CHECK-LABEL: @test_extract_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: ptensor.extract_slice %arg0[[[C0]]] [[[C1]]] [[[C1]]] : !ptensor.ptensor<1 x i64> to !ptensor.ptensor<1 x i64>

// -----
func.func @test_insert_slice(%arg0: !ptensor.ptensor<1 x i64>, %arg1: !ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<1 x i64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    ptensor.insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : !ptensor.ptensor<1 x i64> into !ptensor.ptensor<1 x i64>
    return %arg0 : !ptensor.ptensor<1 x i64>
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: ptensor.insert_slice %arg1 into %arg0[[[C0]]] [[[C1]]] [[[C1]]] : !ptensor.ptensor<1 x i64> into !ptensor.ptensor<1 x i64>

// -----
func.func @test_arange(%arg0: si64, %arg1: si64, %arg2: si64) -> !ptensor.ptensor<1 x i64> {
    %0 = "ptensor.arange"(%arg0, %arg1, %arg2) : (si64, si64, si64) -> !ptensor.ptensor<1 x i64>
    return %0 : !ptensor.ptensor<1 x i64>
}
// CHECK-LABEL: @test_arange
// CHECK-NEXT: "ptensor.arange"(%arg0, %arg1, %arg2) : (si64, si64, si64) -> !ptensor.ptensor<1 x i64>

func.func @test_create(%arg0: index, %arg1: index, %arg2: index, %arg3: i64) -> !ptensor.ptensor<3 x f64> {
    %0 = ptensor.create %arg0, %arg1, %arg2 {dtype = 0 : i8} : (index, index, index) -> !ptensor.ptensor<3 x f64>
    return %0 : !ptensor.ptensor<3 x f64>
}
// CHECK-LABEL: @test_create
// CHECK: %arg0, %arg1, %arg2 {dtype = 0 : i8} : (index, index, index) -> !ptensor.ptensor<3 x f64>

func.func @test_create2(%arg0: index, %arg1: index, %arg2: index, %arg3: i64) -> !ptensor.ptensor<3 x i64> {
    %0 = ptensor.create %arg0, %arg1, %arg2 value %arg3 device %arg3 team %arg3 {dtype = 2 : i8} : (index, index, index, i64, i64, i64) -> !ptensor.ptensor<3 x i64>
    return %0 : !ptensor.ptensor<3 x i64>
}
// CHECK-LABEL: @test_create2
// CHECK: ptensor.create %arg0, %arg1, %arg2 value %arg3 device %arg3 team %arg3 {dtype = 2 : i8} : (index, index, index, i64, i64, i64) -> !ptensor.ptensor<3 x i64>

// -----
func.func @test_ewbin(%arg0: !ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<1 x i64> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<1 x i64>, !ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<1 x i64>
    return %0 : !ptensor.ptensor<1 x i64>
}
// CHECK-LABEL: @test_ewbin
// CHECK-NEXT: "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<1 x i64>, !ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<1 x i64>

// -----
func.func @test_reduction(%arg0: !ptensor.ptensor<1 x i64>) -> si64 {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<0 x si64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<0 x si64> to si64
    return %1 : si64
}
// CHECK-LABEL: @test_reduction
// CHECK-NEXT: "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<0 x si64>
