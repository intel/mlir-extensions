// RUN: imex-opt --split-input-file --convert-ptensor-to-linalg %s -verify-diagnostics -o -| FileCheck %s

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
// CHECK-NEXT: [[V0:%.*]] = memref.subview %arg0[[[C0]]] [[[C1]]] [[[C1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return [[V0]] : memref<?xi64, strided<[?], offset: ?>>

// -----
func.func @test_arange(%arg0: i64, %arg1: i64, %arg2: i64) -> !ptensor.ptensor<1 x index> {
    %0 = "ptensor.arange"(%arg0, %arg1, %arg2) : (i64, i64, i64) -> !ptensor.ptensor<1 x index>
    return %0 : !ptensor.ptensor<1 x index>
}
// CHECK-LABEL: @test_arange
// CHECK: [[C0:%.*]] = arith.select
// CHECK-NEXT: [[C01:%.*]] = arith.subi
// CHECK-NEXT: [[C1:%.*]] = arith.addi
// CHECK-NEXT: [[C2:%.*]] = arith.addi [[C1]], [[C0]] : index
// CHECK: linalg.generic{{.*}}["parallel"]
// CHECK return %{{.}} : tensor<?xindex>

// -----
func.func @test_ewbin(%arg0: !ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<1 x i64> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<1 x i64>, !ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<1 x i64>
    return %0 : !ptensor.ptensor<1 x i64>
}
// CHECK-LABEL: @test_ewbin
// CHECK: [[C0:%.*]] = memref.dim
// CHECK: tensor.empty([[C0]]) : tensor<?xi64>
// CHECK: linalg.generic{{.*}}["parallel"]
// CHECK return %{{.}} : tensor<?xi64>

// -----
func.func @test_reduction(%arg0: !ptensor.ptensor<1 x i64>) -> i64 {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<1 x i64>) -> !ptensor.ptensor<0 x i64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<0 x i64> to i64
    return %1 : i64
}
// CHECK-LABEL: @test_reduction
// CHECK: [[C0:%.*]] = linalg.fill
// CHECK: linalg.generic{{.*}}["reduction"]}{{.*}}outs([[C0]]
// CHECK: return %{{.}} : i64
