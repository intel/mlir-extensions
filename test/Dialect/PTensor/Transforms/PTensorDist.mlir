// RUN: imex-opt --split-input-file --ptensor-dist %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_arange(%arg0: i64, %arg1: i64, %arg2: i64) -> !ptensor.ptensor<tensor<?xi64>, dist = 1> {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %0 = "ptensor.arange"(%arg0, %arg1, %arg2, %c0, %c1) : (i64, i64, i64, i64, i64) -> !ptensor.ptensor<tensor<?xi64>, dist = 1>
    return %0 : !ptensor.ptensor<tensor<?xi64>, dist = 1>
}
// CHECK-LABEL: @test_arange
// CHECK: [[C0:%.*]] = arith.select
// CHECK-NEXT: [[C1:%.*]] = arith.addi
// CHECK-NEXT: [[C2:%.*]] = arith.addi [[C1]], [[C0]] : i64
// CHECK: [[C3:%.*]] = "dist.register_ptensor"
// CHECK: [[C4:%.*]] = "dist.local_shape"([[C3]])
// CHECK: [[C5:%.*]] = tensor.extract [[C4]]
// CHECK: "dist.local_offsets"([[C3]])
// CHECK: "ptensor.arange"
// CHECK: "ptensor.init_ptensor"
// CHECK: return{{.*}}dist = true

// -----
func.func @test_ewbin(%arg0: !ptensor.ptensor<tensor<?xi64>, dist = 1>) -> !ptensor.ptensor<tensor<?xi64>, dist = 1> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<tensor<?xi64>, dist = 1>, !ptensor.ptensor<tensor<?xi64>, dist = 1>) -> !ptensor.ptensor<tensor<?xi64>, dist = 1>
    return %0 : !ptensor.ptensor<tensor<?xi64>, dist = 1>
}
// CHECK-LABEL: @test_ewbin
// CHECK: "dist.register_ptensor"
// CHECK: "ptensor.ewbin"
// CHECK: "ptensor.init_ptensor"
// CHECK: return{{.*}}dist = true

// -----
func.func @test_reduction(%arg0: !ptensor.ptensor<tensor<?xi64>, dist = 1>) -> !ptensor.ptensor<tensor<i64>, dist = 1> {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<tensor<?xi64>, dist = 1>) -> !ptensor.ptensor<tensor<si64>, dist = 1>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<tensor<si64>, dist = 1> to !ptensor.ptensor<tensor<i64>, dist = 1>
    return %1 : !ptensor.ptensor<tensor<i64>, dist = 1>
}
// CHECK-LABEL: @test_reduction
// CHECK: "dist.register_ptensor"
// CHECK: "ptensor.reduction"
// CHECK: "dist.allreduce"
// CHECK: "ptensor.init_ptensor"
// CHECK: return{{.*}}dist = true
