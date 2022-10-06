// RUN: imex-opt --split-input-file --convert-ptensor-to-linalg %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_arange(%arg0: si64, %arg1: si64, %arg2: si64) -> !ptensor.ptensor<tensor<?xi64>> {
    %0 = "ptensor.arange"(%arg0, %arg1, %arg2) : (si64, si64, si64) -> !ptensor.ptensor<tensor<?xi64>>
    return %0 : !ptensor.ptensor<tensor<?xi64>>
}
// CHECK-LABEL: @test_arange
// CHECK: [[C0:%.*]] = arith.select
// CHECK-NEXT: [[C1:%.*]] = arith.addi
// CHECK-NEXT: [[C2:%.*]] = arith.addi [[C1]], [[C0]] : i64
// CHECK: linalg.generic{{.*}}["parallel"]
// CHECK return %{{.}} : tensor<?xi64>

// -----
func.func @test_ewbin(%arg0: !ptensor.ptensor<tensor<?xi64>>) -> !ptensor.ptensor<tensor<?xi64>> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<tensor<?xi64>>, !ptensor.ptensor<tensor<?xi64>>) -> !ptensor.ptensor<tensor<?xi64>>
    return %0 : !ptensor.ptensor<tensor<?xi64>>
}
// CHECK-LABEL: @test_ewbin
// CHECK: [[C0:%.*]] = tensor.dim
// CHECK: linalg.init_tensor [[[C0]]] : tensor<?xi64>
// CHECK: linalg.generic{{.*}}["parallel"]
// CHECK return %{{.}} : tensor<?xi64>

// -----
func.func @test_reduction(%arg0: !ptensor.ptensor<tensor<?xi64>>) -> si64 {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<tensor<?xi64>>) -> !ptensor.ptensor<tensor<si64>>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<tensor<si64>> to si64
    return %1 : si64
}
// CHECK-LABEL: @test_reduction
// CHECK: [[C0:%.*]] = linalg.fill
// CHECK: linalg.generic{{.*}}["reduction"]}{{.*}}outs([[C0]]
// CHECK: return %{{.}} : si64
