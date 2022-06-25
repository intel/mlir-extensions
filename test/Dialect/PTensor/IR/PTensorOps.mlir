// RUN: imex-opt %s | sed s/true\>/1\>/g | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | sed s/true\>/1\>/g | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s |  sed s/true\>/1\>/g | imex-opt | FileCheck %s

// FIXME sed above, for using 1 instead of true

// -----
// CHECK-LABEL: arange
func.func @test_arange(%arg0: si64, %arg1: si64, %arg2: si64) -> !ptensor.ptensor<tensor<?xi64>, 1> {
    %0 = "ptensor.arange"(%arg0, %arg1, %arg2) {dist = true} : (si64, si64, si64) -> !ptensor.ptensor<tensor<?xi64>, 1>
    return %0 : !ptensor.ptensor<tensor<?xi64>, 1>
}

// -----
// CHECK-LABEL: ewbin
func.func @test_ewbin(%arg0: !ptensor.ptensor<tensor<?xi64>, 1>) -> !ptensor.ptensor<tensor<?xi64>, 1> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<tensor<?xi64>, 1>, !ptensor.ptensor<tensor<?xi64>, 1>) -> !ptensor.ptensor<tensor<?xi64>, 1>
    return %0 : !ptensor.ptensor<tensor<?xi64>, 1>
}

// -----
// CHECK-LABEL: reduction
func.func @test_reduction(%arg0: !ptensor.ptensor<tensor<?xi64>, 1>) -> si64 {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<tensor<?xi64>, 1>) -> !ptensor.ptensor<tensor<si64>, 1>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<tensor<si64>, 1> to si64
    return %1 : si64
}
