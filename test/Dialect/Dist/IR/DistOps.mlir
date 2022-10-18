// RUN: imex-opt %s | sed s/true\>/1\>/g | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | sed s/true\>/1\>/g | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s |  sed s/true\>/1\>/g | imex-opt | FileCheck %s

// FIXME sed above, for using 1 instead of true

// -----

module {
    "dist.runtime_prototypes"() : () -> ()
}
// CHECK-LABEL: "dist.runtime_prototypes"() : () -> ()

// -----
func.func @test_register_ptensor() -> i64 {
    %0 = shape.const_shape [-1] : tensor<1xindex>
    %1 = "dist.register_ptensor"(%0) : (tensor<1xindex>) -> i64
    return %1 : i64
}
// CHECK-LABEL: func.func @test_register_ptensor() -> i64 {
// CHECK-NEXT: shape.const_shape
// CHECK-NEXT: dist.register_ptensor

// -----

func.func @test_local_shape(%arg0: i64) -> tensor<?xi64> {
    %0 = "dist.local_shape"(%arg0) {rank = 1 : i64} : (i64) -> tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: func.func @test_local_shape(%arg0: i64) -> tensor<?xi64> {
// CHECK-NEXT: "dist.local_shape"(%arg0) {rank = 1 : i64} : (i64) -> tensor<?xi64>

// -----
func.func @test_local_offsets(%arg0: i64) -> tensor<?xi64> {
    %0 = "dist.local_offsets"(%arg0) {rank = 1 : i64} : (i64) -> tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: func.func @test_local_offsets(%arg0: i64) -> tensor<?xi64> {
// CHECK-NEXT: "dist.local_offsets"(%arg0) {rank = 1 : i64} : (i64) -> tensor<?xi64>

// -----
func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT: "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
