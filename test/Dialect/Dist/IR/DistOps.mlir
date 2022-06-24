// RUN: imex-opt %s | sed s/true\>/1\>/g | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | sed s/true\>/1\>/g | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s |  sed s/true\>/1\>/g | imex-opt | FileCheck %s

// FIXME sed above, for using 1 instead of true

// -----
// CHECK-LABEL: register_ptensor
func.func @test_register_ptensor(%arg0: tensor<index>) -> i64 {
    %0 = "dist.register_ptensor"(%arg0) : (tensor<index>) -> i64
    return %0 : i64
}

// -----
// CHECK-LABEL: local_shape
func.func @test_local_shape(%arg0: i64) -> tensor<index> {
    %0 = "dist.local_shape"(%arg0) : (i64) -> tensor<index>
    return %0 : tensor<index>
}

// -----
// CHECK-LABEL: local_offsets
func.func @test_local_offsets(%arg0: i64) -> index {
    %0 = "dist.local_offsets"(%arg0) : (i64) -> index
    return %0 : index
}

// -----
// CHECK-LABEL: allreduce
func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
}
