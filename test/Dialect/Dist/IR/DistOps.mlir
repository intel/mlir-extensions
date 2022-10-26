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
func.func @test_nprocs(%arg0: i64) -> i64 {
    %1 = "dist.nprocs"(%arg0) : (i64) -> i64
    return %1 : i64
}
// CHECK-LABEL: func.func @test_nprocs(%arg0: i64) -> i64 {
// CHECK-NEXT "dist.nprocs"(%arg0) : (i64) -> i64

// -----
func.func @test_prank(%arg0: i64) -> i64 {
    %1 = "dist.prank"(%arg0) : (i64) -> i64
    return %1 : i64
}
// CHECK-LABEL: func.func @test_prank(%arg0: i64) -> i64 {
// CHECK-NEXT "dist.prank"(%arg0) : (i64) -> i64

// -----
func.func @test_distinfo(%shape: tensor<1xi64>, %team: i64) -> !dist.info<1> {
    %1 = "dist.distinfo"(%shape, %team) {rank = 1 : i64} : (tensor<1xi64>, i64) -> !dist.info<1>
    return %1 : !dist.info<1>
}
// CHECK-LABEL: func.func @test_distinfo(%arg0: tensor<1xi64>, %arg1: i64) -> !dist.info<1> {
// CHECK-NEXT "dist.distinfo"(%arg0, %arg1) {rank = 1 : i64} : (tensor<1xi64>, i64) -> !dist.info<1>

// -----
func.func @test_init_dist_tensor(%arg0: !ptensor.ptensor<tensor<?xi64>>, %arg1: !dist.info<1>) -> !dist.dtensor<<tensor<?xi64>>> {
    %1 = "dist.init_dist_tensor"(%arg0, %arg1) : (!ptensor.ptensor<tensor<?xi64>>, !dist.info<1>) -> !dist.dtensor<<tensor<?xi64>>>
    return %1 : !dist.dtensor<<tensor<?xi64>>>
}
// CHECK-LABEL: func.func @test_init_dist_tensor(%arg0: !ptensor.ptensor<tensor<?xi64>>, %arg1: !dist.info<1>) -> !dist.dtensor<<tensor<?xi64>>> {
// CHECK-NEXT: dist.init_dist_tensor

// -----
func.func @test_get_ptensor(%arg0: !dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>> {
    %1 = "dist.get_ptensor"(%arg0) : (!dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>>
    return %1 : !ptensor.ptensor<tensor<?xi64>>
}
// CHECK-LABEL: func.func @test_get_ptensor(%arg0: !dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>> {
// CHECK-NEXT: "dist.get_ptensor"(%arg0) : (!dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>>

// -----
func.func @test_get_info(%arg0: !dist.dtensor<<tensor<?xi64>>>) -> !dist.info<1> {
    %1 = "dist.get_info"(%arg0) : (!dist.dtensor<<tensor<?xi64>>>) -> !dist.info<1>
    return %1 : !dist.info<1>
}
// CHECK-LABEL: func.func @test_get_info(%arg0: !dist.dtensor<<tensor<?xi64>>>) -> !dist.info<1> {
// CHECK-NEXT: "dist.get_info"(%arg0) : (!dist.dtensor<<tensor<?xi64>>>) -> !dist.info<1>

// -----
func.func @test_extract_fraom_info(%arg0: !dist.info<1>) -> i64 {
    %1 = "dist.extract_from_info"(%arg0) {what = 0 : i32} : (!dist.info<1>) -> tensor<1xi64>
    %2 = "dist.extract_from_info"(%arg0) {what = 1 : i32} : (!dist.info<1>) -> tensor<1xi64>
    %3 = "dist.extract_from_info"(%arg0) {what = 2 : i32} : (!dist.info<1>) -> i64
    return %3 : i64
}
// CHECK-LABEL: func.func @test_extract_fraom_info(%arg0: !dist.info<1>) -> i64 {
// CHECK-NEXT: %0 = "dist.extract_from_info"(%arg0) {what = 0 : i32} : (!dist.info<1>) -> tensor<1xi64>
// CHECK-NEXT: %1 = "dist.extract_from_info"(%arg0) {what = 1 : i32} : (!dist.info<1>) -> tensor<1xi64>
// CHECK-NEXT: %2 = "dist.extract_from_info"(%arg0) {what = 2 : i32} : (!dist.info<1>) -> i64

// -----
func.func @test_local_offsets(%np : i64, %prank: i64, %shape: tensor<1xi64>) -> tensor<1xi64> {
    %0 = "dist.local_offsets"(%np, %prank, %shape) {rank = 1 : i64} : (i64, i64, tensor<1xi64>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @test_local_offsets(%arg0: i64, %arg1: i64, %arg2: tensor<1xi64>) -> tensor<1xi64> {
// CHECK-NEXT: "dist.local_offsets"(%arg0, %arg1, %arg2) {rank = 1 : i64} : (i64, i64, tensor<1xi64>) -> tensor<1xi64>

func.func @test_local_shape(%np : i64, %prank: i64, %shape: tensor<1xi64>) -> tensor<1xi64> {
    %0 = "dist.local_shape"(%np, %prank, %shape) {rank = 1 : i64} : (i64, i64, tensor<1xi64>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @test_local_shape(%arg0: i64, %arg1: i64, %arg2: tensor<1xi64>) -> tensor<1xi64> {
// CHECK-NEXT: "dist.local_shape"(%arg0, %arg1, %arg2) {rank = 1 : i64} : (i64, i64, tensor<1xi64>) -> tensor<1xi64>

// -----
func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT: "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
