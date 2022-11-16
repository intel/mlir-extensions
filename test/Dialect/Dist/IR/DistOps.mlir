// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// -----

module {
    "dist.runtime_prototypes"() : () -> ()
}
// CHECK-LABEL: "dist.runtime_prototypes"() : () -> ()

// -----
func.func @test_nprocs(%arg0: index) -> index {
    %1 = "dist.nprocs"(%arg0) : (index) -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_nprocs(%arg0: index) -> index {
// CHECK-NEXT: "dist.nprocs"(%arg0) : (index) -> index

// -----
func.func @test_prank(%arg0: index) -> index {
    %1 = "dist.prank"(%arg0) : (index) -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_prank(%arg0: index) -> index {
// CHECK-NEXT: "dist.prank"(%arg0) : (index) -> index

// -----
func.func @test_init_dist_tensor(%gshape: tensor<1xindex>, %pt: !ptensor.ptensor<tensor<?xi64>>, %loffs: tensor<1xindex>, %team: i64) -> !dist.dtensor<<tensor<?xi64>>> {
    %1 = "dist.init_dist_tensor"(%gshape, %pt, %loffs, %team) : (tensor<1xindex>, !ptensor.ptensor<tensor<?xi64>>, tensor<1xindex>, i64) -> !dist.dtensor<<tensor<?xi64>>>
    return %1 : !dist.dtensor<<tensor<?xi64>>>
}
// CHECK-LABEL: func.func @test_init_dist_tensor(%arg0: tensor<1xindex>, %arg1: !ptensor.ptensor<tensor<?xi64>>, %arg2: tensor<1xindex>, %arg3: i64) -> !dist.dtensor<<tensor<?xi64>>> {
// CHECK-NEXT: dist.init_dist_tensor

// -----
func.func @test_extract_from_dist(%arg0: !dist.dtensor<<tensor<?xi64>>>) -> index {
    %1 = "dist.extract_from_dist"(%arg0) {what = 0 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> tensor<1xindex>
    %2 = "dist.extract_from_dist"(%arg0) {what = 1 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>>
    %3 = "dist.extract_from_dist"(%arg0) {what = 2 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> tensor<1xindex>
    %4 = "dist.extract_from_dist"(%arg0) {what = 3 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> index
    return %4 : index
}
// CHECK-LABEL: func.func @test_extract_from_dist(%arg0: !dist.dtensor<<tensor<?xi64>>>) -> index {
// CHECK-NEXT: "dist.extract_from_dist"(%arg0) {what = 0 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> tensor<1xindex>
// CHECK-NEXT: "dist.extract_from_dist"(%arg0) {what = 1 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> !ptensor.ptensor<tensor<?xi64>>
// CHECK-NEXT: "dist.extract_from_dist"(%arg0) {what = 2 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> tensor<1xindex>
// CHECK-NEXT: "dist.extract_from_dist"(%arg0) {what = 3 : i32} : (!dist.dtensor<<tensor<?xi64>>>) -> index

// -----
func.func @test_local_offsets(%np : index, %prank: index, %shape: tensor<1xindex>) -> tensor<1xindex> {
    %0 = "dist.local_offsets"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, tensor<1xindex>) -> tensor<1xindex>
    return %0 : tensor<1xindex>
}
// CHECK-LABEL: func.func @test_local_offsets(%arg0: index, %arg1: index, %arg2: tensor<1xindex>) -> tensor<1xindex> {
// CHECK-NEXT: "dist.local_offsets"(%arg0, %arg1, %arg2) {rank = 1 : i64} : (index, index, tensor<1xindex>) -> tensor<1xindex>

func.func @test_local_shape(%np : index, %prank: index, %shape: tensor<1xindex>) -> tensor<1xindex> {
    %0 = "dist.local_shape"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, tensor<1xindex>) -> tensor<1xindex>
    return %0 : tensor<1xindex>
}
// CHECK-LABEL: func.func @test_local_shape(%arg0: index, %arg1: index, %arg2: tensor<1xindex>) -> tensor<1xindex> {
// CHECK-NEXT: "dist.local_shape"(%arg0, %arg1, %arg2) {rank = 1 : i64} : (index, index, tensor<1xindex>) -> tensor<1xindex>

// -----
func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT: "dist.allreduce"(%arg0) {op = 4 : i32} : (tensor<i64>) -> tensor<i64>
