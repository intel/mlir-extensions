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
func.func @test_init_dist_tensor(%gshape: index, %pt: !ptensor.ptensor<1 x i64>, %loffs: index, %team: i64) -> !dist.dtensor<<1 x i64>> {
    %1 = "dist.init_dist_tensor"(%gshape, %pt, %loffs, %team) : (index, !ptensor.ptensor<1 x i64>, index, i64) -> !dist.dtensor<<1 x i64>>
    return %1 : !dist.dtensor<<1 x i64>>
}
// CHECK-LABEL: func.func @test_init_dist_tensor(%arg0: index, %arg1: !ptensor.ptensor<1 x i64>, %arg2: index, %arg3: i64) -> !dist.dtensor<<1 x i64>> {
// CHECK-NEXT: dist.init_dist_tensor

// -----
func.func @test_extract_from_dist(%arg0: !dist.dtensor<<1 x i64>>) -> index {
    %1 = "dist.global_shape_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> index
    %2 = "dist.local_tensor_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> !ptensor.ptensor<1 x i64>
    %3 = "dist.local_offsets_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> index
    %4 = "dist.team_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> index
    return %4 : index
}
// CHECK-LABEL: func.func @test_extract_from_dist(%arg0: !dist.dtensor<<1 x i64>>) -> index {
// CHECK-NEXT: "dist.global_shape_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> index
// CHECK-NEXT: "dist.local_tensor_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> !ptensor.ptensor<1 x i64>
// CHECK-NEXT: "dist.local_offsets_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> index
// CHECK-NEXT: "dist.team_of"(%arg0) : (!dist.dtensor<<1 x i64>>) -> index

// -----
func.func @test_local_partition(%np : index, %prank: index, %shape: index) -> (index, index) {
    %0, %1 = "dist.local_partition"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, index) -> (index, index)
    return %0, %1 : index, index
}
// CHECK-LABEL: func.func @test_local_partition(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
// CHECK-NEXT: "dist.local_partition"(%arg0, %arg1, %arg2) {rank = 1 : i64} : (index, index, index) -> (index, index)

// -----
func.func @test_allreduce(%arg0: memref<i64>) -> memref<i64> {
    %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (memref<i64>) -> memref<i64>
    return %0 : memref<i64>
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: memref<i64>) -> memref<i64> {
// CHECK-NEXT: "dist.allreduce"(%arg0) {op = 4 : i32} : (memref<i64>) -> memref<i64>

// -----
func.func @test_local_of_slice(%arg0: !dist.dtensor<<1 x i64>>) -> (index, index, index) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %l_offsets, %l_sizes, %g_offsets = dist.local_of_slice %arg0[%c0] [%c3] [%c3] : !dist.dtensor<<1 x i64>> to (index, index, index)
    return %l_offsets, %l_sizes, %g_offsets : index, index, index
}
// CHECK-LABEL: @test_local_of_slice
// CHECK: [[C1:%.*]], [[C2:%.*]], [[C3:%.*]] = dist.local_of_slice
// CHECK: return [[C1]], [[C2]], [[C3]]
