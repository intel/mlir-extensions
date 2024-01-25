// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// -----
func.func @test_init_dist_array(%pt: !ndarray.ndarray<?xi64>, %loffs: index) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = ? lparts = ?,?,?>> {
    %1 = dist.init_dist_array l_offset %loffs parts %pt, %pt, %pt : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = ? lparts = ?,?,?>>
    return %1 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = ? lparts = ?,?,?>>
}
// CHECK-LABEL: func.func @test_init_dist_array(%arg0: !ndarray.ndarray<?xi64>, %arg1: index) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = ? lparts = ?,?,?>> {
// CHECK-NEXT: dist.init_dist_array

// -----
func.func @test_extract_from_dist(%arg0: !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) {
    %20, %21, %22 = "dist.parts_of"(%arg0) : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) -> (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>)
    %3 = "dist.local_offsets_of"(%arg0) : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) -> index
    return
}
// CHECK-LABEL: @test_extract_from_dist(%arg0: !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) {
// CHECK-NEXT: :3 = "dist.parts_of"(%arg0) : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) -> (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>)
// CHECK-NEXT: "dist.local_offsets_of"(%arg0) : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) -> index

// -----
func.func @test_default_partition(%np : index, %prank: index, %shape: index) -> (index, index) {
    %0, %1 = "dist.default_partition"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, index) -> (index, index)
    return %0, %1 : index, index
}
// CHECK-LABEL: func.func @test_default_partition(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
// CHECK-NEXT: "dist.default_partition"(%arg0, %arg1, %arg2) {rank = 1 : i64} : (index, index, index) -> (index, index)

// -----
func.func @test_local_target_of_slice(%arg0: !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) -> (index, index) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %l_offsets, %l_sizes = dist.local_target_of_slice %arg0[%c0] [%c3] [%c3] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>> to index, index
    return %l_offsets, %l_sizes : index, index
}
// CHECK-LABEL: @test_local_target_of_slice
// CHECK: [[C1:%.*]], [[C2:%.*]] = dist.local_target_of_slice
// CHECK: return [[C1]], [[C2]]

// -----
func.func @test_repartition(%arg0: !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>>) -> (!ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = ? lparts = ?,?,?>>) {
    %0 = dist.repartition %arg0 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = ? lparts = ?,?,?>>
    return %0 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 1 : i64 loffs = ? lparts = ?,?,?>>
}
// CHECK-LABEL: @test_repartition
// CHECK: [[C1:%.*]] = dist.repartition
// CHECK: return [[C1]]
