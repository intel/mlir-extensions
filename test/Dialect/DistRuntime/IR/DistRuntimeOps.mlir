// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// -----
func.func @test_nprocs() -> index {
    %1 = "distruntime.team_size"() <{team=22 : i64}> : () -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_nprocs() -> index {
// CHECK-NEXT: "distruntime.team_size"() <{team = 22 : i64}> : () -> index

// -----
func.func @test_prank() -> index {
    %1 = "distruntime.team_member"() <{team=22 : i64}> : () -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_prank() -> index {
// CHECK-NEXT: "distruntime.team_member"() <{team = 22 : i64}> : () -> index

// -----
func.func @test_allreduce(%arg0: memref<i64>) {
    "distruntime.allreduce"(%arg0) <{op = 4 : i32}> : (memref<i64>) -> ()
    return
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: memref<i64>) {
// CHECK-NEXT: "distruntime.allreduce"(%arg0) <{op = 4 : i32}> : (memref<i64>) -> ()

// -----
func.func @test_copy_reshape(%arg0: !ndarray.ndarray<?x?xi64>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c9 = arith.constant 9 : index
    %h, %a = distruntime.copy_reshape %arg0 g_shape %c3, %c3 l_offs %c1, %c1 to n_g_shape %c9 n_offs %c3 n_shape %c3 {team=22 : i64} : (!ndarray.ndarray<?x?xi64>, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<?xi64>)
    return
}
// CHECK-LABEL: func.func @test_copy_reshape(%arg0: !ndarray.ndarray<?x?xi64>) {
// CHECK: distruntime.copy_reshape %arg0 g_shape %c3, %c3 l_offs %c1, %c1 to n_g_shape %c9 n_offs %c3 n_shape %c3 {team = 22 : i64} : (!ndarray.ndarray<?x?xi64>, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<?xi64>)

// -----
func.func @test_copy_permute(%arg0: !ndarray.ndarray<5x2xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %h, %a = distruntime.copy_permute %arg0 g_shape %c5, %c2 l_offs %c0, %c0 to n_offs %c0, %c0 n_shape %c2, %c5 axes [1, 0] {team=22 : i64} : (!ndarray.ndarray<5x2xi64>, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<2x5xi64>)
    return
}
// CHECK-LABEL: func.func @test_copy_permute(%arg0: !ndarray.ndarray<5x2xi64>) {
// CHECK: distruntime.copy_permute %arg0 g_shape %c5, %c2 l_offs %c0, %c0 to n_offs %c0, %c0 n_shape %c2, %c5 axes [1, 0] {team = 22 : i64} : (!ndarray.ndarray<5x2xi64>, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<2x5xi64>)
