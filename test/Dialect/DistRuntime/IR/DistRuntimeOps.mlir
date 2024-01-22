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
