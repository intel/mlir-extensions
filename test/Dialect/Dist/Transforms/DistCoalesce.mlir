// RUN: imex-opt --split-input-file --dist-coalesce %s -verify-diagnostics -o -| FileCheck %s

module {
  func.func @test_coalesce1() -> (!ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index
    %c30 = arith.constant 30 : index
    %0 = ndarray.linspace %c0 %c10 %c10 false : (index, index, index) -> !ndarray.ndarray<?xi64>
    %1 = dist.init_dist_array l_offset %c5 parts %0, %0, %0 : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %2 = dist.init_dist_array l_offset %c5 parts %0, %0, %0 : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %3 = dist.repartition %1 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %4 = dist.repartition %2 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %5 = "dist.ewbin"(%3, %4) {op = 0 : i32} : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %6 = dist.repartition %5 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %7 = dist.repartition %1 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %8 = "dist.ewbin"(%6, %7) {op = 0 : i32} : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    return %8 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  }
}
// CHECK-LABEL: func.func @test_coalesce1()
// CHECK: dist.repartition
// CHECK-NEXT: dist.repartition
// CHECK-NEXT: dist.ewbin
// CHECK-NEXT: dist.repartition
// CHECK-NEXT: dist.ewbin
// CHECK-NEXT: return

// -----
module {
  func.func @test_coalesce2() -> (!ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index
    %c30 = arith.constant 30 : index
    %0 = ndarray.linspace %c0 %c10 %c10 false : (index, index, index) -> !ndarray.ndarray<?xi64>
    %1 = dist.init_dist_array l_offset %c5 parts %0, %0, %0 : index, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %v1 = dist.subview %1[%c0] [%c5] [%c2] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %v2 = dist.subview %1[%c5] [%c5] [%c1] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %3 = dist.repartition %v1 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %4 = dist.repartition %v2 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %5 = "dist.ewbin"(%3, %4) {op = 0 : i32} : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>, !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %v3 = dist.subview %1[%c1] [%c5] [%c1] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %6 = dist.repartition %5 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %7 = dist.repartition %v3 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %8 = "dist.ewbin"(%6, %7) {op = 0 : i32} : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>, !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    %t_offsets, %t_sizes = dist.local_target_of_slice %1[%c1] [%c5] [%c2] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> to index, index
    %10 = dist.repartition %8   loffs %t_offsets lsizes %t_sizes : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>, index, index to !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    ndarray.insert_slice %10 into %1[%c1] [%c5] [%c2] : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>> into !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
    return %1 : !ndarray.ndarray<?xi64, #dist.dist_env<team = 22 loffs = 0 lparts = ?,?,?>>
  }
}
// CHECK-LABEL: func.func @test_coalesce2()
// CHECK: dist.init_dist_array
// CHECK: distruntime.team_size
// CHECK: distruntime.team_member
// CHECK: dist.local_target_of_slice
// CHECK-NEXT: dist.local_bounding_box
// CHECK-NEXT: dist.local_bounding_box
// CHECK-NEXT: dist.local_bounding_box
// CHECK-NEXT: dist.repartition
// CHECK-NEXT: dist.subview
// CHECK-NEXT: dist.subview
// CHECK-NEXT: dist.subview
// CHECK: dist.ewbin
// CHECK: dist.ewbin
// CHECK: ndarray.insert_slice
