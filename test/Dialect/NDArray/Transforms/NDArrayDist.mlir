// RUN: imex-opt --split-input-file --ndarray-dist %s -verify-diagnostics -o -| FileCheck %s

func.func @test_linspace(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c33 = arith.constant 33 : i64
    %c22 = arith.constant 22 : index
    %v = arith.constant 55 : i64
    %s = arith.index_cast %arg0 : i64 to index
    %0 = ndarray.linspace %arg0 %arg1 %c33 false {team = 1} : (i64, i64, i64) -> !ndarray.ndarray<33xi64, #dist.dist_env<team = 1>>
    %1 = ndarray.create %c22 value %v {team = 1, dtype = 2 : i8} : (index, i64) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 1>>
    %10 = ndarray.subview %0[%c0][22][%c3] : !ndarray.ndarray<33xi64, #dist.dist_env<team = 1>> to !ndarray.ndarray<?xi64, #dist.dist_env<team = 1>>
    %20 = ndarray.ewbin %10, %1 {op = 0 : i32} : (!ndarray.ndarray<?xi64, #dist.dist_env<team = 1>>, !ndarray.ndarray<?xi64, #dist.dist_env<team = 1>>) -> !ndarray.ndarray<?xi64, #dist.dist_env<team = 1>>
    %21 = ndarray.reduction %20 {op = 4 : i32} : !ndarray.ndarray<?xi64, #dist.dist_env<team = 1>> -> !ndarray.ndarray<i64, #dist.dist_env<team = 1>>
    %30 = builtin.unrealized_conversion_cast %21 : !ndarray.ndarray<i64, #dist.dist_env<team = 1>> to i64
    return %30 : i64
}
// CHECK-LABEL: func.func @test_linspace
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: ndarray.linspace
// CHECK: ndarray.create
// CHECK: dist.subview
// CHECK: dist.repartition
// CHECK: dist.repartition
// CHECK: "dist.ewbin"
// CHECK: ndarray.reduction

// -----
func.func @test_dim(%arg0: !ndarray.ndarray<10x20xi64, #dist.dist_env<team = 22 : i64 loffs = 0,0 lparts = ?x?,?x?,?x?>>) -> index {
    %c0 = arith.constant 0 : index
    %1 = ndarray.dim %arg0 %c0 : !ndarray.ndarray<10x20xi64, #dist.dist_env<team = 22 : i64 loffs = 0,0 lparts = ?x?,?x?,?x?>> -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_dim
// CHECK-NEXT: [[V:%.*]] = arith.constant 10 : index
// CHECK-NEXT: return [[V]] : index

// -----
func.func @test_ewuny(%arg0: !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = 10,20,5>>) -> !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>> {
    %0 ="ndarray.ewuny"(%arg0) {op = 0 : i32} : (!ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = 10,20,5>>) -> !ndarray.ndarray<?xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>
    %1 = builtin.unrealized_conversion_cast %0 : !ndarray.ndarray<?xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>
    return %1 : !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>
}
// CHECK-LABEL: func.func @test_ewuny
// CHECK: "dist.ewuny"

// -----
func.func @test_ewbin(%arg0: !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>, %arg1: !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>) -> !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>, !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>) -> !ndarray.ndarray<?xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>
    %1 = builtin.unrealized_conversion_cast %0 : !ndarray.ndarray<?xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>> to !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>
    return %1 : !ndarray.ndarray<11xf64, #dist.dist_env<team = 22 : i64 loffs = 0 lparts = ?,?,?>>
}
// CHECK-LABEL: func.func @test_ewbin
// CHECK: [[V1:%.*]] = dist.repartition
// CHECK: [[V2:%.*]] = dist.repartition
// CHECK: "dist.ewbin"([[V1]], [[V2]])
