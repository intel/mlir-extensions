// RUN: imex-opt --split-input-file --pass-pipeline="builtin.module(convert-ndarray-to-linalg,func.func(tosa-to-linalg,canonicalize,linalg-fuse-elementwise-ops))" %s -verify-diagnostics -o -| FileCheck %s

func.func @test_binop_fusion_arith(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    %1 = ndarray.ewbin %0, %arg0 {op = 21 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    %2 = ndarray.ewbin %arg0, %1 {op = 0 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    %3 = ndarray.ewbin %arg1, %2 {op = 21 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    return %3 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_binop_fusion_arith
// CHECK-NEXT: arith.constant
// CHECK: memref.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK: return %{{[0-9]+}} : memref<?xi64, strided<[?], offset: ?>>

// NOTE tosa ewbinop with dynamic shapes are broadcast-aware and thus do not fuse
func.func @test_binop_fusion_tosa(%arg0: !ndarray.ndarray<5xi64>, %arg1: !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<5xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 4 : i32} : (!ndarray.ndarray<5xi64>, !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<5xi64>
    %1 = ndarray.ewbin %0, %arg0 {op = 2 : i32} : (!ndarray.ndarray<5xi64>, !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<5xi64>
    %2 = ndarray.ewbin %arg0, %1 {op = 4 : i32} : (!ndarray.ndarray<5xi64>, !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<5xi64>
    %3 = ndarray.ewbin %arg1, %2 {op = 2 : i32} : (!ndarray.ndarray<5xi64>, !ndarray.ndarray<5xi64>) -> !ndarray.ndarray<5xi64>
    return %3 : !ndarray.ndarray<5xi64>
}
// CHECK-LABEL: @test_binop_fusion_tosa
// CHECK: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.ori
// CHECK-NEXT: arith.andi
// CHECK-NEXT: arith.ori
// CHECK-NEXT: arith.andi
// CHECK: return %{{[0-9]+}} : memref<5xi64, strided<[?], offset: ?>>

func.func @test_binop_fusion_bcast(%arg0: !ndarray.ndarray<?x?xi64>, %arg1: !ndarray.ndarray<i64>) -> !ndarray.ndarray<?x?xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<?x?xi64>, !ndarray.ndarray<i64>) -> !ndarray.ndarray<?x?xi64>
    %1 = ndarray.ewbin %0, %arg0 {op = 21 : i32} : (!ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>) -> !ndarray.ndarray<?x?xi64>
    %2 = ndarray.ewbin %arg0, %1 {op = 0 : i32} : (!ndarray.ndarray<?x?xi64>, !ndarray.ndarray<?x?xi64>) -> !ndarray.ndarray<?x?xi64>
    %3 = ndarray.ewbin %arg1, %2 {op = 21 : i32} : (!ndarray.ndarray<i64>, !ndarray.ndarray<?x?xi64>) -> !ndarray.ndarray<?x?xi64>
    return %3 : !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: @test_binop_fusion_bcast
// CHECK-NEXT: arith.constant
// CHECK-NEXT: arith.constant
// CHECK: memref.dim
// CHECK-NEXT: memref.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK: return %{{[0-9]+}} : memref<?x?xi64, strided<[?, ?], offset: ?>

func.func @test_binop_fusion_bcast2(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<1xi64>) -> !ndarray.ndarray<?xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<1xi64>) -> !ndarray.ndarray<?xi64>
    %1 = ndarray.ewbin %0, %arg0 {op = 21 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    %2 = ndarray.ewbin %arg0, %1 {op = 0 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    %3 = ndarray.ewbin %arg1, %2 {op = 21 : i32} : (!ndarray.ndarray<1xi64>, !ndarray.ndarray<?xi64>) -> !ndarray.ndarray<?xi64>
    return %3 : !ndarray.ndarray<?xi64>
}
// CHECK-LABEL: @test_binop_fusion_bcast2
// CHECK-NEXT: arith.constant
// CHECK: memref.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK: return %{{[0-9]+}} : memref<?xi64, strided<[?], offset: ?>>

func.func @test_binop_fusion_bcast3(%arg0: !ndarray.ndarray<?xi64>, %arg1: !ndarray.ndarray<1x?xi64>) -> !ndarray.ndarray<?x?xi64> {
    %0 = ndarray.ewbin %arg0, %arg1 {op = 0 : i32} : (!ndarray.ndarray<?xi64>, !ndarray.ndarray<1x?xi64>) -> !ndarray.ndarray<?x?xi64>
    %1 = ndarray.ewbin %0, %arg1 {op = 21 : i32} : (!ndarray.ndarray<?x?xi64>, !ndarray.ndarray<1x?xi64>) -> !ndarray.ndarray<?x?xi64>
    %2 = ndarray.ewbin %arg1, %1 {op = 0 : i32} : (!ndarray.ndarray<1x?xi64>, !ndarray.ndarray<?x?xi64>) -> !ndarray.ndarray<?x?xi64>
    return %2 : !ndarray.ndarray<?x?xi64>
}
// CHECK-LABEL: @test_binop_fusion_bcast3
// CHECK-NEXT: arith.constant
// CHECK-NEXT: arith.constant
// CHECK: memref.dim
// CHECK-NEXT: memref.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK: return %{{[0-9]+}} : memref<?x?xi64, strided<[?, ?], offset: ?>>
