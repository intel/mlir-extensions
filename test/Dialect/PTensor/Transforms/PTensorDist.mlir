// RUN: imex-opt --split-input-file --ptensor-dist %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_arange(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = "ptensor.arange"(%arg0, %arg1, %arg2, %c0, %c1) : (i64, i64, i64, index, index) -> !ptensor.ptensor<tensor<?xi64>>
    %1 = ptensor.extract_slice %0[%c0][%c3][%c3] : !ptensor.ptensor<tensor<?xi64>> to !ptensor.ptensor<tensor<?xi64>>
    %2 ="ptensor.ewbin"(%1, %1) {op = 0 : i32} : (!ptensor.ptensor<tensor<?xi64>>, !ptensor.ptensor<tensor<?xi64>>) -> !ptensor.ptensor<tensor<?xi64>>
    %3 = builtin.unrealized_conversion_cast %1 : !ptensor.ptensor<tensor<?xi64>> to i64
    return %3 : i64
}
// CHECK-LABEL: func.func @test_arange
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.cmpi
// CHECK: arith.select
// CHECK: arith.addi
// CHECK: arith.addi
// CHECK: arith.subi
// CHECK: arith.divui
// CHECK: "dist.nprocs"
// CHECK: "dist.prank"
// CHECK: memref.alloc
// CHECK: memref.store
// CHECK: "dist.local_shape"
// CHECK: memref.load
// CHECK: "dist.local_offsets"
// CHECK: memref.load
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: "ptensor.arange"
// CHECK: "dist.init_dist_tensor"
// CHECK: "dist.extract_from_dist"
// CHECK: "dist.extract_from_dist"
// CHECK: "ptensor.ewbin"
// CHECK: "dist.extract_from_dist"
// CHECK: "dist.extract_from_dist"
// CHECK: "dist.nprocs"
// CHECK: "dist.prank"
// CHECK: "dist.local_offsets"
// CHECK: builtin.unrealized_conversion_cast
