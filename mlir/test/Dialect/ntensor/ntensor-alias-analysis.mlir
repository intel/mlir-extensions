// RUN: imex-opt %s -pass-pipeline='func.func(ntensor-alias-analysis)' --split-input-file | FileCheck %s

// CHECK-LABEL: func @test({{.*}}) {
func.func @test(%t: !ntensor.ntensor<8x16x4xf32>, %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: ntensor.subview
  // CHECK-SAME: {ntensor_readonly}
  %1 = ntensor.subview %t[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : !ntensor.ntensor<8x16x4xf32> to !ntensor.ntensor<?x?x?xf32>

  return
}

// -----

// CHECK-LABEL: func @test({{.*}}) {
func.func @test(%t: !ntensor.ntensor<?xf32>, %idx : index, %val : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: ntensor.subview
  // CHECK-SAME: %{{.*}}[%{{.*}}] [%{{.*}}] [%{{.*}}] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  %1 = ntensor.subview %t[%c0] [%idx] [%c1] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>

  ntensor.store %val, %1[%idx] : !ntensor.ntensor<?xf32>

  return
}

// -----

// CHECK-LABEL: func @test({{.*}}) {
func.func @test(%t: !ntensor.ntensor<?xf32>, %idx : index, %val : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: ntensor.subview
  // CHECK-SAME: %{{.*}}[%{{.*}}] [%{{.*}}] [%{{.*}}] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  %1 = ntensor.subview %t[%c0] [%idx] [%c1] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>

  ntensor.store %val, %t[%idx] : !ntensor.ntensor<?xf32>

  return
}

// -----

// CHECK-LABEL: func @test({{.*}}) {
func.func @test(%t: !ntensor.ntensor<?xf32>, %idx : index, %val : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: ntensor.subview
  // CHECK-SAME: {ntensor_readonly}
  %1 = ntensor.subview %t[%c0] [%idx] [%c1] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>

  %2 = ntensor.create(%c1) : !ntensor.ntensor<?xf32>
  ntensor.store %val, %2[%idx] : !ntensor.ntensor<?xf32>

  return
}
