// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s
// CHECK-LABEL: func @test_create_nd_tdesc_0({{.*}}) {
func.func @test_create_nd_tdesc_0(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %2 = xegpu.create_nd_tdesc %src[2, 4]
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_1({{.*}}) {
func.func @test_create_nd_tdesc_1(%src: memref<24x32xf32>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg1, %arg2]
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y]
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_2({{.*}}) {
func.func @test_create_nd_tdesc_2(%src: ui64, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: ui64 -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] : ui64 -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_3({{.*}}) {
func.func @test_create_nd_tdesc_3(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_4({{.*}}) {
func.func @test_create_nd_tdesc_4(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {boundary_check = true} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_5({{.*}}) {
func.func @test_create_nd_tdesc_5(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {memory_scope = slm} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_6({{.*}}) {
func.func @test_create_nd_tdesc_6(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {memory_scope = slm, boundary_check = true} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_7({{.*}}) {
func.func @test_create_nd_tdesc_7(%src: memref<1024xf32>, %offset : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<1024xf32> -> !xegpu.tensor_desc<16xf32>
  %1 = xegpu.create_nd_tdesc %src[%offset] : memref<1024xf32> -> !xegpu.tensor_desc<16xf32>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_8({{.*}}) {
func.func @test_create_nd_tdesc_8(%src: memref<?x?xf32>, %w : index, %h : index, %x : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[8, %x], [%h, %w], [%w, %c1] {memory_scope = slm, boundary_check = true} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}
