// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s
// CHECK-LABEL: func @test_load_nd_0({{.*}}) {
func.func @test_load_nd_0(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

  // CHECK: xegpu.load_nd
  // CHECK-SAME:{transpose = [1, 0], l1_hint = cached, l2_hint = uncached, l3_hint = streaming}
  // CHECK-SAME:!xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
  %3 = xegpu.load_nd %1 {transpose = [1, 0], l1_hint = cached, l2_hint = uncached, l3_hint=streaming} : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
  return
}

// CHECK-LABEL: func @test_load_nd_1({{.*}}) {
func.func @test_load_nd_1(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg1, %arg2]
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {vnni_axis = 0, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  %2 = xegpu.load_nd %1 {vnni_axis = 0, l1_hint = cached, l2_hint = uncached} : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// CHECK-LABEL: func @test_load_nd_2({{.*}}) {
func.func @test_load_nd_2(%src: ui64, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: ui64 -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] : ui64 -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.load_nd
  // CHECK-SAME: {vnni_axis = 1, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
  %2 = xegpu.load_nd %1 {vnni_axis = 1, l1_hint = cached, l2_hint = uncached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
  return
}
