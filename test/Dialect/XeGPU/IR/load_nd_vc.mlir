// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// -- SIMD ---
// CHECK-LABEL: func @test_load_nd_simd_f32({{.*}}) {
func.func @test_load_nd_simd_f32(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

  // CHECK: xegpu.load_nd
  // CHECK-SAME:{mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = uncached, l3_hint = streaming}
  // CHECK-SAME:!xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
  %3 = xegpu.load_nd %1 {mode= vc, transpose = [1, 0], l1_hint = cached, l2_hint = uncached, l3_hint=streaming} : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
  return
}

// CHECK-LABEL: func @test_load_nd_simd_f16({{.*}}) {
func.func @test_load_nd_simd_f16(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg1, %arg2]
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y] {mode = vc}
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  %2 = xegpu.load_nd %1 {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = uncached} : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// CHECK-LABEL: func @test_load_nd_simd_bf16({{.*}}) {
func.func @test_load_nd_simd_bf16(%src: ui64, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: ui64 -> !xegpu.tensor_desc<8x16xbf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {mode = vc} : ui64 -> !xegpu.tensor_desc<8x16xbf16>
  // CHECK: xegpu.load_nd
  // CHECK-SAME: {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xbf16> -> vector<8x8x2xbf16>
  %2 = xegpu.load_nd %1 {mode=vc, vnni_axis = 1, l1_hint = cached, l2_hint = uncached} : !xegpu.tensor_desc<8x16xbf16> -> vector<8x8x2xbf16>

  return
}

// CHECK-LABEL: func @test_load_nd_block_array_simd_f16({{.*}}) {
func.func @test_load_nd_block_array_simd_f16(%src: memref<8x32xf16>) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<array_length = 2>>
  %1 = xegpu.create_nd_tdesc %src[0, 0] {mode = vc}
      : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<array_length = 2>>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {mode = vc, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<array_length = 2>> -> vector<2x8x16xf16>
  %2 = xegpu.load_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}
              : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<array_length = 2>> -> vector<2x8x16xf16>
  return
}


// CHECK-LABEL: func @test_load_nd_transpose_bit_width_simd_f16({{.*}}) {
func.func @test_load_nd_transpose_bit_width_simd_f16(%src: memref<8x32xf16>) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
  %1 = xegpu.create_nd_tdesc %src[0, 0] {mode = vc} : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {mode = vc, transpose = [1, 0], transpose_bit_width = 32, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x32xf16> -> vector<16x8x2xf16>
  %2 = xegpu.load_nd %1 {mode = vc, transpose = [1, 0], transpose_bit_width = 32, l1_hint = cached, l2_hint = uncached}
              : !xegpu.tensor_desc<8x32xf16> -> vector<16x8x2xf16>
  return
}
