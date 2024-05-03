// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s
// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_0({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_0(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.prefetch_nd %{{.*}} : !xegpu.tensor_desc<8x16xf32>
  xegpu.prefetch_nd %1 : !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_1({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_1(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16>
  xegpu.prefetch_nd %1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}: !xegpu.tensor_desc<8x16xf16>
  return
}


// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_i8({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_i8(%src: memref<24x32xi8>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] : memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>

  // CHECK: xegpu.prefetch_nd %{{.*}} : !xegpu.tensor_desc<8x16xi8>
  xegpu.prefetch_nd %1 : !xegpu.tensor_desc<8x16xi8>

  return
}

// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_bf16({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_bf16(%src: memref<24x32xbf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y]
      : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  // CHECK: xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xbf16>
  xegpu.prefetch_nd %1 {l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>}: !xegpu.tensor_desc<8x16xbf16>
  return
}
