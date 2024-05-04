// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_store_nd_vc_bf16({{.*}}) {
func.func @test_store_nd_vc_bf16(%src: memref<24x32xbf16>, %dst: memref<24x32xbf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1]
      : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xbf16> -> vector<8x16xbf16>
  %3 = xegpu.load_nd %1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}: !xegpu.tensor_desc<8x16xbf16> -> vector<8x16xbf16>

  // CHECK: xegpu.store_nd
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: vector<8x16xbf16>, !xegpu.tensor_desc<8x16xbf16>
  xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}: vector<8x16xbf16>, !xegpu.tensor_desc<8x16xbf16>
  return
}

// CHECK-LABEL: func @test_store_nd_vc_f64({{.*}}) {
func.func @test_store_nd_vc_f64(%src: memref<24x32xf64>, %dst: memref<24x32xf64>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1]
      : memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf64> -> vector<8x16xf64>
  %3 = xegpu.load_nd %1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}: !xegpu.tensor_desc<8x16xf64> -> vector<8x16xf64>

  // CHECK: xegpu.store_nd
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: vector<8x16xf64>, !xegpu.tensor_desc<8x16xf64>
  xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}: vector<8x16xf64>, !xegpu.tensor_desc<8x16xf64>
  return
}

// CHECK-LABEL: func @test_store_nd_vc_i8({{.*}}) {
func.func @test_store_nd_vc_i8(%src: memref<24x32xi8>, %dst: memref<24x32xi8>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1]
      : memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xi8> -> vector<8x16xi8>
  %3 = xegpu.load_nd %1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}: !xegpu.tensor_desc<8x16xi8> -> vector<8x16xi8>

  // CHECK: xegpu.store_nd
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: vector<8x16xi8>, !xegpu.tensor_desc<8x16xi8>
  xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}: vector<8x16xi8>, !xegpu.tensor_desc<8x16xi8>
  return
}
