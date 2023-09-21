// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s
// CHECK-LABEL: func @test_store_nd_0({{.*}}) {
func.func @test_store_nd_0(%src: memref<24x32xf16>, %dst: memref<24x32xf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {memory_scope = global, boundary_check = true}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {memory_scope = global, boundary_check = true}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 {l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

  // CHECK: xegpu.store_nd
  // CHECK-SAME: {l1_hint = write_back, l2_hint = uncached}
  // CHECK-SAME: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  xegpu.store_nd %3, %2 {l1_hint = write_back, l2_hint = uncached}: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  return
}
