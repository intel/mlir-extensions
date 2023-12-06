// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_store_scatter_vc({{.*}}) {
func.func @test_store_scatter_vc(%src: ui64, %offsets : vector<16 x index>, %dst: ui64) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc}
          : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %2 = xegpu.create_tdesc %dst, %offsets {mode = vc}
          : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.load
  // CHECK-SAME: {mode = vc, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  %3 = xegpu.load %1, %0 {mode = vc, l1_hint = cached, l2_hint = uncached}
                : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  // CHECK: xegpu.store
  // CHECK-SAME: {mode = vc, l1_hint = write_back, l2_hint = uncached}
  // CHECK-SAME: vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1>
  xegpu.store %3, %2, %0 {mode = vc, l1_hint = write_back, l2_hint = uncached}
                : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1>
  return
}
