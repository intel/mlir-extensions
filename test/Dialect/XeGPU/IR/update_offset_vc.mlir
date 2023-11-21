// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_update_offset_VC({{.*}}) {
func.func @test_update_offset_VC(%src: ui64, %offsets : vector<16 x index>) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc, chunk_size_per_lane = 1}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc}
              : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.load
  // CHECK-SAME: {mode = vc, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  %2 = xegpu.load %1, %0 {mode = vc, l1_hint = cached, l2_hint = uncached}
        : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>

  %3 = arith.constant dense<16>: vector<16 x index>
  %4 = arith.addi %offsets, %3: vector<16 x index>

  // CHECK: xegpu.update_offset
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %5 = xegpu.update_offset %1, %4 {mode = vc}
      : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  return
}
