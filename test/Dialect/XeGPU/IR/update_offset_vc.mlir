// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_update_offset_VC({{.*}}) {
func.func @test_update_offset_VC(%src: ui64, %offsets : vector<16 x index>) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc %{{.*}} : ui64 -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64 -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>

  // CHECK: xegpu.load
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  %2 = xegpu.load %1, %0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
        : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>

  // CHECK: xegpu.update_offset %{{.*}} : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %5 = xegpu.update_offset %1, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>

  return
}
