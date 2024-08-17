// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_store_scatter({{.*}}) {
func.func @test_store_scatter(%src: ui64, %offsets : vector<16xindex>, %dst: ui64) {
  %0 = arith.constant dense<true>: vector<16xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME:
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.create_tdesc %src, %offsets
          : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>

  // CHECK: xegpu.create_tdesc
  // CHECK-SAME:
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %2 = xegpu.create_tdesc %dst, %offsets
                : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>

  // CHECK: xegpu.load
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  %3 = xegpu.load %1, %0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
                  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  // CHECK: xegpu.store
  // CHECK-SAME: {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}
  // CHECK-SAME: vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
  xegpu.store %3, %2, %0 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}
                  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
  return
}
