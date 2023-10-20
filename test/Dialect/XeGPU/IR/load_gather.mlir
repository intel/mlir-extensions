// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s


// CHECK-LABEL: func @test_load_gather_vc({{.*}}) {
func.func @test_load_gather_vc(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc, memory_scope = global, chunk_size_per_lane = 1}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc}: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.load
  // CHECK-SAME: {mode = vc, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  %2 = xegpu.load %1, %0 {mode = vc, l1_hint = cached, l2_hint = uncached} 
                : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  return
}

// CHECK-LABEL: func @test_load_gather_vc_2({{.*}}) {
func.func @test_load_gather_vc_2(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16x8xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc, memory_scope = global, chunk_size_per_lane = 8}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 8}
                : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>

  // CHECK: xegpu.load
  // CHECK-SAME: {mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>, vector<16x8xi1> -> vector<8x16xf32>
  %2 = xegpu.load %1, %0 {mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = uncached} 
               : !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>, vector<16x8xi1> -> vector<8x16xf32>
  return
}


// CHECK-LABEL: func @test_load_gather_vc_3({{.*}}) {
func.func @test_load_gather_vc_3(%src: ui64, %offset : index) {
  %0 = arith.constant dense<1>: vector<8xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc, memory_scope = global, chunk_size_per_lane = 8}
  // CHECK-SAME: ui64, index -> !xegpu.tensor_desc<8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offset {mode = vc, chunk_size_per_lane = 8}
                : ui64, index -> !xegpu.tensor_desc<8xf32, #xegpu.scattered>

  // CHECK: xegpu.load
  // CHECK-SAME: {mode = vc, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8xf32, #xegpu.scattered>, vector<8xi1> -> vector<8xf32>
  %2 = xegpu.load %1, %0 {mode = vc, l1_hint = cached, l2_hint = uncached}
               : !xegpu.tensor_desc<8xf32, #xegpu.scattered>, vector<8xi1> -> vector<8xf32>
  return
}


// CHECK-LABEL: func @test_load_gather_vc_4({{.*}}) {
func.func @test_load_gather_vc_4(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc, memory_scope = global, chunk_size_per_lane = 1}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 1}
                : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.load
  // CHECK-SAME: {mode = vc, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  %2 = xegpu.load %1, %0 {mode = vc, l1_hint = cached, l2_hint = uncached} 
                : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  return
}

