// RUN: imex-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// -----
func.func @test_create_nd_tdesc_1(%src: memref<24xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // expected-error@+1 {{Expecting the rank of shape, strides and offsets should match with each other}}
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] : memref<24xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_create_nd_tdesc_2(%input: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // expected-error@+1 {{It is invalid to have both or none of dynamic shape and static shape. Only one of them is needed.}}
  %1 = xegpu.create_nd_tdesc %input[%c0, %c1], [%c8, %c16], [%c16, %c1] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// -----
func.func @test_create_nd_tdesc_3(%input: memref<?xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // expected-error@+1 {{Expecting the rank of shape, strides and offsets should match with each other}}
  %1 = xegpu.create_nd_tdesc %input[%c0, %c1], [%c8, %c16], [%c16, %c1] : memref<?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// -----
func.func @test_create_nd_tdesc_4(%input: memref<?x?xf32>) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{Expecting the rank of shape, strides and offsets should match with each other}}
  %1 = xegpu.create_nd_tdesc %input[%c1], [%c8], [%c1] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_create_nd_tdesc_5(%input: memref<24x32x64xf32>) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{operand #0 must be 1D/2D memref}}
  %1 = xegpu.create_nd_tdesc %input[%c1, %c1, %c8] : memref<24x32x64xf32> -> !xegpu.tensor_desc<8x16x8xf32>
  return
}

// -----
func.func @test_create_tdesc(%src: ui64, %offsets : vector<16x8xindex>) {
  // expected-error@+1 {{operand #1 must be vector of index values of ranks 1}}
  %1 = xegpu.create_tdesc %src, %offsets: ui64, vector<16x8xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  return
}

// -----
func.func @test_load_gather(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16x8xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {memory_scope = global, chunk_size_per_lane = 8}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size_per_lane = 8}: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.scattered>

  // for fp16 the vnni factor should be 2 instead of 4.
  // expected-error@+1 {{Invalid vnni transform.}}
  %2 = xegpu.load %1, %0 {vnni_axis = 0, l1_hint = cached, l2_hint = uncached}
                          : !xegpu.tensor_desc<16x8xf16, #xegpu.scattered>, vector<16x8xi1> -> vector<4x8x4xf16>
  return
}


// -----
func.func @test_load_gather_2(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16x8xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {memory_scope = global, chunk_size_per_lane = 8}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size_per_lane = 8}: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>

  // for fp32, no vnni available.
  // expected-error@+1 {{Invalid vnni transform.}}
  %2 = xegpu.load %1, %0 {transpose = [1, 0], vnni_axis = 1, l1_hint = cached, l2_hint = uncached}
                          : !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>, vector<16x8xi1> -> vector<8x8x2xf32>
  return
}

// -----
func.func @test_load_gather_3(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16x8xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {memory_scope = global, chunk_size_per_lane = 8}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size_per_lane = 8}: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.scattered>

  // for fp16 the vnni factor should be 2 instead of 4.
  // expected-error@+1 {{Invalid vnni transform.}}
  %2 = xegpu.load %1, %0 {vnni_axis = 0, l1_hint = cached, l2_hint = uncached}
                          : !xegpu.tensor_desc<16x8xf16, #xegpu.scattered>, vector<16x8xi1> -> vector<4x8x4xf16>
  return
}
