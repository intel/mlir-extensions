// RUN: imex-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// -----
func.func @test_create_nd_tdesc_vc_1(%src: memref<24xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // expected-error@+1 {{expected mixed offsets rank to match mixed sizes rank (2 vs 1) so the rank of the result type is well-formed}}
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] : memref<24xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_create_nd_tdesc_vc_3(%input: memref<?xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // expected-error@+1 {{expected 1 offset values, got 2}}
  %1 = xegpu.create_nd_tdesc %input[%c0, %c1], [%c8, %c16], [%c16, %c1] : memref<?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// -----
func.func @test_create_nd_tdesc_vc_4(%input: memref<?x?xf32>) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{expected 2 offset values, got 1}}
  %1 = xegpu.create_nd_tdesc %input[%c1], [%c8], [%c1]
                              : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_create_nd_tdesc_vc_5(%input: memref<24x32x64xf32>) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{Expecting the rank of shape, strides, offsets}}
  %1 = xegpu.create_nd_tdesc %input[%c1, %c1, %c8]
                              : memref<24x32x64xf32> -> !xegpu.tensor_desc<8x16x8xf32>
  return
}

// -----
func.func @test_create_tdesc(%src: ui64, %offsets : vector<16x8xindex>) {
  // expected-error@+1 {{operand #1 must be vector of index values of ranks 1}}
  %1 = xegpu.create_tdesc %src, %offsets
                              : ui64, vector<16x8xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
  return
}

// -----
func.func @test_load_gather(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16x8xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {chunk_size = 8}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size = 8}
                              : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>

  // expected-error@+1 {{failed to verify that all of {value, TensorDesc} have same rank}}
  %2 = xegpu.load %1, %0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
                          : !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>, vector<16x8xi1> -> vector<8x8x4xf16>
  return
}
