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
  %1 = xegpu.create_nd_tdesc %input[%c0, %c1], shape: [%c8, %c16], strides: [%c16, %c1] : memref<?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// -----
func.func @test_create_nd_tdesc_vc_4(%input: memref<?x?xf32>) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{expected 2 offset values, got 1}}
  %1 = xegpu.create_nd_tdesc %input[%c1], shape: [%c8], strides: [%c1]
                              : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_load_gather(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc {{.*}} : ui64, vector<16xindex>
  // CHECK-SAME: !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex>
        -> !xegpu.tensor_desc<16x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>

  // expected-error@+1 {{neither a valid distribution for SIMT nor consistent with the tensor descriptor for SIMD}}
  %2 = xegpu.load %1, %0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
                          : !xegpu.tensor_desc<16x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1> -> vector<8x8x4xf16>
  return
}

// -----
func.func @test_create_tdesc_unaligned(%src: ui64, %offsets : vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex>
  // expected-error@+1 {{expected last dim of tensor to be a multiple of 2}}
              -> !xegpu.tensor_desc<16x3xf16, #xegpu.scatter_tdesc_attr<chunk_size = 3>>
  return
}
