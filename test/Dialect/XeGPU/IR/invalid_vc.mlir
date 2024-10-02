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

  // expected-error@+1 {{Expecting the TensorDesc rank is up to 2 and not greater than the ranks of shape, strides, offsets or the memref source}}
  %1 = xegpu.create_nd_tdesc %input[%c1, %c1, %c8]
                              : memref<24x32x64xf32> -> !xegpu.tensor_desc<8x16x8xf32>
  return
}

// -----
func.func @test_create_tdesc(%src: ui64) {
  // expected-error@+1 {{Incorrect TensorDesc shape}}
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
                              : ui64 -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<>>
  return
}

// -----
func.func @test_load_gather(%src: ui64) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc {{.*}} : ui64
  // CHECK-SAME: !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64
        -> !xegpu.tensor_desc<16x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>

  // expected-error@+1 {{failed to verify that all of {value, TensorDesc} have same rank}}
  %2 = xegpu.load %1, %0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}
                          : !xegpu.tensor_desc<16x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1> -> vector<8x8x4xf16>
  return
}

// -----
func.func @test_create_tdesc_oversized(%src: ui64) {
  // expected-error@+1 {{total access size (simd_lanes * chunk_size * sizeof(elemTy)) is upto 512 bytes}}
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64
              -> !xegpu.tensor_desc<16x16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 16>>
  return
}

// -----
func.func @test_create_tdesc_invalid_chunk_size(%src: ui64) {
  // expected-error@+1 {{Invalid chunk_size. Supported values are 1, 2, 3, 4, 8, 16, 32, 64, 128, or 256.}}
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64
              -> !xegpu.tensor_desc<16x7xf32, #xegpu.scatter_tdesc_attr<chunk_size = 7>>
  return
}

// -----
func.func @test_create_tdesc_unaligned(%src: ui64) {
  // expected-error@+1 {{access size (chunk_size * sizeof(elemTy)) should be 32-bit aligned}}
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64
              -> !xegpu.tensor_desc<16x3xf16, #xegpu.scatter_tdesc_attr<chunk_size = 3>>
  return
}
