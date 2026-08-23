// RUN: imex-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// -----
func.func @test_create_nd_tdesc_vc_1(%src: memref<24xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // expected-error@+1 {{Expecting the TensorDesc rank is not greater than the ranks of shape, strides or the memref source.}}
  %1 = xegpu.create_nd_tdesc %src : memref<24xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_create_nd_tdesc_vc_4(%input: ui64) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{Expecting the TensorDesc rank is not greater than the ranks of shape, strides or the memref source.}}
  %1 = xegpu.create_nd_tdesc %input, shape: [%c8], strides: [%c1]
                              : ui64 -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// Explicit shape/strides on a memref source is deprecated and rejected; they
// are inferred from the memref instead.
func.func @test_create_nd_tdesc_vc_5(%input: memref<?x?xf32>, %h : index, %w : index) {
  %c1 = arith.constant 1 : index

  // expected-error@+1 {{shape and strides should not be specified for a memref source}}
  %1 = xegpu.create_nd_tdesc %input, shape: [%h, %w], strides: [%w, %c1]
                              : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}
