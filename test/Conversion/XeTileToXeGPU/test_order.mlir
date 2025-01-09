// RUN: imex-opt --split-input-file --xetile-canonicalization --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: @test_func
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x64xf16>, %[[ARG1:.*]]: memref<64x128xf16, strided<[1, 64]>>) {
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[R_CAST:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [128, 64], strides: [64, 1] : memref<64x128xf16, strided<[1, 64]>> to memref<128x64xf16, strided<[64, 1]>>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[R_CAST]][%[[C0]], %[[C0]]] : memref<128x64xf16, strided<[64, 1]>> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[R_CAST]][%[[C16]], %[[C0]]] : memref<128x64xf16, strided<[64, 1]>> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T8:.*]] = xegpu.load_nd %[[T1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf16>
// CHECK: %[[T9:.*]] = xegpu.load_nd %[[T2]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf16>
// CHECK: %[[T19:.*]] = xegpu.update_nd_offset %[[T1]], [%[[C0]], %[[C16]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T20:.*]] = xegpu.update_nd_offset %[[T2]], [%[[C0]], %[[C16]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T26:.*]] = xegpu.load_nd %[[T19]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf16>
// CHECK: %[[T27:.*]] = xegpu.load_nd %[[T20]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf16>
gpu.module @test_kernel {
func.func @test_func(%A : memref<128x64xf16>, %B : memref<64x128xf16, strided<[1, 64], offset: 0>>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c16  = arith.constant 16 : index
  %A_block_iter0 = xetile.init_tile %A[%c0, %c0] : memref<128x64xf16> -> !xetile.tile<32x16xf16>
  %B_block_iter0 = xetile.init_tile %B[%c0, %c0] : memref<64x128xf16, strided<[1, 64], offset: 0>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1]>>

  %A_block_value0 = xetile.load_tile %A_block_iter0 : !xetile.tile<32x16xf16> -> vector<32x16xf16>
  %B_block_value0 = xetile.load_tile %B_block_iter0 : !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0,1]>> -> vector<16x32xf16>

  %mma_out0 = xetile.tile_mma %A_block_value0, %B_block_value0 : vector<32x16xf16>, vector<16x32xf16> -> vector<32x32xf32>

  %A_block_iter1 = xetile.update_tile_offset %A_block_iter0, [%c0, %c16] : !xetile.tile<32x16xf16>
  %B_block_iter1 = xetile.update_tile_offset %B_block_iter0, [%c16, %c0] : !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0,1]>>

  %A_block_value1 = xetile.load_tile %A_block_iter1 : !xetile.tile<32x16xf16> -> vector<32x16xf16>
  %B_block_value1 = xetile.load_tile %B_block_iter1  : !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0,1]>> -> vector<16x32xf16>

  %mma_out1 = xetile.tile_mma %A_block_value1, %B_block_value1, %mma_out0 : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>

  return
}
}
