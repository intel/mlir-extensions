// RUN: imex-opt --split-input-file --xetile-blocking --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: @test_func
// CHECK-SAME: (%[[A:.*]]: memref<128x64xf16>, %[[B:.*]]: memref<64x128xf16, strided<[1, 64]>>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[D0:.*]] = xegpu.create_nd_tdesc %[[A]][%[[C0]], %[[C0]]] : memref<128x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
// CHECK: %[[D1:.*]] = xegpu.create_nd_tdesc %[[B]][%[[C0]], %[[C0]]] : memref<64x128xf16, strided<[1, 64]>> -> !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
// CHECK: %[[D2:.*]] = xegpu.create_nd_tdesc %[[B]][%[[C16]], %[[C0]]] : memref<64x128xf16, strided<[1, 64]>> -> !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
// CHECK: %{{.*}} = xegpu.load_nd %[[D0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<32x16xf16>
// CHECK: %{{.*}} = xegpu.load_nd %[[D1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16x2xf16>
// CHECK: %{{.*}} = xegpu.load_nd %[[D2]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16x2xf16>
// CHECK: %[[D3:.*]] = xegpu.update_nd_offset %[[D0]], [%[[C0]], %[[C16]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
// CHECK: %[[D4:.*]] = xegpu.update_nd_offset %[[D1]], [%[[C16]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
// CHECK: %[[D5:.*]] = xegpu.update_nd_offset %[[D2]], [%[[C16]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
// CHECK: %{{.*}} = xegpu.load_nd %[[D3]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<32x16xf16>
// CHECK: %{{.*}} = xegpu.load_nd %[[D4]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16x2xf16>
// CHECK: %{{.*}} = xegpu.load_nd %[[D5]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16x2xf16>
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

  %A_block_iter1 = xetile.update_tile_offset %A_block_iter0, [%c0, %c16] : !xetile.tile<32x16xf16>, index, index -> !xetile.tile<32x16xf16>
  %B_block_iter1 = xetile.update_tile_offset %B_block_iter0, [%c16, %c0] : !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0,1]>>, index, index -> !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0,1]>>

  %A_block_value1 = xetile.load_tile %A_block_iter1 : !xetile.tile<32x16xf16> -> vector<32x16xf16>
  %B_block_value1 = xetile.load_tile %B_block_iter1  : !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0,1]>> -> vector<16x32xf16>

  %mma_out1 = xetile.tile_mma %A_block_value1, %B_block_value1, %mma_out0 : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>

  return
}
}
