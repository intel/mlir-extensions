// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
  gpu.module @test_kernel {
    // CHECK: sg_tiled_store(%[[arg0:.*]]: memref<1024x1024xf32>)
    gpu.func @sg_tiled_store(%arg0: memref<1024x1024xf32>) {
      // CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32x16xf32>
      %cst = arith.constant dense<0.000000e+00> : vector<1x2x32x16xf32>

      // CHECK: %[[c0:.*]] = arith.constant 0 : index
      // CHECK: %[[c32:.*]] = arith.constant 32 : index
      // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[c48:.*]] = arith.constant 48 : index
      // CHECK: %[[R1:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c48]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[c8:.*]] = arith.constant 8 : index
      // CHECK: %[[R2:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c8]], %[[c32]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[R3:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c8]], %[[c48]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[c16:.*]] = arith.constant 16 : index
      // CHECK: %[[R4:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c16]], %[[c32]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[R5:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c16]], %[[c48]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[c24:.*]] = arith.constant 24 : index
      // CHECK: %[[R6:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c24]], %[[c32]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[R7:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c24]], %[[c48]]] : memref<1024x1024xf32>
      // CHECK-SAME: !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %0 = xetile.init_tile %arg0[0, 32] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>

      // CHECK: %[[R8:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      // CHECK: %[[R9:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      // CHECK: %[[R10:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      // CHECK: %[[R11:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %1 = xetile.tile_unpack %cst {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf32> -> vector<32x32xf32>
      %2 = xetile.tile_pack %1 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf32> -> vector<4x2x8x16xf32>

      // CHECK: xegpu.store_nd %[[R8]], %[[R0]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R8]], %[[R1]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R9]], %[[R2]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R9]], %[[R3]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R10]], %[[R4]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R10]], %[[R5]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R11]], %[[R6]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R11]], %[[R7]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}>
      // CHECK-SAME: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      xetile.store_tile %2,  %0 : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      gpu.return
    }
  }
