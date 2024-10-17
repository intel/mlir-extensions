// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
  gpu.module @test_kernel {
    // CHECK: sg_scf_for(%[[ARG0:.*]]: memref<1024x1024xf16>, %[[ARG1:.*]]: memref<1024x1024xf16>)
    gpu.func @sg_scf_for(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) {
      // CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32x32xf16>
      %cst = arith.constant dense<0.000000e+00> : vector<1x1x32x32xf16>

      // CHECK: %[[c0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index

      // CHECK: %[[c64:.*]] = arith.constant 64 : index
      %c64 = arith.constant 64 : index

      // CHECK: %[[c1024:.*]] = arith.constant 1024 : index
      %c1024 = arith.constant 1024 : index

      // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[c0]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %0 = xetile.init_tile %arg0[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>

      // CHECK: %[[R1:.*]]:2 = scf.for %[[arg2:.*]] = %[[c0]] to %[[c1024]] step %[[c64]]
      // CHECK-SAME: iter_args(%[[arg3:.*]] = %[[R0]], %[[arg4:.*]] = %[[cst]]) -> (!xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, vector<32x32xf16>)
      %1:2 = scf.for %arg2 = %c0 to %c1024 step %c64 iter_args(%arg3 = %0, %arg4 = %cst) -> (!xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, vector<1x1x32x32xf16>) {

        // CHECK: %[[R10:.*]] = xegpu.load_nd %[[arg3]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
        %5 = xetile.load_tile %arg3 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>

        // CHECK: %[[R11:.*]] = xegpu.update_nd_offset %[[arg3]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
        %6 = xetile.update_tile_offset %arg3, [%c0,  %c64] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>

        // CHECK: scf.yield %[[R11]], %[[R10]] : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, vector<32x32xf16>
        scf.yield %6, %5 : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, vector<1x1x32x32xf16>
      }

      // CHECK: %[[R2:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%[[c0]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[c8:.*]] = arith.constant 8 : index
      // CHECK: %[[R3:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%[[c8]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[c16:.*]] = arith.constant 16 : index
      // CHECK: %[[R4:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%[[c16]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: %[[c24:.*]] = arith.constant 24 : index
      // CHECK: %[[R5:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%[[c24]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %2 = xetile.init_tile %arg1[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>

      // CHECK: %[[R6:.*]] = vector.extract_strided_slice %[[R1]]#1 {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
      // CHECK: %[[R7:.*]] = vector.extract_strided_slice %[[R1]]#1 {offsets = [8, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
      // CHECK: %[[R8:.*]] = vector.extract_strided_slice %[[R1]]#1 {offsets = [16, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
      // CHECK: %[[R9:.*]] = vector.extract_strided_slice %[[R1]]#1 {offsets = [24, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
      %3 = xetile.tile_unpack %1#1 {inner_blocks = array<i64: 32, 32>}  : vector<1x1x32x32xf16> -> vector<32x32xf16>
      %4 = xetile.tile_pack %3 {inner_blocks = array<i64: 8, 32>}: vector<32x32xf16> -> vector<4x1x8x32xf16>

      // CHECK: xegpu.store_nd %[[R6]], %[[R2]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R7]], %[[R3]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R8]], %[[R4]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      // CHECK: xegpu.store_nd %[[R9]], %[[R5]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      xetile.store_tile %4,  %2 : vector<4x1x8x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      gpu.return
    }
  }
