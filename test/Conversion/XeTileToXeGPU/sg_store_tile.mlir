// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_tiled_store(%[[arg0:.*]]: memref<1024x1024xf32>) {
	gpu.func @sg_tiled_store(%a: memref<1024x1024xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32x16xf32>
    //CHECK: %[[R0:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[R1:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[R2:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[R3:.*]] = vector.extract_strided_slice %[[cst]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
		%result = arith.constant dense<0.0>: vector<32x32xf32>
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[R4:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c48:.*]] = arith.constant 48 : index
    //CHECK: %[[R5:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c48]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[R6:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c8]], %[[c32]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[R7:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c8]], %[[c48]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[R8:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c16]], %[[c32]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[R9:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c16]], %[[c48]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[R10:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c24]], %[[c32]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[R11:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c24]], %[[c48]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
		%1 = xetile.init_tile %a[0, 32] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>

    //CHECK: xegpu.store_nd %[[R0]], %[[R4]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[R0]], %[[R5]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[R1]], %[[R6]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[R1]], %[[R7]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[R2]], %[[R8]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[R2]], %[[R9]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[R3]], %[[R10]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[R3]], %[[R11]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
		xetile.store_tile %result, %1: vector<32x32xf32>, !xetile.tile<32x32xf32>
		gpu.return
	}
}
