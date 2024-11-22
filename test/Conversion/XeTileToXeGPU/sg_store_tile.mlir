// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_tiled_store(%[[arg0:.*]]: memref<1024x1024xf32>) {
	gpu.func @sg_tiled_store(%a: memref<1024x1024xf32>) {

		%result = arith.constant dense<0.0>: vector<32x32xf32>
    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
		%1 = xetile.init_tile %a[0, 32] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>

    //CHECK-COUNT-8: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
		xetile.store_tile %result, %1: vector<32x32xf32>, !xetile.tile<32x32xf32>
		gpu.return
	}

  // test cache attributes acceptance
  //CHECK: gpu.func @sg_tiled_store_cache_attr(%[[arg0:.*]]: memref<1024x1024xf32>) {
	gpu.func @sg_tiled_store_cache_attr(%a: memref<1024x1024xf32>) {

		%result = arith.constant dense<0.0>: vector<32x32xf32>
    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
		%1 = xetile.init_tile %a[0, 32] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>

    //CHECK: xegpu.store_nd %cst, %0 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %cst, %1 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %cst, %2 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %cst, %3 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %cst, %4 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %cst, %5 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %cst, %6 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %cst, %7 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xetile.store_tile %result, %1 {l1_hint = #xetile.cache_hint<uncached>, l3_hint = #xetile.cache_hint<uncached>} : vector<32x32xf32>, !xetile.tile<32x32xf32>
		gpu.return
	}
}
