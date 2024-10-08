// RUN: imex-opt --split-input-file --xetile-init-duplicate --new-xetile-blocking --canonicalize \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -o -| FileCheck %s
// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {
  // CHECK: gpu.func @test_gemm(%[[A:.*]]: memref<1024x1024xf16>, %[[arg1:.*]]: memref<1024x1024xf16>, %[[C:.*]]: memref<1024x1024xf32>)
  gpu.func @test_gemm(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index

    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y

    %m = arith.muli %block_id_x, %c32 : index
    %n = arith.muli %block_id_y, %c32 : index
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<32x32xf32> -> vector<32x32xf32>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[arg1]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[arg1]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
// CHECK: scf.for %{{.*}}= %{{.*}}to %{{.*}}step %{{.*}}iter_args(%{{.*}}= %{{.*}}, %[[ARG5:.*]] = %[[T1]], %[[ARG6:.*]] = %[[T2]], %{{.*}}= %{{.*}}, %{{.*}}= %{{.*}}, %{{.*}}= %{{.*}}, %{{.*}}= %{{.*}}, %{{.*}}= %{{.*}}, %{{.*}}= %{{.*}}, %{{.*}}= %{{.*}}, %{{.*}} = %{{.*}}) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
    %out:3 = scf.for %k = %c0 to %c1024 step %c32
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      %a_value = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  // Check if array_length is 1 for the load + transpose + MMA B case.
  //
  // xegpu.load_nd %[[ARG5]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
  // xegpu.load_nd %[[ARG6]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %b_transpose = xetile.transpose %b_value, [1, 0] : vector<32x32xf16> -> vector<32x32xf16>
      %c_new_value = xetile.tile_mma %a_value, %b_transpose, %c_value : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32] : !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0] : !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }
    xetile.store_tile %out#2, %c_init_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}
