// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -o -| FileCheck %s

// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {

  //CHECK: gpu.func @test_gemm(%[[arg0:.*]]: memref<1024x1024xi8>, %[[arg1:.*]]: memref<1024x1024xi8>, %[[arg2:.*]]: memref<1024x1024xi32>)
  gpu.func @test_gemm(%A: memref<1024x1024xi8>, %B: memref<1024x1024xi8>, %C: memref<1024x1024xi32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y

    %m = arith.muli %block_id_x, %c32 : index
    %n = arith.muli %block_id_y, %c32 : index

    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xi32> -> !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xi32> -> !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xi32> -> !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xi32> -> !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xi32> -> !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xi32> -> !xegpu.tensor_desc<32x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<32x32xi32>

    //CHECK-COUNT-2: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>> -> vector<32x16xi32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<32x32xi32> -> vector<32x32xi32>

    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xi8> -> !xegpu.tensor_desc<32x32xi8, #xegpu.block_tdesc_attr<boundary_check = true>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xi8> -> !xetile.tile<32x32xi8>

    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg1]][{{.*}}] : memref<1024x1024xi8> -> !xegpu.tensor_desc<32x16xi8, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xi8> -> !xetile.tile<32x32xi8>

    //CHECK-COUNT-8: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [8, 16], strides = [1, 1]} : vector<32x16xi32> to vector<8x16xi32>
    //CHECK: {{.*}} = scf.for {{.*}}
    %out:3 = scf.for %k = %c0 to %c1024 step %c32 iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
                  -> (!xetile.tile<32x32xi8>, !xetile.tile<32x32xi8>, vector<32x32xi32>) {

      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x32xi8, #xegpu.block_tdesc_attr<boundary_check = true>> -> vector<32x32xi8>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<32x32xi8> -> vector<32x32xi8>

      //CHECK: {{.*}} = xegpu.load_nd {{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xi8, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xi8>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<32x32xi8> -> vector<32x32xi8>

      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xi8> from vector<2x32x16xi8>
      //CHECK-COUNT-4: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [8, 32], strides = [1, 1]} : vector<32x32xi8> to vector<8x32xi8>
      //CHECK-COUNT-8: {{.*}} = xegpu.dpas {{.*}} : vector<8x32xi8>, vector<32x16xi8>, vector<8x16xi32> -> vector<8x16xi32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<32x32xi8>, vector<32x32xi8>, vector<32x32xi32> -> vector<32x32xi32>

      //CHECK: {{.*}} = xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<32x32xi8, #xegpu.block_tdesc_attr<boundary_check = true>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32] : !xetile.tile<32x32xi8>
      //CHECK: {{.*}} = xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<32x16xi8, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0] : !xetile.tile<32x32xi8>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xi8>, !xetile.tile<32x32xi8>, vector<32x32xi32>
    }
    //CHECK-COUNT-8: xegpu.store_nd {{.*}} : vector<8x16xi32>, !xegpu.tensor_desc<8x16xi32, #xegpu.block_tdesc_attr<boundary_check = true>>
    xetile.store_tile %out#2, %c_init_tile {innner_blocks = [8, 16]}: vector<32x32xi32>, !xetile.tile<32x32xi32>
    gpu.return
  }
}
