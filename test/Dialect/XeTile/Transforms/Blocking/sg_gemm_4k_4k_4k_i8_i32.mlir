// RUN: imex-opt -xetile-init-duplicate --xetile-blocking --canonicalize --cse %s | FileCheck %s

gpu.module @test_kernel {

  //CHECK: gpu.func @test_gemm(%[[arg0:.*]]: memref<4096x4096xi8>, %[[arg1:.*]]: memref<4096x4096xi8>, %[[arg2:.*]]: memref<4096x4096xi32>) {
  gpu.func @test_gemm(%A: memref<4096x4096xi8>, %B: memref<4096x4096xi8>, %C: memref<4096x4096xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c128 : index
    %n = arith.muli %block_id_y, %c256 : index
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>

    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-64: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xi32> -> !xetile.tile<32x16xi32>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xi32> -> !xetile.tile<128x256xi32>

    //CHECK-COUNT-64: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x16xi32> -> vector<32x16xi32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x256xi32> -> vector<128x256xi32>

    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}] : memref<4096x4096xi8> -> !xetile.tile<32x32xi8, #xetile.tile_attr<array_length = 2 : i64>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<4096x4096xi8> -> !xetile.tile<128x256xi8>

    //CHECK-COUNT-32: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}] : memref<4096x4096xi8> -> !xetile.tile<32x16xi8, #xetile.tile_attr<array_length = 4 : i64>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xi8> -> !xetile.tile<256x256xi8>

    //CHECK-COUNT-256: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xi32> to vector<8x16xi32>
    %out:3 = scf.for %k = %c0 to %c4096 step %c256 iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
                                                          -> (!xetile.tile<128x256xi8>, !xetile.tile<256x256xi8>, vector<128x256xi32>) {

      //CHECK-COUNT-16: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x32xi8, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x32xi8>, vector<32x32xi8>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<128x256xi8> -> vector<128x256xi8>

      //CHECK-COUNT-32: %{{.*}} = xetile.load_tile %{{.*}}: !xetile.tile<32x16xi8, #xetile.tile_attr<array_length = 4 : i64>> -> vector<32x16xi8>, vector<32x16xi8>, vector<32x16xi8>, vector<32x16xi8>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<256x256xi8> -> vector<256x256xi8>

      //CHECK-COUNT-128: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xi8> to vector<8x32xi8>
      //CHECK-COUNT-2048: %{{.*}} = xetile.tile_mma %{{.*}} : vector<8x32xi8>, vector<32x16xi8>, vector<8x16xi32> -> vector<8x16xi32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
        : vector<128x256xi8>, vector<256x256xi8>, vector<128x256xi32> -> vector<128x256xi32>

      //CHECK-COUNT-16: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}] : !xetile.tile<32x32xi8, #xetile.tile_attr<array_length = 2 : i64>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c256] : !xetile.tile<128x256xi8>
      //CHECK-COUNT-32: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}] : !xetile.tile<32x16xi8, #xetile.tile_attr<array_length = 4 : i64>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c256, %c0] : !xetile.tile<256x256xi8>

      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<128x256xi8>, !xetile.tile<256x256xi8>, vector<128x256xi32>
    }
    //CHECK-COUNT-256: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xi32>, !xetile.tile<8x16xi32>
    xetile.store_tile %out#2, %c_init_tile : vector<128x256xi32>, !xetile.tile<128x256xi32>
    gpu.return
  }
}
