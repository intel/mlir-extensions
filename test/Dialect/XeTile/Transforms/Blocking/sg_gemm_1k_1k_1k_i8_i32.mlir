// RUN: imex-opt --xetile-init-duplicate --xetile-blocking --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {

  //CHECK: gpu.func @test_gemm(%[[arg0:.*]]: memref<1024x1024xi8>, %[[arg1:.*]]: memref<1024x1024xi8>, %[[arg2:.*]]: memref<1024x1024xi32>)
  gpu.func @test_gemm(%A: memref<1024x1024xi8>, %B: memref<1024x1024xi8>, %C: memref<1024x1024xi32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c64 : index
    %n = arith.muli %block_id_y, %c64 : index

    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<8x16xi32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<1024x1024xi32> -> !xetile.tile<32x16xi32>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<64x64xi32>

    //CHECK-COUNT-8: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x16xi32> -> vector<32x16xi32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x64xi32> -> vector<64x64xi32>

    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xi8> -> !xetile.tile<32x32xi8, #xetile.tile_attr<array_length = 2 : i64>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xi8> -> !xetile.tile<64x64xi8>

    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1024x1024xi8> -> !xetile.tile<32x16xi8, #xetile.tile_attr<array_length = 4 : i64>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xi8> -> !xetile.tile<64x64xi8>

    //CHECK-COUNT-32: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xi32> to vector<8x16xi32>
    %out:3 = scf.for %k = %c0 to %c1024 step %c64 iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
                  -> (!xetile.tile<64x64xi8>, !xetile.tile<64x64xi8>, vector<64x64xi32>) {
      //CHECK-COUNT-2: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x32xi8, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x32xi8>, vector<32x32xi8>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<64x64xi8> -> vector<64x64xi8>

      //CHECK-COUNT-2: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x16xi8, #xetile.tile_attr<array_length = 4 : i64>> -> vector<32x16xi8>, vector<32x16xi8>, vector<32x16xi8>, vector<32x16xi8>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<64x64xi8> -> vector<64x64xi8>

      //CHECK-COUNT-16: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xi8> to vector<8x32xi8>
      //CHECK-COUNT-64: %{{.*}} = xetile.tile_mma %{{.*}} : vector<8x32xi8>, vector<32x16xi8>, vector<8x16xi32> -> vector<8x16xi32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<64x64xi8>, vector<64x64xi8>, vector<64x64xi32> -> vector<64x64xi32>

      //CHECK-COUNT-2: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<32x32xi8, #xetile.tile_attr<array_length = 2 : i64>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64] : !xetile.tile<64x64xi8>

      //CHECK-COUNT-2: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<32x16xi8, #xetile.tile_attr<array_length = 4 : i64>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0] : !xetile.tile<64x64xi8>

      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<64x64xi8>, !xetile.tile<64x64xi8>, vector<64x64xi32>
    }
    //CHECK-COUNT-32: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xi32>, !xetile.tile<8x16xi32>
    xetile.store_tile %out#2, %c_init_tile {innner_blocks = [8, 16]}: vector<64x64xi32>, !xetile.tile<64x64xi32>
    gpu.return
  }
}
