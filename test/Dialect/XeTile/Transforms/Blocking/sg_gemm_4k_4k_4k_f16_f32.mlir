// RUN: imex-opt --xetile-init-duplicate --xetile-blocking --canonicalize --cse %s | FileCheck %s

gpu.module @test_kernel {

  //CHECK:  gpu.func @test_gemm(%[[arg0:.*]]: memref<4096x4096xf16>, %[[arg1:.*]]: memref<4096x4096xf16>, %[[arg2:.*]]: memref<4096x4096xf32>)
  gpu.func @test_gemm(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c64 : index
    %n = arith.muli %block_id_y, %c128 : index

    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}] : memref<4096x4096xf32> -> !xetile.tile<32x16xf32>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32> -> !xetile.tile<64x128xf32>

    //CHECK-COUNT-16: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x16xf32> -> vector<32x16xf32>
    //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x128xf32> -> vector<64x128xf32>

    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}] : memref<4096x4096xf16> -> !xetile.tile<32x16xf16>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<4096x4096xf16> -> !xetile.tile<64x128xf16>

    //CHECK-COUNT-32: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}] : memref<4096x4096xf16> -> !xetile.tile<32x16xf16>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xf16> -> !xetile.tile<128x128xf16>

    %out:3 = scf.for %k = %c0 to %c4096 step %c128
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<64x128xf16>, !xetile.tile<128x128xf16>, vector<64x128xf32>) {
      //CHECK-COUNT-16: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<64x128xf16> -> vector<64x128xf16>

      //CHECK-COUNT-32: %{{.*}} = xetile.load_tile %{{.*}}: !xetile.tile<32x16xf16> -> vector<32x16xf16>
      //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<128x128xf16> -> vector<128x128xf16>

      //CHECK-COUNT-512: %{{.*}} = xetile.tile_mma %{{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
        : vector<64x128xf16>, vector<128x128xf16>, vector<64x128xf32> -> vector<64x128xf32>

      //CHECK-COUNT-16: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}] : !xetile.tile<32x16xf16>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c128] : !xetile.tile<64x128xf16>

      //CHECK-COUNT-32: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}] : !xetile.tile<32x16xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c128, %c0]
        : !xetile.tile<128x128xf16>

      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<64x128xf16>, !xetile.tile<128x128xf16>, vector<64x128xf32>
    }
    //CHECK-COUNT-64: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %out#2, %c_init_tile : vector<64x128xf32>, !xetile.tile<64x128xf32>
    gpu.return
  }
}
