// RUN: imex-opt -xetile-init-duplicate --xetile-blocking --canonicalize --cse %s | FileCheck %s

gpu.module @test_kernel {

  //CHECK: gpu.func @test_gemm(%[[arg0:.*]]: memref<4096x4096xi8>, %[[arg1:.*]]: memref<4096x4096xi8>, %[[arg2:.*]]: memref<4096x4096xi32>) {
  gpu.func @test_gemm(%A: memref<4096x4096xi8>, %B: memref<4096x4096xi8>, %C: memref<4096x4096xi32>) {

    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[c4096:.*]] = arith.constant 4096 : index


    //CHECK: %[[x:.*]] = gpu.block_id  x
    //CHECK: %[[y:.*]] = gpu.block_id  y
    //CHECK: %[[r0:.*]] = arith.muli %[[x]], %[[c128]] : index
    //CHECK: %[[r1:.*]] = arith.muli %[[y]], %[[c256]] : index
    //CHECK: %[[r2:.*]] = xetile.init_tile %[[arg2]][%[[r0]], %[[r1]]] : memref<4096x4096xi32> -> !xetile.tile<128x256xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    //CHECK: %[[r3:.*]] = xetile.init_tile %[[arg2]][%[[r0]], %[[r1]]] : memref<4096x4096xi32> -> !xetile.tile<128x256xi32, #xetile.tile_attr<inner_blocks = [32, 16]>>
    //CHECK: %[[r4:.*]] = xetile.load_tile %[[r3]] : !xetile.tile<128x256xi32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<4x16x32x16xi32>
    //CHECK: %[[r5:.*]] = xetile.tile_unpack %[[r4]] {inner_blocks = array<i64: 32, 16>} : vector<4x16x32x16xi32> -> vector<128x256xi32>
    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg0]][%[[r0]], %[[c0]]] : memref<4096x4096xi8> -> !xetile.tile<128x256xi8, #xetile.tile_attr<inner_blocks = [32, 32]>>
    //CHECK: %[[r7:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[r1]]] : memref<4096x4096xi8> -> !xetile.tile<256x256xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>
    //CHECK: %[[r8:.*]] = xetile.tile_pack %[[r5]] {inner_blocks = array<i64: 8, 16>} : vector<128x256xi32> -> vector<16x16x8x16xi32>



    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c128 : index
    %n = arith.muli %block_id_y, %c256 : index

    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xi32> -> !xetile.tile<128x256xi32>

    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x256xi32> -> vector<128x256xi32>

    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<4096x4096xi8> -> !xetile.tile<128x256xi8>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xi8> -> !xetile.tile<256x256xi8>


    //CHECK: %[[r9:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c4096]] step %[[c256]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[r6]], %[[arg5:.*]] = %[[r7]], %[[arg6:.*]] = %[[r8]])
    //CHECK-SAME: !xetile.tile<128x256xi8, #xetile.tile_attr<inner_blocks = [32, 32]>>
    //CHECK-SAME: !xetile.tile<256x256xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<16x16x8x16xi32>
    %out:3 = scf.for %k = %c0 to %c4096 step %c256 iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
                                                          -> (!xetile.tile<128x256xi8>, !xetile.tile<256x256xi8>, vector<128x256xi32>) {

      //CHECK: %[[r10:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<128x256xi8, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<4x8x32x32xi8>
      //CHECK: %[[r11:.*]] = xetile.tile_unpack %[[r10]] {inner_blocks = array<i64: 32, 32>} : vector<4x8x32x32xi8> -> vector<128x256xi8>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<128x256xi8> -> vector<128x256xi8>

      //CHECK: %[[r12:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<256x256xi8, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<8x16x32x16xi8>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<256x256xi8> -> vector<256x256xi8>

      //CHECK: %[[r13:.*]] = xetile.tile_pack %[[r11]] {inner_blocks = array<i64: 8, 32>} : vector<128x256xi8> -> vector<16x8x8x32xi8>
      //CHECK: %[[r14:.*]] = xetile.tile_mma %[[r13]], %[[r12]], %[[arg6]] : vector<16x8x8x32xi8>, vector<8x16x32x16xi8>, vector<16x16x8x16xi32> -> vector<16x16x8x16xi32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
        : vector<128x256xi8>, vector<256x256xi8>, vector<128x256xi32> -> vector<128x256xi32>

      //CHECK: %[[r15:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c256]]] : !xetile.tile<128x256xi8, #xetile.tile_attr<inner_blocks = [32, 32]>>, index, index -> !xetile.tile<128x256xi8, #xetile.tile_attr<inner_blocks = [32, 32]>>
      //CHECK: %[[r16:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c256]],  %[[c0]]] : !xetile.tile<256x256xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index -> !xetile.tile<256x256xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c256] : !xetile.tile<128x256xi8>, index, index -> !xetile.tile<128x256xi8>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c256, %c0] : !xetile.tile<256x256xi8>, index, index -> !xetile.tile<256x256xi8>

      //CHECK: scf.yield %[[r15]], %[[r16]], %[[r14]] : !xetile.tile<128x256xi8, #xetile.tile_attr<inner_blocks = [32, 32]>>, !xetile.tile<256x256xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<16x16x8x16xi32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<128x256xi8>, !xetile.tile<256x256xi8>, vector<128x256xi32>
    }

    //CHECK: xetile.store_tile %[[r9]]#2,  %[[r2]] : vector<16x16x8x16xi32>, !xetile.tile<128x256xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_init_tile : vector<128x256xi32>, !xetile.tile<128x256xi32>
    gpu.return
  }
}
