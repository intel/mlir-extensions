// RUN: imex-opt --xetile-init-duplicate --xetile-blocking --canonicalize --cse %s | FileCheck %s

gpu.module @test_kernel {

  //CHECK:  gpu.func @test_gemm(%[[arg0:.*]]: memref<4096x4096xf16>, %[[arg1:.*]]: memref<4096x4096xf16>, %[[arg2:.*]]: memref<4096x4096xf32>)
  gpu.func @test_gemm(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[c4096:.*]] = arith.constant 4096 : index
    //CHECK: %[[x:.*]] = gpu.block_id  x
    //CHECK: %[[y:.*]] = gpu.block_id  y
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y

    //CHECK: %[[r0:.*]] = arith.muli %[[x]], %[[c64]] : index
    //CHECK: %[[r1:.*]] = arith.muli %[[y]], %[[c128]] : index
    %m = arith.muli %block_id_x, %c64 : index
    %n = arith.muli %block_id_y, %c128 : index

    //CHECK: %[[r2:.*]] = xetile.init_tile %[[arg2]][%[[r0]], %[[r1]]] : memref<4096x4096xf32> -> !xetile.tile<64x128xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    //CHECK: %[[r3:.*]] = xetile.init_tile %[[arg2]][%[[r0]], %[[r1]]] : memref<4096x4096xf32> -> !xetile.tile<64x128xf32, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32> -> !xetile.tile<64x128xf32>

    //CHECK: %[[r4:.*]] = xetile.load_tile %[[r3]] : !xetile.tile<64x128xf32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<2x8x32x16xf32>
    //CHECK: %[[r5:.*]] = xetile.tile_unpack %[[r4]] {inner_blocks = array<i64: 32, 16>} : vector<2x8x32x16xf32> -> vector<64x128xf32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x128xf32> -> vector<64x128xf32>

    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg0]][%[[r0]], %[[c0]]] : memref<4096x4096xf16> -> !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<4096x4096xf16> -> !xetile.tile<64x128xf16>

    //CHECK: %[[r7:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[r1]]] : memref<4096x4096xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xf16> -> !xetile.tile<128x128xf16>

    //CHECK: %[[r8:.*]] = xetile.tile_pack %[[r5]] {inner_blocks = array<i64: 8, 16>} : vector<64x128xf32> -> vector<8x8x8x16xf32>
    //CHECK: %[[r9:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c4096]] step %[[c128]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[r6]], %[[arg5:.*]] = %[[r7]], %[[arg6:.*]] = %[[r8]])
    //CHECK-SAME: !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    //CHECK-SAME: !xetile.tile<128x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<8x8x8x16xf32>
    %out:3 = scf.for %k = %c0 to %c4096 step %c128
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<64x128xf16>, !xetile.tile<128x128xf16>, vector<64x128xf32>) {
      //CHECK: %[[r10:.*]] = xetile.load_tile %[[arg4]]
      //CHECK-SAME: !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<2x8x32x16xf16>
      //CHECK: %[[r11:.*]] = xetile.tile_unpack %[[r10]] {inner_blocks = array<i64: 32, 16>}  : vector<2x8x32x16xf16> -> vector<64x128xf16>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<64x128xf16> -> vector<64x128xf16>

      //CHECK: %[[r12:.*]] = xetile.load_tile %[[arg5]]
      //CHECK-SAME: !xetile.tile<128x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<4x8x32x16xf16>
      //CHECK: %[[r13:.*]] = xetile.tile_unpack %[[r12]] {inner_blocks = array<i64: 32, 16>}  : vector<4x8x32x16xf16> -> vector<128x128xf16>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<128x128xf16> -> vector<128x128xf16>

      //CHECK: %[[r14:.*]] = xetile.tile_pack %[[r11]] {inner_blocks = array<i64: 8, 16>}  : vector<64x128xf16> -> vector<8x8x8x16xf16>
      //CHECK: %[[r15:.*]] = xetile.tile_pack %[[r13]] {inner_blocks = array<i64: 16, 16>}  : vector<128x128xf16> -> vector<8x8x16x16xf16>
      //CHECK: %[[r16:.*]] = xetile.tile_mma %[[r14]], %[[r15]], %[[arg6]] : vector<8x8x8x16xf16>, vector<8x8x16x16xf16>, vector<8x8x8x16xf32> -> vector<8x8x8x16xf32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
        : vector<64x128xf16>, vector<128x128xf16>, vector<64x128xf32> -> vector<64x128xf32>

      //CHECK: %[[r17:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c128]]]
      //CHECK-SAME: !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      //CHECK: %[[r18:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c128]],  %[[c0]]]
      //CHECK-SAME: !xetile.tile<128x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c128] : !xetile.tile<64x128xf16> %b_next_tile = xetile.update_tile_offset %b_tile, [%c128, %c0]
        : !xetile.tile<128x128xf16>

      //CHECK: scf.yield %[[r17]], %[[r18]], %[[r16]]
      //CHECK-SAME: !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      //CHECK-SAME: !xetile.tile<128x128xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<8x8x8x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<64x128xf16>, !xetile.tile<128x128xf16>, vector<64x128xf32>
    }
    //CHECK: xetile.store_tile %[[r9]]#2,  %[[r2]] : vector<8x8x8x16xf32>, !xetile.tile<64x128xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_init_tile : vector<64x128xf32>, !xetile.tile<64x128xf32>

    //CHECK: gpu.return
    gpu.return
  }
}
