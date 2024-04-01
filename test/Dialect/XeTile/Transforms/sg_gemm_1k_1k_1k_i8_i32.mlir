// RUN: imex-opt --xetile-init-duplicate --xetile-blocking %s | FileCheck %s

// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {

  //CHECK: gpu.func @test_gemm(%[[arg0:.*]]: memref<1024x1024xi8>, %[[arg1:.*]]: memref<1024x1024xi8>, %[[arg2:.*]]: memref<1024x1024xi32>)








  gpu.func @test_gemm(%A: memref<1024x1024xi8>, %B: memref<1024x1024xi8>, %C: memref<1024x1024xi32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[R0:.*]] = gpu.block_id  x
    //CHECK: %[[R1:.*]] = gpu.block_id  y
    //CHECK: %[[R2:.*]] = arith.muli %[[R0]], %[[c64]] : index
    //CHECK: %[[R3:.*]] = arith.muli %[[R1]], %[[c64]] : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c64 : index
    %n = arith.muli %block_id_y, %c64 : index

    //CHECK: %[[R4:.*]] = xetile.init_tile %[[arg2]][%[[R2]], %[[R3]]]
    //CHECK-SAME: memref<1024x1024xi32> -> !xetile.tile<64x64xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    //CHECK: %[[R5:.*]] = xetile.init_tile %[[arg2]][%[[R2]], %[[R3]]]
    //CHECK-SAME: memref<1024x1024xi32> -> !xetile.tile<64x64xi32, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<64x64xi32>

    //CHECK: %[[R6:.*]] = xetile.load_tile %[[R5]] { padding = 0 : i32 }
    //CHECK-SAME: !xetile.tile<64x64xi32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<2x4x32x16xi32>
    //CHECK: %[[R7:.*]] = xetile.tile_unpack %[[R6]] { inner_blocks = [32, 16] }  : vector<2x4x32x16xi32> -> vector<64x64xi32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x64xi32> -> vector<64x64xi32>

    //CHECK: %[[R8:.*]] = xetile.init_tile %[[arg0]][%[[R2]], %[[c0]]]
    //CHECK-SAME: memref<1024x1024xi8> -> !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 64]>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xi8> -> !xetile.tile<64x64xi8>

    //CHECK: %[[R9:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[R3]]]
    //CHECK-SAME: memref<1024x1024xi8> -> !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xi8> -> !xetile.tile<64x64xi8>

    //CHECK: %[[R10:.*]] = xetile.tile_pack %[[R7]] { inner_blocks = [32, 16] }  : vector<64x64xi32> -> vector<2x4x32x16xi32>
    //CHECK: %[[R11:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c1024]] step %[[c64]] iter_args(%[[arg4:.*]] = %[[R8]], %[[arg5:.*]] = %[[R9]], %[[arg6:.*]] = %[[R10]])
    //CHECK-SAME: !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 64]>>, !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<2x4x32x16xi32>
    %out:3 = scf.for %k = %c0 to %c1024 step %c64 iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
                  -> (!xetile.tile<64x64xi8>, !xetile.tile<64x64xi8>, vector<64x64xi32>) {
      //CHECK: %[[R14:.*]] = xetile.load_tile %[[arg4]] { padding = 0 : i32 }  : !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 64]>> -> vector<2x1x32x64xi8>
      //CHECK: %[[R15:.*]] = xetile.tile_unpack %[[R14]] { inner_blocks = [32, 64] }  : vector<2x1x32x64xi8> -> vector<64x64xi8>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<64x64xi8> -> vector<64x64xi8>

      //CHECK: %[[R16:.*]] = xetile.load_tile %[[arg5]] { padding = 0 : i32 }  : !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<2x4x32x16xi8>
      //CHECK: %[[R17:.*]] = xetile.tile_unpack %[[R16]] { inner_blocks = [32, 16] }  : vector<2x4x32x16xi8> -> vector<64x64xi8>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<64x64xi8> -> vector<64x64xi8>

      //CHECK: %[[R18:.*]] = xetile.tile_pack %[[R15]] { inner_blocks = [8, 8] }  : vector<64x64xi8> -> vector<8x8x8x8xi8>
      //CHECK: %[[R19:.*]] = xetile.tile_pack %[[R17]] { inner_blocks = [8, 8] }  : vector<64x64xi8> -> vector<8x8x8x8xi8>
      //CHECK: %[[R20:.*]] = xetile.tile_unpack %[[arg6]] { inner_blocks = [32, 16] }  : vector<2x4x32x16xi32> -> vector<64x64xi32>
      //CHECK: %[[R21:.*]] = xetile.tile_pack %[[R20]] { inner_blocks = [8, 8] }  : vector<64x64xi32> -> vector<8x8x8x8xi32>
      //CHECK: %[[R22:.*]] = xetile.tile_mma %[[R18]], %[[R19]], %[[R21]] : vector<8x8x8x8xi8>, vector<8x8x8x8xi8>, vector<8x8x8x8xi32> -> vector<8x8x8x8xi32>
      //CHECK: %[[R23:.*]] = xetile.tile_unpack %[[R22]] { inner_blocks = [8, 8] }  : vector<8x8x8x8xi32> -> vector<64x64xi32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<64x64xi8>, vector<64x64xi8>, vector<64x64xi32> -> vector<64x64xi32>

      //CHECK: %[[R24:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c64]]]
      //CHECK-SAME: !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 64]>>, index, index
      //CHECK-SAME: !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 64]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64] : !xetile.tile<64x64xi8>, index, index -> !xetile.tile<64x64xi8>

      //CHECK: %[[R25:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c64]],  %[[c0]]]
      //CHECK-SAME: !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index
      //CHECK-SAME: !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0] : !xetile.tile<64x64xi8>, index, index -> !xetile.tile<64x64xi8>

      //CHECK: %[[R26:.*]] = xetile.tile_pack %[[R23]] { inner_blocks = [32, 16] }  : vector<64x64xi32> -> vector<2x4x32x16xi32>
      //CHECK: scf.yield %[[R24]], %[[R25]], %[[R26]]
      //CHECK-SAME: !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 64]>>
      //CHECK-SAME: !xetile.tile<64x64xi8, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<2x4x32x16xi32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<64x64xi8>, !xetile.tile<64x64xi8>, vector<64x64xi32>
    }
    //CHECK: %[[R12:.*]] = xetile.tile_unpack %[[R11]]#2 { inner_blocks = [32, 16] }  : vector<2x4x32x16xi32> -> vector<64x64xi32>

    //CHECK: %[[R13:.*]] = xetile.tile_pack %[[R12]] { inner_blocks = [8, 16] }  : vector<64x64xi32> -> vector<8x4x8x16xi32>
    //CHECK: xetile.store_tile %[[R13]],  %[[R4]] : vector<8x4x8x16xi32>, !xetile.tile<64x64xi32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_init_tile {innner_blocks = [8, 16]}: vector<64x64xi32>, !xetile.tile<64x64xi32>

    //CHECK: gpu.return
    gpu.return
  }
}
