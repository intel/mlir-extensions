// RUN: imex-opt --xetile-init-duplicate --xetile-blocking %s | FileCheck %s

// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {
  // CHECK: gpu.func @test_gemm(%[[A:.*]]: memref<1024x1024xf16>, %[[B:.*]]: memref<1024x1024xf16>, %[[C:.*]]: memref<1024x1024xf32>)
  gpu.func @test_gemm(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {

    //CHECK: %[[C0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    //CHECK: %[[C1:.*]] = arith.constant 64 : index
    %c64 = arith.constant 64 : index

    //CHECK: %[[C2:.*]] = arith.constant 1024 : index
    %c1024 = arith.constant 1024 : index

    //CHECK: %[[R0:.*]] = gpu.block_id  x
    %block_id_x = gpu.block_id x

    //CHECK: %[[R1:.*]] = gpu.block_id  y
    %block_id_y = gpu.block_id y

    //CHECK: %[[R2:.*]] = arith.muli %[[R0]], %[[C1]] : index
    %m = arith.muli %block_id_x, %c64 : index

    //CHECK: %[[R3:.*]] = arith.muli %[[R1]], %[[C1]] : index
    %n = arith.muli %block_id_y, %c64 : index

    // intialize C tile and load it. It is used by both load and store
    // a duplicated copy will be created by xetile-init-duplicate pass. one for
    // load and one for store.

    //CHECK: %[[R4:.*]] = xetile.init_tile %[[C]][%[[R2]], %[[R3]]]
    //CHECK-SAME: memref<1024x1024xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    //CHECK: %[[R5:.*]] = xetile.init_tile %[[C]][%[[R2]], %[[R3]]]
    //CHECK-SAME: memref<1024x1024xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<64x64xf32>

    //CHECK: %[[R6:.*]] = xetile.load_tile %[[R5]] { padding = 0.000000e+00 : f32 }
    //CHECK-SAME: !xetile.tile<64x64xf32, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<2x4x32x16xf32>
    //CHECK: %[[R7:.*]] = xetile.tile_unpack %[[R6]] { inner_blocks = [32, 16] }  : vector<2x4x32x16xf32> -> vector<64x64xf32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x64xf32> -> vector<64x64xf32>

    // initalize A and B tiles

    //CHECK: %[[R8:.*]] = xetile.init_tile %[[A]][%[[R2]], %[[C0]]]
    //CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>

    //CHECK: %[[R9:.*]] = xetile.init_tile %[[B]][%[[C0]], %[[R3]]]
    //CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>

    // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
    //CHECK: %[[R10:.*]] = xetile.tile_pack %[[R7]] { inner_blocks = [32, 16] }  : vector<64x64xf32> -> vector<2x4x32x16xf32>
    //CHECK: %[[R11:.*]]:3 = scf.for %[[arg3:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[R8]], %[[arg5:.*]] = %[[R9]], %[[arg6:.*]] = %[[R10]])
    //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
    //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<2x4x32x16xf32>
    %out:3 = scf.for %k = %c0 to %c1024 step %c64
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>) {

      //CHECK: %[[R14:.*]] = xetile.load_tile %[[arg4]] { padding = 0.000000e+00 : f32 }
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<2x2x32x32xf16>
      //CHECK: %[[R15:.*]] = xetile.tile_unpack %[[R14]] { inner_blocks = [32, 32] }  : vector<2x2x32x32xf16> -> vector<64x64xf16>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>

      //CHECK: %[[R16:.*]] = xetile.load_tile %[[arg5]] { padding = 0.000000e+00 : f32 }
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<2x4x32x16xf16>
      //CHECK: %[[R17:.*]] = xetile.tile_unpack %[[R16]] { inner_blocks = [32, 16] }  : vector<2x4x32x16xf16> -> vector<64x64xf16>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>

      // perform dpas and accumulate
      //CHECK: %[[R18:.*]] = xetile.tile_pack %[[R15]] { inner_blocks = [8, 16] }  : vector<64x64xf16> -> vector<8x4x8x16xf16>
      //CHECK: %[[R19:.*]] = xetile.tile_pack %[[R17]] { inner_blocks = [16, 16] }  : vector<64x64xf16> -> vector<4x4x16x16xf16>
      //CHECK: %[[R20:.*]] = xetile.tile_unpack %[[arg6]] { inner_blocks = [32, 16] }  : vector<2x4x32x16xf32> -> vector<64x64xf32>
      //CHECK: %[[R21:.*]] = xetile.tile_pack %[[R20]] { inner_blocks = [8, 16] }  : vector<64x64xf32> -> vector<8x4x8x16xf32>
      //CHECK: %[[R22:.*]] = xetile.tile_mma %[[R18]], %[[R19]], %[[R21]] : vector<8x4x8x16xf16>, vector<4x4x16x16xf16>, vector<8x4x8x16xf32> -> vector<8x4x8x16xf32>
      //CHECK: %[[R23:.*]] = xetile.tile_unpack %[[R22]] { inner_blocks = [8, 16] }  : vector<8x4x8x16xf32> -> vector<64x64xf32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf32> -> vector<64x64xf32>

      // update the offsets for A and B tiles
      //CHECK: %[[R24:.*]] = xetile.update_tile_offset %[[arg4]], [%[[C0]],  %[[C1]]]
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, index, index
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64] : !xetile.tile<64x64xf16>, index, index -> !xetile.tile<64x64xf16>

      //CHECK: %[[R25:.*]] = xetile.update_tile_offset %[[arg5]], [%[[C1]],  %[[C0]]]
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0] : !xetile.tile<64x64xf16>, index, index -> !xetile.tile<64x64xf16>

      //CHECK: %[[R26:.*]] = xetile.tile_pack %[[R23]] { inner_blocks = [32, 16] }  : vector<64x64xf32> -> vector<2x4x32x16xf32>
      //CHECK: scf.yield %[[R24]], %[[R25]], %[[R26]]
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      //CHECK-SAME: !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<2x4x32x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>
    }
    //CHECK: %[[REG12:.*]] = xetile.tile_unpack %[[R11]]#2 { inner_blocks = [32, 16] }  : vector<2x4x32x16xf32> -> vector<64x64xf32>

    // store the final accumulated C tile result back to memory
    //CHECK: %[[REG13:.*]] = xetile.tile_pack %[[REG12]] { inner_blocks = [8, 16] }  : vector<64x64xf32> -> vector<8x4x8x16xf32>
    //CHECK: xetile.store_tile %[[REG13]],  %[[R4]] : vector<8x4x8x16xf32>, !xetile.tile<64x64xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_init_tile: vector<64x64xf32>, !xetile.tile<64x64xf32>

    //CHECK: gpu.return
    gpu.return
  }
}
