// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

#wg_map_a = #xetile.wg_map<sg_layout = [4, 4], sg_data = [64, 256]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a>

#wg_map_b = #xetile.wg_map<sg_layout = [4, 4], sg_data = [256, 64]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b>

#wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [64, 64]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c>

gpu.module @test_wg_to_sg_4k {
    //CHECK:  gpu.func @test_kernel(%[[arg0:.*]]:   memref<4096x4096xf16>, %[[arg1:.*]]:  memref<4096x4096xf16>, %[[arg2:.*]]: memref<4096x4096xf32>)
    gpu.func @test_kernel(%A: memref<4096x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<4096x4096xf32>) {
        //CHECK: %[[c0:.*]] = arith.constant 0 : index
        //CHECK: %[[c256:.*]] = arith.constant 256 : index
        //CHECK: %[[c4096:.*]] = arith.constant 4096 : index
        //CHECK: %[[R0:.*]] = gpu.block_id  x
        //CHECK: %[[R1:.*]] = gpu.block_id  y
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c256 = arith.constant 256 : index
        %c4096 = arith.constant 4096 : index

        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y

        //CHECK: %[[R2:.*]] = arith.muli %[[R0]], %[[c256]] : index
        //CHECK: %[[R3:.*]] = arith.muli %[[R1]], %[[c256]] : index
        //CHECK: %[[R4:.*]] = gpu.subgroup_id : index
        //CHECK: %[[c4:.*]] = arith.constant 4 : index
        //CHECK: %[[c64:.*]] = arith.constant 64 : index
        //CHECK: %[[R5:.*]] = index.floordivs %[[R4]], %[[c4]]
        //CHECK: %[[R6:.*]] = index.remu %[[R4]], %[[c4]]
        //CHECK: %[[R7:.*]] = index.add %[[R5]], %[[c0]]
        //CHECK: %[[R8:.*]] = index.remu %[[R7]], %[[c4]]
        //CHECK: %[[R9:.*]] = index.mul %[[R8]], %[[c64]]
        //CHECK: %[[R10:.*]] = index.add %[[R2]], %[[R9]]
        //CHECK: %[[R11:.*]] = index.add %[[R6]], %[[c0]]
        //CHECK: %[[R12:.*]] = index.remu %[[R11]], %[[c4]]
        //CHECK: %[[R13:.*]] = index.mul %[[R12]], %[[c64]]
        //CHECK: %[[R14:.*]] = index.add %[[R3]], %[[R13]]

        %m = arith.muli %block_id_x, %c256 : index
        %n = arith.muli %block_id_y, %c256 : index

        //CHECK: %[[R15:.*]] = xetile.init_tile %[[arg2]][%[[R10]], %[[R14]]] : memref<4096x4096xf32> -> !xetile.tile<64x64xf32>
        //CHECK: %[[R16:.*]] =  xetile.load_tile %[[R15]]
        // intialize C tile and load it
        %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32>
          -> !xetile.tile<256x256xf32, #tile_attr_c>
        %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<256x256xf32, #tile_attr_c>
          -> vector<256x256xf32>

        //CHECK: %[[c1:.*]] = arith.constant 1 : index
        //CHECK: %[[R17:.*]] = index.remu %[[R11]], %[[c1]]
        //CHECK: %[[R18:.*]] = index.mul %[[R17]], %[[c256]]
        //CHECK: %[[R19:.*]] = index.add %[[R18]], %[[c0]]
        //CHECK: %[[R20:.*]] = xetile.init_tile %[[arg0]][%[[R10]], %[[R19]]] : memref<4096x4096xf16> -> !xetile.tile<64x256xf16>
        // initalize A and B tiles
        %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<4096x4096xf16>
          -> !xetile.tile<256x256xf16, #tile_attr_a>

        //CHECK: %[[R21:.*]] = index.remu %[[R7]], %[[c1]]
        //CHECK: %[[R22:.*]] = index.mul %[[R21]], %[[c256]]
        //CHECK: %[[R23:.*]] = index.add %[[R22]], %[[c0]]
        //CHECK: %[[R24:.*]] = xetile.init_tile %[[arg1]][%[[R23]], %[[R14]]] : memref<4096x4096xf16> -> !xetile.tile<256x64xf16>
        %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xf16>
          -> !xetile.tile<256x256xf16, #tile_attr_b>

        //CHECK: %[[R25:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c4096]] step %[[c256]]
        //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[R20]], %[[arg5:.*]] = %[[R24]], %[[arg6:.*]] = %[[R16]])
        //CHECK-SAME: -> (!xetile.tile<64x256xf16>, !xetile.tile<256x64xf16>, vector<64x64xf32>)
        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
        %out:3 = scf.for %k = %c0 to %c4096 step %c256
          iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
          -> (!xetile.tile<256x256xf16, #tile_attr_a>,
              !xetile.tile<256x256xf16, #tile_attr_b>,
              vector<256x256xf32>) {

           //CHECK: %[[R26:.*]] =  xetile.load_tile %[[arg4]] : !xetile.tile<64x256xf16> -> vector<64x256xf16>
          // load A and B tiles
          %a_value = xetile.load_tile %a_tile  : !xetile.tile<256x256xf16, #tile_attr_a>
            -> vector<256x256xf16>

          //CHECK: %[[R27:.*]] =  xetile.load_tile %[[arg5]] : !xetile.tile<256x64xf16> -> vector<256x64xf16>
          %b_value = xetile.load_tile %b_tile : !xetile.tile<256x256xf16, #tile_attr_b>
            -> vector<256x256xf16>

          //CHECK: %[[R28:.*]] = xetile.tile_mma %[[R26]], %[[R27]],  %[[arg6]] : vector<64x256xf16>, vector<256x64xf16>, vector<64x64xf32> -> vector<64x64xf32>
          // perform dpas and accumulate
          %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c}
            : vector<256x256xf16>, vector<256x256xf16>, vector<256x256xf32> -> vector<256x256xf32>

          //CHECK: %[[R29:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c256]]] : !xetile.tile<64x256xf16>
          // update the offsets for A and B tiles
          %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c256] : !xetile.tile<256x256xf16, #tile_attr_a>

          //CHECK: %[[R30:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c256]],  %[[c0]]] : !xetile.tile<256x64xf16>
          %b_next_tile = xetile.update_tile_offset %b_tile, [%c256, %c0] : !xetile.tile<256x256xf16, #tile_attr_b>

          //CHECK: scf.yield %[[R29]], %[[R30]], %[[R28]] : !xetile.tile<64x256xf16>, !xetile.tile<256x64xf16>, vector<64x64xf32>
          // partial C tile result
          scf.yield %a_next_tile, %b_next_tile, %c_new_value
            : !xetile.tile<256x256xf16, #tile_attr_a>,
            !xetile.tile<256x256xf16, #tile_attr_b>, vector<256x256xf32>
        }
        //CHECK: xetile.store_tile %[[R25]]#2,  %[[R15]] : vector<64x64xf32>, !xetile.tile<64x64xf32>
        // store the final accumulated C tile result back to memory
        xetile.store_tile %out#2, %c_init_tile : vector<256x256xf32>,
          !xetile.tile<256x256xf32, #tile_attr_c>
        //CHECK: gpu.return
        gpu.return
    }
  }
