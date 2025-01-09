// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

#wg_map_a = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 128]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a>

#wg_map_b = #xetile.wg_map<sg_layout = [4, 4], sg_data = [128, 32]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b>

#wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 32]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c>


gpu.module @test_wg_to_sg  {
    //CHECK:  gpu.func @test_kernel(%[[arg0:.*]]:  memref<1024x1024xf16>, %[[arg1:.*]]:  memref<1024x1024xf16>, %[[arg2:.*]]: memref<1024x1024xf32>)
    gpu.func @test_kernel(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
        //CHECK: %[[c0:.*]] = arith.constant 0 : index
        //CHECK: %[[c128:.*]] = arith.constant 128 : index
        //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
        //CHECK: %[[R0:.*]] = gpu.block_id  x
        //CHECK: %[[R1:.*]] = gpu.block_id  y
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index

        //CHECK: %[[R2:.*]] = arith.muli %[[R0]], %[[c128]] : index
        //CHECK: %[[R3:.*]] = arith.muli %[[R1]], %[[c128]] : index
        //CHECK: %[[R4:.*]] = gpu.subgroup_id : index
        //CHECK: %[[c4:.*]] = arith.constant 4 : index
        //CHECK: %[[c32:.*]] = arith.constant 32 : index
        //CHECK: %[[R5:.*]] = index.divu %[[R4]], %[[c4]]
        //CHECK: %[[R6:.*]] = index.remu %[[R4]], %[[c4]]
        //CHECK: %[[R7:.*]] = index.add %[[R5]], %[[c0]]
        //CHECK: %[[R8:.*]] = index.remu %[[R7]], %[[c4]]
        //CHECK: %[[R9:.*]] = index.mul %[[R8]], %[[c32]]
        //CHECK: %[[R10:.*]] = index.add %[[R2]], %[[R9]]
        //CHECK: %[[R11:.*]] = index.add %[[R6]], %[[c0]]
        //CHECK: %[[R12:.*]] = index.remu %[[R11]], %[[c4]]
        //CHECK: %[[R13:.*]] = index.mul %[[R12]], %[[c32]]
        //CHECK: %[[R14:.*]] = index.add %[[R3]], %[[R13]]

        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y
        %m = arith.muli %block_id_x, %c128 : index
        %n = arith.muli %block_id_y, %c128 : index


        //CHECK: %[[R15:.*]] = xetile.init_tile %[[arg2]][%[[R10]], %[[R14]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
        //CHECK: %[[R16:.*]] =  xetile.load_tile %[[R15]]
        // intialize C tile and load it
        %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32>
          -> !xetile.tile<128x128xf32, #tile_attr_c>
        %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x128xf32, #tile_attr_c>
          -> vector<128x128xf32>


        //CHECK: %[[c1:.*]] = arith.constant 1 : index
        //CHECK: %[[R17:.*]] = index.remu %[[R11]], %[[c1]]
        //CHECK: %[[R18:.*]] = index.mul %[[R17]], %[[c128]]
        //CHECK: %[[R19:.*]] = index.add %[[R18]], %[[c0]]
        //CHECK: %[[R20:.*]] = xetile.init_tile %[[arg0]][%[[R10]], %[[R19]]] : memref<1024x1024xf16> -> !xetile.tile<32x128xf16>
        %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_a>

        //CHECK: %[[R21:.*]] = index.remu %[[R7]], %[[c1]]
        //CHECK: %[[R22:.*]] = index.mul %[[R21]], %[[c128]]
        //CHECK: %[[R23:.*]] = index.add %[[R22]], %[[c0]]
        //CHECK: %[[R24:.*]] = xetile.init_tile %[[arg1]][%[[R23]], %[[R14]]] : memref<1024x1024xf16> -> !xetile.tile<128x32xf16>
        %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_b>


        //CHECK: %[[R25:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c1024]] step %[[c128]]
        //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[R20]], %[[arg5:.*]] = %[[R24]], %[[arg6:.*]] = %[[R16]])
        //CHECK-SAME: -> (!xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>)

        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
        %out:3 = scf.for %k = %c0 to %c1024 step %c128
          iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
          -> (!xetile.tile<128x128xf16, #tile_attr_a>,
              !xetile.tile<128x128xf16, #tile_attr_b>,
              vector<128x128xf32>) {

          //CHECK: %[[R26:.*]] =  xetile.load_tile %[[arg4]] : !xetile.tile<32x128xf16> -> vector<32x128xf16>
          // load A and B tiles
          %a_value = xetile.load_tile %a_tile  : !xetile.tile<128x128xf16, #tile_attr_a>
            -> vector<128x128xf16>

          //CHECK: %[[R27:.*]] =  xetile.load_tile %[[arg5]] : !xetile.tile<128x32xf16> -> vector<128x32xf16>
          %b_value = xetile.load_tile %b_tile : !xetile.tile<128x128xf16, #tile_attr_b>
            -> vector<128x128xf16>

          //CHECK: %[[R28:.*]] = xetile.tile_mma %[[R26]], %[[R27]],  %[[arg6]] : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          // perform dpas and accumulate
          %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c}
            : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>

          //CHECK: %[[R29:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c128]]] : !xetile.tile<32x128xf16>
          // update the offsets for A and B tiles
          %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c128] : !xetile.tile<128x128xf16, #tile_attr_a>

          //CHECK: %[[R30:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c128]],  %[[c0]]] : !xetile.tile<128x32xf16>
          %b_next_tile = xetile.update_tile_offset %b_tile, [%c128, %c0] : !xetile.tile<128x128xf16, #tile_attr_b>

          //CHECK: scf.yield %[[R29]], %[[R30]], %[[R28]] : !xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>
          // partial C tile result
          scf.yield %a_next_tile, %b_next_tile, %c_new_value
            : !xetile.tile<128x128xf16, #tile_attr_a>,
            !xetile.tile<128x128xf16, #tile_attr_b>, vector<128x128xf32>
        }
        //CHECK: xetile.store_tile %[[R25]]#2,  %[[R15]] : vector<32x32xf32>, !xetile.tile<32x32xf32>
        // store the final accumulated C tile result back to memory
        xetile.store_tile %out#2, %c_init_tile : vector<128x128xf32>,
          !xetile.tile<128x128xf32, #tile_attr_c>
        //CHECK: gpu.return
        gpu.return
    }
}
