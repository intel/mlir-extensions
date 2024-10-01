// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

#wg_map_a = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a>

#wg_map_b = #xetile.wg_map<sg_layout = [2, 2], sg_data = [128, 32]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b>

#wg_map_c = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 32]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c>


gpu.module @test_wg_to_sg_rr  {
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
        //CHECK: %[[c2:.*]] = arith.constant 2 : index
        //CHECK: %[[c32:.*]] = arith.constant 32 : index
        //CHECK: %[[R5:.*]] = index.floordivs %[[R4]], %[[c2]]
        //CHECK: %[[R6:.*]] = index.remu %[[R4]], %[[c2]]
        //CHECK: %[[R7:.*]] = index.add %[[R5]], %[[c0]]
        //CHECK: %[[c4:.*]] = arith.constant 4 : index
        //CHECK: %[[R8:.*]] = index.remu %[[R7]], %[[c4]]
        //CHECK: %[[R9:.*]] = index.mul %[[R8]], %[[c32]]
        //CHECK: %[[R10:.*]] = index.add %[[R2]], %[[R9]]
        //CHECK: %[[R11:.*]] = index.add %[[R5]], %[[c2]]
        //CHECK: %[[R12:.*]] = index.remu %[[R11]], %[[c4]]
        //CHECK: %[[R13:.*]] = index.mul %[[R12]], %[[c32]]
        //CHECK: %[[R14:.*]] = index.add %[[R2]], %[[R13]]
        //CHECK: %[[R15:.*]] = index.add %[[R6]], %[[c0]]
        //CHECK: %[[R16:.*]] = index.remu %[[R15]], %[[c4]]
        //CHECK: %[[R17:.*]] = index.mul %[[R16]], %[[c32]]
        //CHECK: %[[R18:.*]] = index.add %[[R3]], %[[R17]]
        //CHECK: %[[R19:.*]] = index.add %[[R6]], %[[c2]]
        //CHECK: %[[R20:.*]] = index.remu %[[R19]], %[[c4]]
        //CHECK: %[[R21:.*]] = index.mul %[[R20]], %[[c32]]
        //CHECK: %[[R22:.*]] = index.add %[[R3]], %[[R21]]

        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y
        %m = arith.muli %block_id_x, %c128 : index
        %n = arith.muli %block_id_y, %c128 : index


        //CHECK: %[[R23:.*]] = xetile.init_tile %[[arg2]][%[[R10]], %[[R18]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
        //CHECK: %[[R24:.*]] = xetile.init_tile %[[arg2]][%[[R10]], %[[R22]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
        //CHECK: %[[R25:.*]] = xetile.init_tile %[[arg2]][%[[R14]], %[[R18]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
        //CHECK: %[[R26:.*]] = xetile.init_tile %[[arg2]][%[[R14]], %[[R22]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
        %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32>
          -> !xetile.tile<128x128xf32, #tile_attr_c>

        //CHECK: %[[R27:.*]] =  xetile.load_tile %[[R23]] : !xetile.tile<32x32xf32> -> vector<32x32xf32>
        //CHECK: %[[R28:.*]] =  xetile.load_tile %[[R24]] : !xetile.tile<32x32xf32> -> vector<32x32xf32>
        //CHECK: %[[R29:.*]] =  xetile.load_tile %[[R25]] : !xetile.tile<32x32xf32> -> vector<32x32xf32>
        //CHECK: %[[R30:.*]] =  xetile.load_tile %[[R26]] : !xetile.tile<32x32xf32> -> vector<32x32xf32>
        %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x128xf32, #tile_attr_c>
          -> vector<128x128xf32>


        //CHECK: %[[c1:.*]] = arith.constant 1 : index
        //CHECK: %[[R31:.*]] = index.remu %[[R15]], %[[c1]]
        //CHECK: %[[R32:.*]] = index.mul %[[R31]], %[[c128]]
        //CHECK: %[[R33:.*]] = index.add %[[R32]], %[[c0]]
        //CHECK: %[[R34:.*]] = xetile.init_tile %[[arg0]][%[[R10]], %[[R33]]] : memref<1024x1024xf16> -> !xetile.tile<32x128xf16>
        //CHECK: %[[R35:.*]] = xetile.init_tile %[[arg0]][%[[R14]], %[[R33]]] : memref<1024x1024xf16> -> !xetile.tile<32x128xf16>
        %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_a>

        //CHECK: %[[R36:.*]] = index.remu %[[R7]], %[[c1]]
        //CHECK: %[[R37:.*]] = index.mul %[[R36]], %[[c128]]
        //CHECK: %[[R38:.*]] = index.add %[[R37]], %[[c0]]
        //CHECK: %[[R39:.*]] = xetile.init_tile %[[arg1]][%[[R38]], %[[R18]]] : memref<1024x1024xf16> -> !xetile.tile<128x32xf16>
        //CHECK: %[[R40:.*]] = xetile.init_tile %[[arg1]][%[[R38]], %[[R22]]] : memref<1024x1024xf16> -> !xetile.tile<128x32xf16>
        %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_b>


        //CHECK: %[[R41:.*]]:8 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c1024]] step %[[c128]]
        //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[R34]], %[[arg5:.*]] = %[[R35]], %[[arg6:.*]] = %[[R39]],  %[[arg7:.*]] = %[[R40]], %[[arg8:.*]] = %[[R27]], %[[arg9:.*]] = %[[R28]], %[[arg10:.*]] = %[[R29]], %[[arg11:.*]] = %[[R30]])
        //CHECK-SAME: -> (!xetile.tile<32x128xf16>, !xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>)
        %out:3 = scf.for %k = %c0 to %c1024 step %c128
          iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
          -> (!xetile.tile<128x128xf16, #tile_attr_a>,
              !xetile.tile<128x128xf16, #tile_attr_b>,
              vector<128x128xf32>) {

          //CHECK: %[[R42:.*]] =  xetile.load_tile %[[arg4]] : !xetile.tile<32x128xf16> -> vector<32x128xf16>
          //CHECK: %[[R43:.*]] =  xetile.load_tile %[[arg5]] : !xetile.tile<32x128xf16> -> vector<32x128xf16>
          %a_value = xetile.load_tile %a_tile  : !xetile.tile<128x128xf16, #tile_attr_a>
            -> vector<128x128xf16>

          //CHECK: %[[R44:.*]] =  xetile.load_tile %[[arg6]] : !xetile.tile<128x32xf16> -> vector<128x32xf16>
          //CHECK: %[[R45:.*]] =  xetile.load_tile %[[arg7]] : !xetile.tile<128x32xf16> -> vector<128x32xf16>
          %b_value = xetile.load_tile %b_tile : !xetile.tile<128x128xf16, #tile_attr_b>
            -> vector<128x128xf16>

          //CHECK: %[[R46:.*]] = xetile.tile_mma %[[R42]], %[[R44]],  %[[arg8]] : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          //CHECK: %[[R47:.*]] = xetile.tile_mma %[[R42]], %[[R45]],  %[[arg9]] : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          //CHECK: %[[R48:.*]] = xetile.tile_mma %[[R43]], %[[R44]],  %[[arg10]] : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          //CHECK: %[[R49:.*]] = xetile.tile_mma %[[R43]], %[[R45]],  %[[arg11]] : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c}
            : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>

          //CHECK: %[[R50:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c128]]] : !xetile.tile<32x128xf16>, index, index -> !xetile.tile<32x128xf16>
          //CHECK: %[[R51:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c0]],  %[[c128]]] : !xetile.tile<32x128xf16>, index, index -> !xetile.tile<32x128xf16>
          %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c128]
            : !xetile.tile<128x128xf16, #tile_attr_a>, index, index
            -> !xetile.tile<128x128xf16, #tile_attr_a>

          //CHECK: %[[R52:.*]] = xetile.update_tile_offset %[[arg6]], [%[[c128]],  %[[c0]]] : !xetile.tile<128x32xf16>, index, index -> !xetile.tile<128x32xf16>
          //CHECK: %[[R53:.*]] = xetile.update_tile_offset %[[arg7]], [%[[c128]],  %[[c0]]] : !xetile.tile<128x32xf16>, index, index -> !xetile.tile<128x32xf16>
          %b_next_tile = xetile.update_tile_offset %b_tile, [%c128, %c0]
            : !xetile.tile<128x128xf16, #tile_attr_b>, index, index
            -> !xetile.tile<128x128xf16, #tile_attr_b>

          //CHECK: scf.yield %[[R50]], %[[R51]], %[[R52]], %[[R53]], %[[R46]], %[[R47]], %[[R48]], %[[R49]]
          //CHECK-SAME: !xetile.tile<32x128xf16>, !xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>, vector<32x32xf32>
          scf.yield %a_next_tile, %b_next_tile, %c_new_value
            : !xetile.tile<128x128xf16, #tile_attr_a>,
            !xetile.tile<128x128xf16, #tile_attr_b>, vector<128x128xf32>
        }

        //CHECK: xetile.store_tile %[[R41]]#4,  %[[R23]] : vector<32x32xf32>, !xetile.tile<32x32xf32>
        //CHECK: xetile.store_tile %[[R41]]#5,  %[[R24]] : vector<32x32xf32>, !xetile.tile<32x32xf32>
        //CHECK: xetile.store_tile %[[R41]]#6,  %[[R25]] : vector<32x32xf32>, !xetile.tile<32x32xf32>
        //CHECK: xetile.store_tile %[[R41]]#7,  %[[R26]] : vector<32x32xf32>, !xetile.tile<32x32xf32>
        xetile.store_tile %out#2, %c_init_tile : vector<128x128xf32>,
          !xetile.tile<128x128xf32, #tile_attr_c>
        %cst = arith.constant {map = #wg_map_c} dense<0.000000e+00> : vector<128x128xf32>
        %result_post_op = arith.addf %out#2, %cst {map = #wg_map_c} : vector<128x128xf32>
        //CHECK: gpu.return
        gpu.return
    }
  }
