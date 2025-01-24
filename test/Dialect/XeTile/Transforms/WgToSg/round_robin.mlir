// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

#wg_map_a = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 128]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a>

#wg_map_b = #xetile.wg_map<sg_layout = [4, 4], sg_data = [128, 32]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b>

#wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 32]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c>


gpu.module @test_wg_to_sg_rr  {
   //CHECK:  gpu.func @test_kernel(%[[arg0:.*]]:  memref<1024x1024xf16>, %[[arg1:.*]]:  memref<1024x1024xf16>, %[[arg2:.*]]: memref<1024x1024xf32>)
    gpu.func @test_kernel(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
        //CHECK: %[[c0:.*]] = arith.constant 0 : index
        //CHECK: %[[c128:.*]] = arith.constant 128 : index
        //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
        //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
        //CHECK: %[[block_id_y:.*]] = gpu.block_id  y
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index

        //CHECK: %[[R0:.*]] = arith.muli %[[block_id_x]], %[[c128]] : index
        //CHECK: %[[R1:.*]] = arith.muli %[[block_id_y]], %[[c128]] : index
        //CHECK: %[[R2:.*]] = gpu.subgroup_id : index
        //CHECK: %[[c4:.*]] = arith.constant 4 : index
        //CHECK: %[[c32:.*]] = arith.constant 32 : index
        //CHECK: %[[R3:.*]] = index.divu %[[R2]], %[[c4]]
        //CHECK: %[[R4:.*]] = index.remu %[[R2]], %[[c4]]
        //CHECK: %[[R5:.*]] = index.add %[[R3]], %[[c0]]
        //CHECK: %[[R6:.*]] = index.remu %[[R5]], %[[c4]]
        //CHECK: %[[R7:.*]] = index.mul %[[R6]], %[[c32]]
        //CHECK: %[[R8:.*]] = index.add %[[R0]], %[[R7]]
        //CHECK: %[[R9:.*]] = index.add %[[R4]], %[[c0]]
        //CHECK: %[[R10:.*]] = index.remu %[[R9]], %[[c4]]
        //CHECK: %[[R11:.*]] = index.mul %[[R10]], %[[c32]]
        //CHECK: %[[R12:.*]] = index.add %[[R1]], %[[R11]]

        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y
        %m = arith.muli %block_id_x, %c128 : index
        %n = arith.muli %block_id_y, %c128 : index


        //CHECK: %[[R13:.*]] = xetile.init_tile %[[arg2]][%[[R8]], %[[R12]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
        %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32>
          -> !xetile.tile<128x128xf32, #tile_attr_c>

        //CHECK: %[[R14:.*]] = xetile.load_tile %[[R13]] : !xetile.tile<32x32xf32> -> vector<32x32xf32>
        %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x128xf32, #tile_attr_c>
          -> vector<128x128xf32>


        //CHECK: %[[c1:.*]] = arith.constant 1 : index
        //CHECK: %[[R15:.*]] = index.remu %[[R9]], %[[c1]]
        //CHECK: %[[R16:.*]] = index.mul %[[R15]], %[[c128]]
        //CHECK: %[[R17:.*]] = index.add %[[R16]], %[[c0]]
        //CHECK: %[[R18:.*]] = xetile.init_tile %[[arg0]][%[[R8]], %[[R17]]] : memref<1024x1024xf16> -> !xetile.tile<32x128xf16>
        %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_a>

        //CHECK: %[[R19:.*]] = index.remu %[[R5:.*]], %[[c1]]
        //CHECK: %[[R20:.*]] = index.mul %[[R19:.*]], %[[c128:.*]]
        //CHECK: %[[R21:.*]] = index.add %[[R20:.*]], %[[c0]]
        //CHECK: %[[R22:.*]] = xetile.init_tile %[[arg1]][%[[R21]], %[[R12]]] : memref<1024x1024xf16> -> !xetile.tile<128x32xf16>
        %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_b>

        //CHECK: %[[R23:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c1024]] step %[[c128]]
        //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[R18]], %[[arg5:.*]] = %[[R22]], %[[arg6:.*]] = %[[R14]])
        //CHECK-SAME: -> (!xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>) {
        %out:3 = scf.for %k = %c0 to %c1024 step %c128
          iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
          -> (!xetile.tile<128x128xf16, #tile_attr_a>,
              !xetile.tile<128x128xf16, #tile_attr_b>,
              vector<128x128xf32>) {

          //CHECK: %[[R24:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<32x128xf16> -> vector<32x128xf16>
          //CHECK: %[[R25:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<128x32xf16> -> vector<128x32xf16>
          //CHECK: %[[R26:.*]] = xetile.tile_mma %[[R24]], %[[R25]], %[[arg6]] : vector<32x128xf16>, vector<128x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          //CHECK: %[[R27:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]], %[[c128]]] : !xetile.tile<32x128xf16>
          //CHECK: %[[R28:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c128]], %[[c0]]] : !xetile.tile<128x32xf16>
          //CHECK: scf.yield %[[R27]], %[[R28]], %[[R26]] : !xetile.tile<32x128xf16>, !xetile.tile<128x32xf16>, vector<32x32xf32>

          %a_value = xetile.load_tile %a_tile  : !xetile.tile<128x128xf16, #tile_attr_a> -> vector<128x128xf16>
          %b_value = xetile.load_tile %b_tile : !xetile.tile<128x128xf16, #tile_attr_b> -> vector<128x128xf16>
          %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c}
            : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>
          %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c128] : !xetile.tile<128x128xf16, #tile_attr_a>
          %b_next_tile = xetile.update_tile_offset %b_tile, [%c128, %c0] : !xetile.tile<128x128xf16, #tile_attr_b>
          scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<128x128xf16, #tile_attr_a>, !xetile.tile<128x128xf16, #tile_attr_b>, vector<128x128xf32>
        }

        //CHECK: xetile.store_tile %[[R23]]#2,  %[[R13]] : vector<32x32xf32>, !xetile.tile<32x32xf32>
        xetile.store_tile %out#2, %c_init_tile : vector<128x128xf32>, !xetile.tile<128x128xf32, #tile_attr_c>
        %cst = arith.constant {map = #wg_map_c} dense<0.000000e+00> : vector<128x128xf32>
        %result_post_op = arith.addf %out#2, %cst {map = #wg_map_c} : vector<128x128xf32>
        //CHECK: gpu.return
        gpu.return
    }
  }
