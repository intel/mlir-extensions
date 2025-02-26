// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

gpu.module @test_convert_layout{
  //CHECK:  gpu.func @test_kernel()
  gpu.func @test_kernel() {
    //CHECK: %[[c0:.*]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
    //CHECK: %[[c0_0:.*]] = arith.constant dense<0.000000e+00> : vector<8x256xf32>
    //CHECK: %[[SLMALLOC:.*]] = memref.alloc() : memref<262144xi8, 3>
    //CHECK: %[[cst_0:.*]] = arith.constant 0 : index
    //CHECK: %[[SLMVIEW:.*]] = memref.view %[[SLMALLOC]][%[[cst_0]]][] : memref<262144xi8, 3> to memref<256x256xf32, 3>
    //CHECK: %[[R0:.*]] = gpu.subgroup_id : index
    //CHECK: %[[c4:.*]] = arith.constant 4 : index
    //CHECK: %[[R1:.*]] = index.divu %[[R0]], %[[c4]]
    //CHECK: %[[R2:.*]] = index.remu %[[R0]], %[[c4]]
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[R3:.*]] = index.mul %[[R1]], %[[c32]]
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    //CHECK: %[[R4:.*]] = index.mul %[[R2]], %[[c64]]
    //CHECK: %[[INITTILESRCMAP:.*]] = xetile.init_tile %[[SLMVIEW]][%[[R3]], %[[R4]]] : memref<256x256xf32, 3> -> !xetile.tile<32x64xf32, #xetile.tile_attr<memory_space = 3 : i32>>
    //CHECK: xetile.store_tile  %[[c0]], %[[INITTILESRCMAP]] : vector<32x64xf32>, !xetile.tile<32x64xf32, #xetile.tile_attr<memory_space = 3 : i32>>
    //CHECK: gpu.barrier
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[R5:.*]] = index.divu %[[R0]], %[[c1]]
    //CHECK: %[[R6:.*]] = index.remu %[[R0]], %[[c1]]
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[R7:.*]] = index.mul %[[R5]], %[[c8]]
    //CHECK: %[[c256:.*]] = arith.constant 256 : index
    //CHECK: %[[R8:.*]] = index.mul %[[R6]], %[[c256]]
    //CHECK: %[[R9:.*]] = index.remu %[[R7]], %c256
    //CHECK: %[[R10:.*]] = index.remu %[[R8]], %c256
    //CHECK: %[[INITTILEDSTMAP:.*]] = xetile.init_tile %[[SLMVIEW]][%[[R9]], %[[R10]]] : memref<256x256xf32, 3> -> !xetile.tile<8x256xf32, #xetile.tile_attr<memory_space = 3 : i32>>
    //CHECK: %[[LOADTILE:.*]] = xetile.load_tile %[[INITTILEDSTMAP]] : !xetile.tile<8x256xf32, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x256xf32>

    %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} dense<0.000000e+00> : vector<256x256xf32>
    %cst_temp =  arith.constant {map = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>} dense<0.000000e+00> : vector<256x256xf32>
    %convert_layout = xetile.convert_layout %cst {wg_map_result = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>, wg_map_source = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x256xf32>
    %add = arith.addf %cst_temp, %convert_layout {map = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>} : vector<256x256xf32>
    gpu.return
  }

  gpu.func @test_fold_transpose_to_conv_layout(%arg0: memref<128x8xf32>) {
    %init = xetile.init_tile %arg0 [0, 0]: memref<128x8xf32> -> !xetile.tile<128x8xf32, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 8]>>>
    %data = xetile.load_tile %init : !xetile.tile<128x8xf32, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 8]>>> -> vector<128x8xf32>
    //CHECK: %[[alloc:.*]] = memref.alloc() : memref<4096xi8, 3>
    //CHECK: %[[view:.*]] = memref.view %[[alloc]][{{.*}}][] : memref<4096xi8, 3> to memref<8x128xf32, 3>

    //CHECK: %[[transpose:.*]] = memref.transpose %[[view]] (d0, d1) -> (d1, d0) : memref<8x128xf32, 3> to memref<128x8xf32, strided<[1, 128]>, 3>
    //CHECK: %[[st:.*]] = xetile.init_tile %[[transpose]][{{.*}}] : memref<128x8xf32, strided<[1, 128]>, 3> -> !xetile.tile<4x8xf32, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i32>>
    //CHECK: xetile.store_tile %{{.*}},  %[[st]] : vector<4x8xf32>, !xetile.tile<4x8xf32, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i32>>

    //CHECK: %[[ld:.*]] = xetile.init_tile %[[view]][{{.*}}] : memref<8x128xf32, 3> -> !xetile.tile<4x8xf32, #xetile.tile_attr<memory_space = 3 : i32>>
    //CHECK: %[[data:.*]] = xetile.load_tile %[[ld]] : !xetile.tile<4x8xf32, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<4x8xf32>
    %trans = xetile.transpose %data, [1, 0] {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [8, 4]>} : vector<128x8xf32> -> vector<8x128xf32>
    %conv_layout = xetile.convert_layout %trans {wg_map_result = #xetile.wg_map<sg_layout = [2, 16], sg_data = [4, 8]>} : vector<8x128xf32>
    gpu.return
  }
}
