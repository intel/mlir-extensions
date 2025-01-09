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
    //CHECK: %[[INITTILEDSTMAP:.*]] = xetile.init_tile %[[SLMVIEW]][%[[R7]], %[[R8]]] : memref<256x256xf32, 3> -> !xetile.tile<8x256xf32, #xetile.tile_attr<memory_space = 3 : i32>>
    //CHECK: %[[LOADTILE:.*]] = xetile.load_tile %[[INITTILEDSTMAP]] : !xetile.tile<8x256xf32, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x256xf32>

    %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} dense<0.000000e+00> : vector<256x256xf32>
    %cst_temp =  arith.constant {map = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>} dense<0.000000e+00> : vector<256x256xf32>
    %convert_layout = xetile.convert_layout %cst {wg_map_result = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>, wg_map_source = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x256xf32>
    %add = arith.addf %cst_temp, %convert_layout {map = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 256]>} : vector<256x256xf32>
    gpu.return
    }
  }
