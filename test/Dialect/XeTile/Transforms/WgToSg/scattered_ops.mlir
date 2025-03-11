// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-wg-to-sg %s -verify-diagnostics | FileCheck %s

gpu.module @test_scattered_ops {
//CHECK:  gpu.func @test_gather_scatter(%[[arg0:.*]]:  memref<32768xf32>)
gpu.func @test_gather_scatter(%arg0 : memref<32768xf32>) {
        //CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<32x32xi1>
        //CHECK: %[[ADDRESS:.*]] = arith.constant dense<1> : vector<32x32xindex>
        //CHECK: %[[INITTILE:.*]] = xetile.init_tile %[[arg0]], %[[ADDRESS]] : memref<32768xf32>, vector<32x32xindex> -> !xetile.tile<32x32xf32, #xetile.tile_attr<scattered = true>>
        //CHECK: %[[LOADTILE:.*]] = xetile.load %[[INITTILE]], %[[MASK]] : !xetile.tile<32x32xf32, #xetile.tile_attr<scattered = true>>, vector<32x32xi1> -> vector<32x32xf32>
        //CHECK: xetile.store %[[LOADTILE]], %[[INITTILE]], %[[MASK]] : vector<32x32xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<scattered = true>>, vector<32x32xi1>
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} dense<true> : vector<128x256xi1>
        %cst_0 = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} dense<1> : vector<128x256xindex>
        %tile = xetile.init_tile %arg0, %cst_0 : memref<32768xf32>, vector<128x256xindex> -> !xetile.tile<128x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, scattered = true>>
        %load_tile_0 = xetile.load %tile, %cst : !xetile.tile<128x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, scattered = true>>, vector<128x256xi1> -> vector<128x256xf32>
        xetile.store %load_tile_0, %tile, %cst :  vector<128x256xf32>, !xetile.tile<128x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, scattered = true>>, vector<128x256xi1>
        gpu.return
    }
}
