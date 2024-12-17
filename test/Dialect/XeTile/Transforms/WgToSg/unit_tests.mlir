// RUN: imex-opt --split-input-file --xetile-wg-to-sg %s -verify-diagnostics | FileCheck %s

gpu.module @test_arith_extf {
    gpu.func @test_kernel(%arg0: memref<128x32xf16>) {
        %c0 = arith.constant 0 : index
        %tile = xetile.init_tile %arg0[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %load_tile = xetile.load_tile %tile : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xf16>
        //CHECK: arith.extf {{%.*}} : vector<32x32xf16> to vector<32x32xf32>
        //CHECK: arith.truncf {{%.*}} : vector<32x32xf32> to vector<32x32xf16>
        %extf = arith.extf %load_tile {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf16> to vector<128x32xf32>
        %trucf = arith.truncf %extf {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf32> to vector<128x32xf16>
       gpu.return
    }
}
