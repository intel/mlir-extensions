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

    gpu.func @test_reduction_and_shape_cast(%arg0 : vector<256x128xf32>) {
        //CHECK: %[[CST_0:.*]] = arith.constant dense<-1.000000e+00> : vector<32x32xf32>
        //CHECK: %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<1x32xf32>
        //CHECK: %[[SQRT:.*]] = math.sqrt %[[CST_0]] : vector<32x32xf32>
        //CHECK: %[[SHAPECAST_0:.*]] = vector.shape_cast %[[CST_1]] : vector<1x32xf32> to vector<32xf32>
        //CHECK: %[[REDUCTION_0:.*]] = vector.multi_reduction <add>, %[[SQRT]], %[[SHAPECAST_0]] [0] : vector<32x32xf32> to vector<32xf32>
        //CHECK: %[[SHAPECAST_1:.*]] = vector.shape_cast %[[REDUCTION_0]] : vector<32xf32> to vector<1x32xf32>
        %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} dense<-1.0> : vector<256x128xf32>
        %cst_0 = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} dense<0.000000e+00> : vector<8x128xf32>
        %sqrt = math.sqrt %cst {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32>
        %reshape = vector.shape_cast %sqrt {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32> to vector<8x32x128xf32>
        %reduction = vector.multi_reduction <add>, %reshape, %cst_0 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} [1] : vector<8x32x128xf32> to vector<8x128xf32>
        //CHECK: xetile.store_tile  {{%.*}}, {{%.*}} : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3 : i32>>
        //CHECK: gpu.barrier
        //CHECK: %[[LOADTILE_SLM:.*]] = xetile.load_tile {{%.*}} : !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x4xf32>
        //CHECK: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        //CHECK: %[[REDUCTION_1:.*]] = vector.multi_reduction <add>, %[[LOADTILE_SLM]], %[[CST_2]] [0] : vector<8x4xf32> to vector<4xf32>
        //CHECK: %[[SHAPECAST_2:.*]] = vector.shape_cast %[[REDUCTION_1]] : vector<4xf32> to vector<1x4xf32>
        %conv_layout = xetile.convert_layout %reduction {wg_map_result = #xetile.wg_map<sg_layout = [1, 32], sg_data = [8, 4]>} : vector<8x128xf32>
        %cst_1 = arith.constant {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} dense<0.000000e+00> : vector<128xf32>
        %reduce = vector.multi_reduction <add>, %conv_layout, %cst_1 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} [0] : vector<8x128xf32> to vector<128xf32>
        %shape_cast = vector.shape_cast %reduce {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} : vector<128xf32> to vector<1x128xf32>
        gpu.return
    }
}
