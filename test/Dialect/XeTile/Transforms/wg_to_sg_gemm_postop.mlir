// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

#wg_map_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [40, 32]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a, inner_blocks = [], memory_scope = 0>
#wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 96]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b, inner_blocks = [], memory_scope = 0>
#wg_map_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [40, 96]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c, inner_blocks = [], memory_scope = 0>

#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (12288)>

gpu.module @test_gemm_postop  {
     //CHECK:  gpu.func @test_kernel(%[[arg0:.*]]: memref<16384x12288xf16>, %[[arg1:.*]]: memref<12288x1536xf16>, %[[arg2:.*]]:  memref<16384x1536xf32>)
     gpu.func @test_kernel(%arg0: memref<16384x12288xf16>, %arg1: memref<12288x1536xf16>, %arg2: memref<16384x1536xf32>){
    //CHECK: %[[c384:.*]] = arith.constant 384 : index
    //CHECK: %[[c320:.*]] = arith.constant 320 : index
    //CHECK: %[[c2240:.*]] = arith.constant 2240 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c12288:.*]] = arith.constant 12288 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<40x96xf32>
    //CHECK: %[[c4:.*]] = arith.constant 4 : index
    //CHECK: %[[R0:.*]] = gpu.block_id  x
    //CHECK: %[[R1:.*]] = gpu.block_id  y

    %c384 = arith.constant 384 : index
    %c320 = arith.constant 320 : index
    %c2240 = arith.constant 2240 : index
    %c32 = arith.constant 32 : index
    %c12288 = arith.constant 12288 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant {map = #wg_map_c} dense<0.000000e+00> : vector<320x384xf32>
    %c4 = arith.constant 4 : index
    %0 = gpu.block_id  x
    %1 = gpu.block_id  y
    %2 = arith.divsi %1, %c4 : index
    %3 = arith.remsi %1, %c4 : index
    %4 = arith.muli %0, %c2240 : index
    %5 = arith.muli %2, %c320 : index
    %6 = arith.addi %4, %5 : index
    %7 = arith.muli %3, %c384 : index
    %8 = xetile.init_tile %arg0[%6, %c0] : memref<16384x12288xf16> -> !xetile.tile<320x32xf16, #tile_attr_a>
    %9 = xetile.init_tile %arg1[%7, %c0] : memref<12288x1536xf16> -> !xetile.tile<32x384xf16, #tile_attr_b>
    %10:3 = scf.for %arg3 = %c0 to %c12288 step %c32 iter_args(%a_tile = %8, %b_tile = %9, %arg4 = %cst)
    -> (!xetile.tile<320x32xf16, #tile_attr_a>,
        !xetile.tile<32x384xf16, #tile_attr_b>,
        vector<320x384xf32>) {
      //CHECK: %[[LOADTILE1:.*]] = xetile.load_tile {{%.*}} {padding = 0.000000e+00 : f32}  : !xetile.tile<40x32xf16> -> vector<40x32xf16>
      %23 = xetile.load_tile %a_tile {padding = 0.000000e+00 : f32}  : !xetile.tile<320x32xf16, #tile_attr_a> -> vector<320x32xf16>
      //CHECK: %[[LOADTILE2:.*]] = xetile.load_tile {{%.*}} {padding = 0.000000e+00 : f32}  : !xetile.tile<32x96xf16> -> vector<32x96xf16>
      %24 = xetile.load_tile %b_tile {padding = 0.000000e+00 : f32}  : !xetile.tile<32x384xf16, #tile_attr_b> -> vector<32x384xf16>
      %25 = xetile.update_tile_offset %a_tile, [%c0,  %c32] : !xetile.tile<320x32xf16, #tile_attr_a>, index, index -> !xetile.tile<320x32xf16, #tile_attr_a>
      %26 = xetile.update_tile_offset %b_tile, [%c32,  %c0] : !xetile.tile<32x384xf16, #tile_attr_b>, index, index -> !xetile.tile<32x384xf16, #tile_attr_b>
      //CHECK: %[[TILEMMA:.*]] = xetile.tile_mma {{%.*}}, {{%.*}}, {{%.*}} : vector<40x32xf16>, vector<32x96xf16>, vector<40x96xf32> -> vector<40x96xf32>
      %27 = xetile.tile_mma %23, %24, %arg4 {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c}: vector<320x32xf16>, vector<32x384xf16>, vector<320x384xf32> -> vector<320x384xf32>
      scf.yield %25, %26, %27 :
        !xetile.tile<320x32xf16, #tile_attr_a>,
        !xetile.tile<32x384xf16, #tile_attr_b>,
        vector<320x384xf32>
    } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 1>, step = 32 : index, upperBoundMap = #map1}
    //CHECK: %[[POSTOP:.*]] = math.exp {{%.*}} : vector<40x96xf32>
    %11 = math.exp %10#2 {map = #wg_map_c} : vector<320x384xf32>
    %12 = arith.muli %0, %c2240 : index
    %13 = arith.muli %2, %c320 : index
    %14 = arith.addi %12, %13 : index
    %15 = arith.muli %3, %c384 : index
    %16 = xetile.init_tile %arg2[%14, %15] : memref<16384x1536xf32> -> !xetile.tile<320x384xf32, #tile_attr_c>
    xetile.store_tile %11,  %16 : vector<320x384xf32>, !xetile.tile<320x384xf32, #tile_attr_c>
    gpu.return
    }
}
