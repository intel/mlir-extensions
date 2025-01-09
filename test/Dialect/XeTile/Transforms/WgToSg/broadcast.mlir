// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

gpu.module @test_broadcast {
  gpu.func @test_kernel(%arg0: memref<256x384xf16>, %arg1: memref<1x384xf16>, %arg2: memref<256x512xf32>) attributes {gemm_tiles_b = 1 : i64, gemm_tiles_x = dense<[1, 1, 1, 4]> : vector<4xi64>, gemm_tiles_y = dense<[1, 1, 1, 8]> : vector<4xi64>, habana_runner.num_inputs = 2 : i64, habana_runner.tests = [{inputs = [dense<1.000000e+00> : tensor<256x384xf16>, dense<1.000000e+00> : tensor<1x384xf16>], outputs = [dense<3.840000e+02> : tensor<256x512xf32>]}], physical_nd_range = dense<1> : vector<2xi64>, region_partition = 0 : i64, region_size = 1 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<256x384xf16>, tensor<1x384xf16>) -> tensor<256x512xf32>, synFusionGenOps = 6 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1000015571.16 : f64} {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1_1 = arith.constant 1 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1_0, %arg11 = %c1_1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c4, %arg13 = %c8, %arg14 = %c1_1) {
      %c384 = arith.constant 384 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [64, 64]>} dense<0.000000e+00> : vector<256x512xf32>
      %c0 = arith.constant 0 : index
      %0 = xetile.init_tile %arg0[%c0, %c0] : memref<256x384xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>>>
      %1 = xetile.init_tile %arg1[%c0, %c0] : memref<1x384xf16> -> !xetile.tile<1x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [1, 32]>>>
      %2:3 = scf.for %arg15 = %c0 to %c384 step %c32 iter_args(%arg16 = %cst, %arg17 = %0, %arg18 = %1) -> (vector<256x512xf32>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>>>, !xetile.tile<1x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [1, 32]>>>) {
        %4 = xetile.update_tile_offset %arg18, [%c0,  %c32] : !xetile.tile<1x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [1, 32]>>>
        %5 = xetile.update_tile_offset %arg17, [%c0,  %c32] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>>>
        %6 = xetile.load_tile %arg17 { padding = 0.000000e+00 : f32 }  : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>>> -> vector<256x32xf16>
        %7 = xetile.load_tile %arg18 { padding = 0.000000e+00 : f32 }  : !xetile.tile<1x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [1, 32]>>> -> vector<1x32xf16>
        //CHECK: %[[TRANSPOSE:.*]] = vector.transpose {{%.*}}, [1, 0] : vector<1x32xf16> to vector<32x1xf16>
        %8 = vector.transpose %7, [1, 0] {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 1]>} : vector<1x32xf16> to vector<32x1xf16>
        //CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[TRANSPOSE]] : vector<32x1xf16> to vector<32x64xf16>
        %9 = vector.broadcast %8 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 64]>} : vector<32x1xf16> to vector<32x512xf16>
        xegpu.compile_hint
        %10 = xetile.tile_mma %6, %9, %cst {wg_map_a =#xetile.wg_map<sg_layout = [4, 8], sg_data = [64, 32]>, wg_map_b =#xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 64]>, wg_map_c =#xetile.wg_map<sg_layout = [4, 8], sg_data = [64, 64]>} : vector<256x32xf16>, vector<32x512xf16>, vector<256x512xf32> -> vector<256x512xf32>
        xegpu.compile_hint
        %11 = arith.addf %arg16, %10 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [64, 64]>} : vector<256x512xf32>
        scf.yield %11, %5, %4 : vector<256x512xf32>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>>>, !xetile.tile<1x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [1, 32]>>>
      }
      %3 = xetile.init_tile %arg2[%c0, %c0] : memref<256x512xf32> -> !xetile.tile<256x512xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 64]>>>
      xetile.store_tile %2#0,  %3 : vector<256x512xf32>, !xetile.tile<256x512xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 64]>>>
      gpu.terminator
    }
    gpu.return
  }
}
