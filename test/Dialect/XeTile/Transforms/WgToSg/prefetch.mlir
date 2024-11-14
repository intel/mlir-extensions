// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

gpu.module @test_prefetch{
  gpu.func @preop_addexp_m512_n256_k4096(%arg0: memref<512x4096xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<512x256xf32>, %arg3: memref<512x256xf32>) attributes {gemm_tiles_b = 1 : i64, gemm_tiles_x = dense<[2, 1, 2, 4]> : vector<4xi64>, gemm_tiles_y = dense<[1, 1, 1, 8]> : vector<4xi64>, habana_runner.num_inputs = 3 : i64, habana_runner.tests = [{inputs = [dense<1.000000e+00> : tensor<512x4096xf16>, dense<0.000000e+00> : tensor<256x4096xf16>, dense<1.900000e+01> : tensor<512x256xf32>], outputs = [dense<8.211000e+03> : tensor<512x256xf32>]}], physical_nd_range = dense<2> : vector<2xi64>, region_partition = 0 : i64, region_size = 2 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<512x4096xf16>, tensor<256x4096xf16>, tensor<512x256xf32>) -> tensor<512x256xf32>, synFusionGenOps = 6 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1000081812.83 : f64} {
    %c2 = arith.constant 2 : index
    %c2_0 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %c2, %arg11 = %c2_0, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c4, %arg14 = %c8, %arg15 = %c1) {
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c320 = arith.constant 320 : index
      %c4096 = arith.constant 4096 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} dense<0.000000e+00> : vector<128x256xf32>
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 : index
      %1 = arith.muli %block_id_y, %c128 : index
      %2 = arith.addi %0, %1 : index
      %3 = xetile.init_tile %arg0[%2, %c0] : memref<512x4096xf16> -> !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      %4 = xetile.init_tile %arg1[%c0, %c0] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      %5:2 = scf.for %arg16 = %c0 to %c320 step %c32 iter_args(%arg17 = %3, %arg18 = %4) -> (!xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>) {
        %18 = xetile.update_tile_offset %arg18, [%c0, %c32] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        %19 = xetile.update_tile_offset %arg17, [%c0, %c32] : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        //CHECK: xetile.prefetch_tile {{%.*}} {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
        //CHECK: xetile.prefetch_tile {{%.*}} {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
        xetile.prefetch_tile %arg17 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        xetile.prefetch_tile %arg18 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        scf.yield %19, %18 : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      }
      %6 = xetile.init_tile %arg0[%2, %c320] : memref<512x4096xf16> -> !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      %7 = xetile.init_tile %arg1[%c0, %c320] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      %8 = xetile.init_tile %arg0[%2, %c0] : memref<512x4096xf16> -> !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      %9 = xetile.init_tile %arg1[%c0, %c0] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      %10:5 = scf.for %arg16 = %c0 to %c4096 step %c32 iter_args(%arg17 = %cst, %arg18 = %6, %arg19 = %7, %arg20 = %8, %arg21 = %9) -> (vector<128x256xf32>, !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>) {
        %18 = xetile.update_tile_offset %arg21, [%c0, %c32] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        %19 = xetile.update_tile_offset %arg20, [%c0, %c32] : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        %20 = xetile.update_tile_offset %arg19, [%c0, %c32] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        %21 = xetile.update_tile_offset %arg18, [%c0, %c32] : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        %22 = arith.addi %arg16, %c320 : index
        %23 = arith.cmpi sge, %22, %c4096 : index
        scf.if %23 {
        } else {
          //CHECK: xetile.prefetch_tile {{%.*}} {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
          //CHECK: xetile.prefetch_tile {{%.*}} {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
          xetile.prefetch_tile %arg18 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
          xetile.prefetch_tile %arg19 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        }
        %24 = xetile.load_tile %arg20 : !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>> -> vector<128x32xf16>
        %25 = arith.addf %24, %24 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf16>
        %26 = xetile.load_tile %arg21 : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>> -> vector<256x32xf16>
        %27 = vector.transpose %26, [1, 0] {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<256x32xf16> to vector<32x256xf16>
        %28 = math.exp %27 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<32x256xf16>
        xegpu.compile_hint
        %29 = xetile.tile_mma %25, %28, %cst {wg_map_a = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>, wg_map_c = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x32xf16>, vector<32x256xf16>, vector<128x256xf32> -> vector<128x256xf32>
        xegpu.compile_hint
        %30 = arith.addf %arg17, %29 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x256xf32>
        scf.yield %30, %21, %20, %19, %18 : vector<128x256xf32>, !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [4, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<128x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      }
      %11 = arith.muli %block_id_x, %c256 : index
      %12 = arith.muli %block_id_y, %c128 : index
      %13 = arith.addi %11, %12 : index
      %14 = xetile.init_tile %arg2[%13, %c0] : memref<512x256xf32> -> !xetile.tile<128x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      %15 = xetile.load_tile %14 : !xetile.tile<128x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>> -> vector<128x256xf32>
      %16 = arith.addf %10#0, %15 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<128x256xf32>
      %17 = xetile.init_tile %arg3[%13, %c0] : memref<512x256xf32> -> !xetile.tile<128x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      xetile.store_tile %16,  %17 : vector<128x256xf32>, !xetile.tile<128x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      gpu.terminator
    }
    gpu.return
  }
}
