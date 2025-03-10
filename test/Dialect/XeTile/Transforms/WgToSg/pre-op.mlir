// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

gpu.module @preop_m512_n256_k64 {
  gpu.func @preop_m512_n256_k64(%arg0: memref<512x64xf16>, %arg1: memref<256x64xf16>, %arg2: memref<256x64xf16>, %arg3: memref<512x256xf32>, %arg4: memref<512x256xf32>) kernel attributes {} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %block_id_x = gpu.block_id  x
    %thread_id_x = gpu.thread_id  x
    %c0 = arith.constant 0 : index
    %0 = arith.shrsi %block_id_x, %c0 : index
    %c2 = arith.constant 2 : index
    %1 = arith.shrsi %thread_id_x, %c2 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 64]>} dense<0.000000e+00> : vector<512x256xf32>
    %2 = xetile.init_tile %arg0[%c0, %c0] : memref<512x64xf16> -> !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
    %3 = xetile.init_tile %arg1[%c0, %c0] : memref<256x64xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
    %4 = xetile.init_tile %arg2[%c0, %c0] : memref<256x64xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
    %5:4 = scf.for %arg5 = %c0 to %c64 step %c32 iter_args(%arg6 = %cst, %arg7 = %2, %arg8 = %3, %arg9 = %4) -> (vector<512x256xf32>, !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>) {
      %10 = xetile.load_tile %arg7 : !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<512x32xf16>
      %11 = xetile.load_tile %arg8 : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<256x32xf16>
      %12 = xetile.load_tile %arg9 : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<256x32xf16>
      //CHECK: %[[ADDF:.*]] arith.addf {{%.*}}, {{%.*}} : vector<64x32xf16>
      %13 = arith.addf %11, %12 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [64, 32]>} : vector<256x32xf16>
      xegpu.compile_hint
      %14 = vector.transpose %13, [1, 0] {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 64]>} : vector<256x32xf16> to vector<32x256xf16>
      %15 = xetile.tile_mma %10, %14, %arg6 {wg_map_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 64]>, wg_map_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 64]>} : vector<512x32xf16>, vector<32x256xf16>, vector<512x256xf32> -> vector<512x256xf32>
      xegpu.compile_hint
      %16 = xetile.update_tile_offset %arg7, [%c0, %c32] : !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
      %17 = xetile.update_tile_offset %arg8, [%c0, %c32] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
      %18 = xetile.update_tile_offset %arg9, [%c0, %c32] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
      scf.yield %15, %16, %17, %18 : vector<512x256xf32>, !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
    }
    %6 = xetile.init_tile %arg3[%c0, %c0] : memref<512x256xf32> -> !xetile.tile<512x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 64]>, memory_space = 0 : i32, scattered = false>>
    %7 = xetile.load_tile %6 : !xetile.tile<512x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 64]>, memory_space = 0 : i32, scattered = false>> -> vector<512x256xf32>
    %8 = arith.addf %5#0, %7 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 64]>} : vector<512x256xf32>
    %9 = xetile.init_tile %arg4[%c0, %c0] : memref<512x256xf32> -> !xetile.tile<512x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 64]>, memory_space = 0 : i32, scattered = false>>
    xetile.store_tile %8,  %9 : vector<512x256xf32>, !xetile.tile<512x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 64]>, memory_space = 0 : i32, scattered = false>>
    gpu.return
  }
}
