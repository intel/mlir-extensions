// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

module attributes {gpu.container_module} {
  func.func @gemm_m512_n256_k4096_entry(%arg0: memref<512x4096xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<512x256xf32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    gpu.launch_func  @gemm_m512_n256_k4096::@gemm_m512_n256_k4096 blocks in (%c1, %c1, %c1) threads in (%c8, %c4, %c1)  args(%arg0 : memref<512x4096xf16>, %arg1 : memref<256x4096xf16>, %arg2 : memref<512x256xf32>)
    return
  }
  gpu.module @gemm_m512_n256_k4096 {
    gpu.func @gemm_m512_n256_k4096(%arg0: memref<512x4096xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<512x256xf32>) kernel attributes {} {
      %c-4000 = arith.constant -4000 : index
      %c-4064 = arith.constant -4064 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 64]>} dense<0.000000e+00> : vector<512x256xf32>
      %c96 = arith.constant 96 : index
      %c32 = arith.constant 32 : index
      scf.for %arg3 = %c0 to %c96 step %c32 {
        %4 = xetile.init_tile %arg0[%c0, %arg3] : memref<512x4096xf16> -> !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
        xetile.prefetch_tile %4 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
        %5 = xetile.init_tile %arg1[%c0, %arg3] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
        xetile.prefetch_tile %5 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
      }
      %0 = xetile.init_tile %arg0[%c0, %c0] : memref<512x4096xf16> -> !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
      xetile.prefetch_tile %0 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
      %1 = xetile.init_tile %arg1[%c0, %c0] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
      xetile.prefetch_tile %1 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
      //CHECK: %[[c4096:.*]] = arith.constant 4096 : index
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<64x64xf32>
      //CHECK: %[[c32:.*]] = arith.constant 32 : index
      //CHECK: %[[SCF_FOR:.*]] = scf.for %[[arg3:.*]] = %[[c0]] to %[[c4096]] step %[[c32]]
      //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[CST]]) -> (vector<64x64xf32>)
      %2 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %cst) -> (vector<512x256xf32>) {
        %4 = arith.addi %arg3, %c32 : index
        %5 = arith.addi %arg3, %c-4064 : index
        %6 = arith.cmpi sge, %5, %c0 : index
        scf.if %6 {
        } else {
          %16 = xetile.init_tile %arg0[%c0, %4] : memref<512x4096xf16> -> !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
          xetile.prefetch_tile %16 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
          %17 = xetile.init_tile %arg1[%c0, %4] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
          xetile.prefetch_tile %17 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
        }
        %7 = arith.addi %arg3, %c96 : index
        %8 = arith.addi %arg3, %c-4000 : index
        %9 = arith.cmpi sge, %8, %c0 : index
        scf.if %9 {
        } else {
          %16 = xetile.init_tile %arg0[%c0, %7] : memref<512x4096xf16> -> !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
          xetile.prefetch_tile %16 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
          %17 = xetile.init_tile %arg1[%c0, %7] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
          xetile.prefetch_tile %17 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [32, 1], sg_data = [8, 32]>, memory_space = 0 : i32, scattered = false>>
        }
        %10 = xetile.init_tile %arg0[%c0, %arg3] : memref<512x4096xf16> -> !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
        %11 = xetile.load_tile %10 : !xetile.tile<512x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<512x32xf16>
        %12 = xetile.init_tile %arg1[%c0, %arg3] : memref<256x4096xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>>
        %13 = xetile.load_tile %12 : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<256x32xf16>
        %14 = vector.transpose %13, [1, 0] {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x32xf16> to vector<32x256xf16>
        xegpu.compile_hint
        %15 = xetile.tile_mma %11, %14, %arg4 {wg_map_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>, wg_map_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [64, 64]>} : vector<512x32xf16>, vector<32x256xf16>, vector<512x256xf32> -> vector<512x256xf32>
        xegpu.compile_hint
        scf.yield %15 : vector<512x256xf32>
      }
      %3 = xetile.init_tile %arg2[%c0, %c0] : memref<512x256xf32> -> !xetile.tile<512x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 64]>, memory_space = 0 : i32, scattered = false>>
      xetile.store_tile %2,  %3 : vector<512x256xf32>, !xetile.tile<512x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [64, 64]>, memory_space = 0 : i32, scattered = false>>
      gpu.return
    }
  }
}
