
// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

gpu.module @test_dynamic_memref {
  gpu.func @dynamic_memref(%arg0: memref<?x64xf16, strided<[?, 1]>>, %arg1: memref<?x64xf16, strided<[?, 1]>>, %arg2: memref<?x?xf32, strided<[?, 1]>>, %arg3: memref<?x?xf32, strided<[?, 1]>>) kernel attributes {} {
    %c1 = arith.constant 1 : index
    %block_id_x = gpu.block_id  x
    %thread_id_x = gpu.thread_id  x
    %0 = arith.shrsi %block_id_x, %c1 : index
    %1 = arith.shli %0, %c1 : index
    %2 = arith.subi %block_id_x, %1 : index
    %3 = arith.shrsi %thread_id_x, %c1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant {map = #xetile.wg_map<sg_layout = [2, 2], sg_data = [16, 16]>} dense<0.000000e+00> : vector<32x32xf32>
    %4 = arith.shrsi %0, %c1 : index
    %5 = arith.shrsi %4, %c1 : index
    %6 = arith.shli %5, %c1 : index
    %7 = arith.subi %4, %6 : index
    %8 = arith.shli %4, %c1 : index
    %9 = arith.subi %0, %8 : index
    %10 = arith.shrsi %9, %c1 : index
    %11 = arith.shli %10, %c1 : index
    %12 = arith.subi %9, %11 : index
    %13 = arith.shrsi %2, %c1 : index
    %14 = arith.shli %13, %c1 : index
    %15 = arith.subi %2, %14 : index
    %dim = memref.dim %arg1, %c0 : memref<?x64xf16, strided<[?, 1]>>
    %16 = arith.ceildivui %dim, %c64 : index
    %17 = arith.muli %16, %c64 : index
    %dim_0 = memref.dim %arg0, %c0 : memref<?x64xf16, strided<[?, 1]>>
    %18 = arith.ceildivui %dim_0, %c128 : index
    %19 = arith.muli %18, %c128 : index
    %20 = arith.divui %19, %c128 : index
    %stride = memref.dim %arg0, %c0 : memref<?x64xf16, strided<[?, 1]>>
    %21 = arith.divui %17, %c64 : index
    %stride_1 = memref.dim %arg1, %c0 : memref<?x64xf16, strided<[?, 1]>>
    %dim_2 = memref.dim %arg3, %c0 : memref<?x?xf32, strided<[?, 1]>>
    %stride_3 = memref.dim %arg3, %c0 : memref<?x?xf32, strided<[?, 1]>>
    %dim_4 = memref.dim %arg3, %c1 : memref<?x?xf32, strided<[?, 1]>>
    %22 = arith.muli %15, %c32 : index
    %23 = arith.muli %20, %c64 : index
    %24 = arith.muli %7, %23 : index
    %25 = arith.addi %22, %24 : index
    %26 = arith.muli %21, %c32 : index
    %27 = arith.muli %12, %26 : index
    //CHECK: %[[INITTILE_1:.*]] = xetile.init_tile  {{%.*}}[{{%.*}}, {{%.*}}], [{{%.*}}, {{%.*}}], [{{%.*}},  {{%.*}}] : memref<?x?xf32, strided<[?, 1]>> -> !xetile.tile<16x16xf32>
    //CHECK: %[[INITTILE_2:.*]] = xetile.init_tile  {{%.*}}[{{%.*}}, {{%.*}}], [{{%.*}}, {{%.*}}], [{{%.*}},  {{%.*}}] : memref<?x64xf16, strided<[?, 1]>> -> !xetile.tile<16x32xf16>
    //CHECK: %[[INITTILE_3:.*]] = xetile.init_tile  {{%.*}}[{{%.*}}, {{%.*}}], [{{%.*}}, {{%.*}}], [{{%.*}},  {{%.*}}] : memref<?x64xf16, strided<[?, 1]>> -> !xetile.tile<16x32xf16>
    %28 = xetile.init_tile %arg3[%25, %27], [%dim_2, %dim_4], [%stride_3, %c1] : memref<?x?xf32, strided<[?, 1]>> -> !xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>
    %29 = xetile.init_tile %arg0[%25, %c0], [%dim_0, %c64], [%stride, %c1] : memref<?x64xf16, strided<[?, 1]>> -> !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
    %30 = xetile.init_tile %arg1[%27, %c0], [%dim, %c64], [%stride_1, %c1] : memref<?x64xf16, strided<[?, 1]>> -> !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
    %31:2 = scf.for %arg4 = %c0 to %18 step %c1 iter_args(%arg5 = %28, %arg6 = %29) -> (!xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>) {
      %32:2 = scf.for %arg7 = %c0 to %16 step %c1 iter_args(%arg8 = %arg5, %arg9 = %30) -> (!xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>) {
        %35:3 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %cst, %arg12 = %arg6, %arg13 = %arg9) -> (vector<32x32xf32>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>) {
          %38 = xetile.load_tile %arg12 {padding = 0.000000e+00 : f32} : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<32x32xf16>
          %39 = xetile.load_tile %arg13 {padding = 0.000000e+00 : f32} : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<32x32xf16>
          %40 = vector.transpose %39, [1, 0] {map = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 16]>} : vector<32x32xf16> to vector<32x32xf16>
          xegpu.compile_hint
          %41 = xetile.tile_mma %38, %40, %arg11 {wg_map_a = #xetile.wg_map<sg_layout = [2, 2], sg_data = [16, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 16]>, wg_map_c = #xetile.wg_map<sg_layout = [2, 2], sg_data = [16, 16]>} : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
          xegpu.compile_hint
          %42 = xetile.update_tile_offset %arg12, [%c0, %c32] : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
          %43 = xetile.update_tile_offset %arg13, [%c0, %c32] : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
          scf.yield %41, %42, %43 : vector<32x32xf32>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
        }
        xetile.store_tile %35#0,  %arg8 : vector<32x32xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>
        %36 = xetile.update_tile_offset %arg8, [%c0, %c32] : !xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>
        %37 = xetile.update_tile_offset %arg9, [%c32, %c0] : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
        scf.yield %36, %37 : !xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
      }
      %33 = xetile.update_tile_offset %arg5, [%c64, %c0] : !xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>
      %34 = xetile.update_tile_offset %arg6, [%c64, %c0] : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
      scf.yield %33, %34 : !xetile.tile<32x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 16]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 2], sg_data = [16, 32]>, memory_space = 0 : i32, scattered = false>>
    }
    gpu.return
  }
}
