// RUN: imex-opt --xetile-init-duplicate --xetile-blocking --canonicalize --cse %s | FileCheck %s

#slm = #xetile.tile_attr<memory_space = 3>

// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {
  //CHECK: gpu.func @test_gemm(%[[arg0:.*]]: memref<128x128xf16, 3>, %[[arg1:.*]]: memref<128x128xf16, 3>, %[[arg2:.*]]: memref<128x128xf32>) {
  gpu.func @test_gemm(%arg0: memref<128x128xf16, 3>, %arg1: memref<128x128xf16, 3>, %arg2: memref<128x128xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index

    //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
    //CHECK: %[[block_id_y:.*]] = gpu.block_id  y
    //CHECK: %[[r0:.*]] = arith.muli %block_id_x, %[[c16]] : index
    //CHECK: %[[r1:.*]] = arith.muli %block_id_y, %[[c16]] : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c16 : index
    %1 = arith.muli %block_id_y, %c16 : index

    //CHECK: %[[r2:.*]] = xetile.init_tile %[[arg2]][%[[r0]], %[[r1]]] : memref<128x128xf32> -> !xetile.tile<8x16xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r2]] {padding = 0.000000e+00 : f32}  : !xetile.tile<8x16xf32, #xetile.tile_attr<inner_blocks = [8, 16]>> -> vector<1x1x8x16xf32>
    %2 = xetile.init_tile %arg2[%0, %1] : memref<128x128xf32> -> !xetile.tile<8x16xf32>
    %3 = xetile.load_tile %2 {padding = 0.000000e+00 : f32}  : !xetile.tile<8x16xf32> -> vector<8x16xf32>

    //CHECK: %[[r5:.*]] = xetile.init_tile %[[arg0]][%[[r0]], %[[c0]]] : memref<128x128xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<inner_blocks = [1, 16], memory_space = 3 : i64>>
    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[r1]]] : memref<128x128xf16, 3> -> !xetile.tile<16x16xf16, #xetile.tile_attr<inner_blocks = [1, 16], memory_space = 3 : i64>>
    %4 = xetile.init_tile %arg0[%0, %c0] : memref<128x128xf16, 3> -> !xetile.tile<8x16xf16, #slm>
    %5 = xetile.init_tile %arg1[%c0, %1] : memref<128x128xf16, 3> -> !xetile.tile<16x16xf16, #slm>
    %6:3 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3)
          -> (!xetile.tile<8x16xf16, #slm>, !xetile.tile<16x16xf16, #slm>, vector<8x16xf32>) {
      %7 = xetile.load_tile %arg4 {padding = 0.000000e+00 : f32}  : !xetile.tile<8x16xf16, #slm> -> vector<8x16xf16>
      %8 = xetile.load_tile %arg5 {padding = 0.000000e+00 : f32}  : !xetile.tile<16x16xf16, #slm> -> vector<16x16xf16>
      %9 = xetile.tile_mma %7, %8, %arg6 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %10 = xetile.update_tile_offset %arg4, [%c0,  %c16] : !xetile.tile<8x16xf16, #slm>
      %11 = xetile.update_tile_offset %arg5, [%c16,  %c0] : !xetile.tile<16x16xf16, #slm>
      scf.yield %10, %11, %9 : !xetile.tile<8x16xf16, #slm>, !xetile.tile<16x16xf16, #slm>, vector<8x16xf32>
    }
    xetile.store_tile %6#2,  %2 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}
