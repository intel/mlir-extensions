// RUN: imex-opt --split-input-file --xetile-blockop-fallback=device=pvc %s -verify-diagnostics -o -| FileCheck %s

gpu.module @m64_n16_k896 {
  gpu.func @m64_n16_k896(%arg0: memref<256x896xf16>, %arg1: memref<896x256xf16>, %arg2: memref<256x256xf32>, %arg3: memref<14336x128xf32>) kernel attributes {} {
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c28 = arith.constant 28 : index
    %c4 = arith.constant 4 : index
    %block_id_x = gpu.block_id  x
    %thread_id_x = gpu.thread_id  x
    %0 = arith.shrsi %block_id_x, %c5 : index
    %1 = arith.addi %0, %c1 : index
    %2 = arith.muli %1, %c4 : index
    %3 = arith.addi %block_id_x, %2 : index
    %4 = arith.shrsi %3, %c5 : index
    %5 = arith.muli %4, %c28 : index
    %6 = arith.subi %block_id_x, %5 : index
    %7 = arith.shrsi %4, %c1 : index
    %8 = arith.muli %7, %c2 : index
    %9 = arith.subi %4, %8 : index
    %10 = arith.shrsi %9, %c1 : index
    %11 = arith.muli %10, %c2 : index
    %12 = arith.subi %9, %11 : index
    %13 = arith.muli %6, %c32 : index
    %14 = arith.shrsi %thread_id_x, %c2 : index
    %15 = arith.muli %14, %c4 : index
    %16 = arith.subi %thread_id_x, %15 : index
    %17 = arith.muli %14, %c32 : index
    %18 = arith.shrsi %17, %c8 : index
    %19 = arith.muli %18, %c256 : index
    %20 = arith.subi %17, %19 : index
    %21 = arith.muli %16, %c32 : index
    %22 = arith.shrsi %21, %c7 : index
    %23 = arith.muli %22, %c128 : index
    %24 = arith.subi %21, %23 : index
    %25 = arith.muli %12, %c128 overflow<nsw> : index
    %26 = arith.addi %25, %24 : index
    %27 = xetile.init_tile %arg0[%20, %13] : memref<256x896xf16> -> !xetile.tile<32x32xf16>
    %28 = xetile.init_tile %arg1[%13, %26] : memref<896x256xf16> -> !xetile.tile<32x32xf16>
    %29 = xetile.load_tile %27 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    %30 = xetile.load_tile %28 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    xegpu.compile_hint
    xegpu.compile_hint
    %31 = xetile.tile_mma %29, %30, %cst : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
    xegpu.compile_hint
    %32 = xetile.update_tile_offset %27, [%c0, %c32] : !xetile.tile<32x32xf16>
    %33 = xetile.update_tile_offset %28, [%c32, %c0] : !xetile.tile<32x32xf16>
    //CHECK: %[[INITTILE:.*]] = xetile.init_tile {{%.*}}, {{%.*}} : memref<65536xf32>, vector<32x32xindex> -> !xetile.tile<32x32xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    //CHECK: %[[RMW:.*]] = xetile.atomic_rmw addf {{%.*}}, {{%.*}} : vector<32x32xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>> -> vector<32x32xf32>
    %34 = xetile.init_tile %arg2[%20, %26] : memref<256x256xf32> -> !xetile.tile<32x32xf32>
    %35 = xetile.atomic_rmw addf %31, %34 : vector<32x32xf32>, !xetile.tile<32x32xf32> -> vector<32x32xf32>
    gpu.return
  }
}
