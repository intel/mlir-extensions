// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu='device=pvc' --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s

func.func @sglevel_tiled_gemm(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>) {
   //CHECK: arith.constant 0 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index

    %1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<4x2x8x16xf16>

    %2 = xetile.load_tile %1 : !xetile.tile<4x2x8x16xf16> -> vector<4x2x8x16xf16>

    %3 = xetile.init_tile %b[%c64, %c0] : memref<1024x1024xf16> -> !xetile.tile<2x4x16x16xf16>

    %4 = xetile.load_tile %3 : !xetile.tile<2x4x16x16xf16> -> vector<2x4x16x16xf16>

    %6 = xetile.tile_mma %2, %4: vector<4x2x8x16xf16>, vector<2x4x16x16xf16> -> vector<4x4x8x16xf32>

    return
}


func.func @sglevel_tiled_store(%a: memref<1024x1024xf32>) {
	// CHECK: arith.constant 0 : index
    %1 = xetile.init_tile %a[0, 32] : memref<1024x1024xf32> -> !xetile.tile<8x4x8x16xf32>
    %result = arith.constant dense<0.0>: vector<8x4x8x16xf32>
    xetile.store_tile %result, %1: vector<8x4x8x16xf32>, !xetile.tile<8x4x8x16xf32>
    return
}
