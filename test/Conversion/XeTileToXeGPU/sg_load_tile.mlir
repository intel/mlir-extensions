// RUN: imex-opt --split-input-file --xetile-blocking --convert-xetile-to-xegpu --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  gpu.func @sg_load_tile(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
  	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  	gpu.return
  }
}
