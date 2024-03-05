// RUN: imex-opt --split-input-file --xetile-blocking --convert-xetile-to-xegpu --remove-dead-values %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
	gpu.func @sg_tiled_store(%a: memref<1024x1024xf32>) {
		%result = arith.constant dense<0.0>: vector<32x32xf32>
		%1 = xetile.init_tile %a[0, 32] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
		xetile.store_tile %result, %1: vector<32x32xf32>, !xetile.tile<32x32xf32>
		gpu.return
	}
}
