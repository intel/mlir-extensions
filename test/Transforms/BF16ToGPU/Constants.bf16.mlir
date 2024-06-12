// RUN: imex-opt %s --bf16-to-gpu | FileCheck %s

module @bf16_constants {
  gpu.module @test_kernel attributes {} {
    gpu.func @test_kernel(%arg0: memref<10x10xbf16>) kernel attributes {} {
      %cst0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // CHECK: arith.constant 16384 : i16
      %2 = arith.constant 2.0 : bf16
      // CHECK: arith.constant dense<16256> : vector<10xi16>
      %3 = arith.constant dense<1.0> : vector<10xbf16>
      %4 = arith.addf %2, %2 : bf16
      vector.store %3, %arg0[%1, %cst0] : memref<10x10xbf16>, vector<10xbf16>
      memref.store %4, %arg0[%1, %0] : memref<10x10xbf16>
      gpu.return
    }
  }
}
