// RUN: imex-opt %s --bf16-to-gpu | FileCheck %s

module @bf16_constants {
  gpu.module @test_kernel attributes {} {
    gpu.func @test_kernel(%arg0: memref<10x10xbf16>, %arg1: memref<10x10xf32>) kernel attributes {} {
      %cst0 = arith.constant 0 : index
      // CHECK: %[[LOAD:.*]] = vector.load %arg0[%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>
      // CHECK: %[[BCAST:.*]] = arith.bitcast %[[LOAD]] : vector<10x10xi16> to vector<10x10xbf16>
      // CHECK: arith.extf %[[BCAST]] : vector<10x10xbf16> to vector<10x10xf32>
      %2 = vector.load %arg0[%cst0, %cst0] : memref<10x10xbf16>, vector<10x10xbf16>
      %3 = arith.extf %2 : vector<10x10xbf16> to vector<10x10xf32>

      vector.store %3, %arg1[%cst0, %cst0] : memref<10x10xf32>, vector<10x10xf32>
      gpu.return
    }
  }
}
