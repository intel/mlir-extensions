// RUN: imex-opt -convert-xegpu-to-vc %s | FileCheck %s

gpu.module @test_kernel {
  gpu.func @test(%arg0: memref<4x16xf32>) -> vector<2x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %1 = arith.constant dense<1.0> : vector<2x4xf32>
    %tdesc = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<4x16xf32> -> !xegpu.tensor_desc<2x4xf32>
    //CHECK-NOT: builtin.unrealized_conversion_cast {{.*}}: vector<16xi32> -> !xegpu.tensor_desc<2x4xf32>
    %r:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg1 = %tdesc, %arg2 = %1) -> (!xegpu.tensor_desc<2x4xf32>, vector<2x4xf32>) {
      %data = xegpu.load_nd %arg1 : !xegpu.tensor_desc<2x4xf32> -> vector<2x4xf32>
      %next = xegpu.update_nd_offset %arg1, [%c0, %c4] : !xegpu.tensor_desc<2x4xf32>
      %2 = arith.addf %data, %arg2 : vector<2x4xf32>
      scf.yield %next, %2 : !xegpu.tensor_desc<2x4xf32>, vector<2x4xf32>
    }
    gpu.return %r#1 : vector<2x4xf32>
  }
}
