// RUN: imex-opt -convert-xegpu-to-vc -cse --split-input-file %s | FileCheck %s --check-prefixes=CHECK
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {

    // CHECK: gpu.func @test_eltwise(%[[arg0:.*]]: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_eltwise(%arg0: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      %cv1 = arith.constant dense<1.0> : vector<16xf32>
      %v1 = vector.load %arg0[%c0, %c0] : memref<8x16xf32>, vector<16xf32>
      // CHECK: %[[LOG2E_VEC:.*]] = arith.constant dense<1.44{{.*}}> : vector<16xf32>
      // CHECK-NEXT: %[[MULF:.*]] = arith.mulf {{.*}} %[[LOG2E_VEC]]
      // CHECK-NEXT: func.call @llvm.genx.exp.v16f32(%[[MULF]])
      %1 = math.exp %v1 fastmath<nnan> : vector<16xf32>
      // CHECK-NEXT: func.call @llvm.genx.exp.v16f32(%[[MULF]])
      %2 = math.exp %v1 : vector<16xf32>
      // CHECK-NEXT: func.call @llvm.genx.fmax.v16f32
      %4 = arith.maximumf %v1, %cv1 fastmath<nnan> : vector<16xf32>
      %5 = arith.maximumf %v1, %cv1 : vector<16xf32>
      // CHECK-NEXT: gpu.return
      gpu.return
    }
  }
}

// -----

module @gemm attributes {gpu.container_module} {
  gpu.module @exp_f16 {
    // CHECK-LABEL: gpu.func @exp_f16
    gpu.func @exp_f16(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      %v1 = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<16xf16>
      // CHECK: %[[LOG2E_VEC:.*]] = arith.constant dense<1.44{{.*}}> : vector<16xf16>
      // CHECK-NEXT: %[[MULF:.*]] = arith.mulf {{.*}} %[[LOG2E_VEC]]
      // CHECK-NEXT: func.call @llvm.genx.exp.v8i32(%[[MULF]])
      %2 = math.exp %v1 : vector<16xf16>
      gpu.return
    }
  }
}
