// RUN: imex-opt -convert-math-to-vc -verify-diagnostics  %s | FileCheck %s --check-prefixes=CHECK
// RUN: imex-opt -convert-math-to-vc="enable-high-precision-interim-calculation=true" -verify-diagnostics %s | FileCheck %s --check-prefixes=HIGH_PRECISION

module @gemm attributes {gpu.container_module} {
  gpu.module @math_to_vc {
    // CHECK-LABEL: gpu.func @ceil_f16
    gpu.func @ceil_f16(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      %0 = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<16xf16>
      // CHECK: %[[EXTF_F32:.*]] = arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
      // CHECK-NEXT: %[[CEILF:.*]] = func.call @llvm.genx.rndu.v16f32(%[[EXTF_F32]]) : (vector<16xf32>) -> vector<16xf32>
      // CHECK-NEXT: %[[TRUNC_F16:.*]] = arith.truncf %[[CEILF]] : vector<16xf32> to vector<16xf16>
      %2 = math.ceil %0 : vector<16xf16>
      gpu.return
    }

    // -----

    // CHECK-LABEL: gpu.func @ceil_f64
    gpu.func @ceil_f64(%arg0: memref<8x16xf64>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      %0 = vector.load %arg0[%c0, %c0] : memref<8x16xf64>, vector<16xf64>
      // CHECK: %[[TRUNCF_F32:.*]] = arith.truncf {{.*}} : vector<16xf64> to vector<16xf32>
      // CHECK-NEXT: %[[CEILF:.*]] = func.call @llvm.genx.rndu.v16f32(%[[TRUNCF_F32]]) : (vector<16xf32>) -> vector<16xf32>
      // CHECK-NEXT: %[[EXTF_F64:.*]] = arith.extf %[[CEILF]] : vector<16xf32> to vector<16xf64>
      // expected-warning@+1 {{Truncation is done on input during conversion, may result in wrong result.}}
      %2 = math.ceil %0 : vector<16xf64>
      gpu.return
    }

    // -----

    // CHECK-LABEL: gpu.func @floor_f16
    gpu.func @floor_f16(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      %0 = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<16xf16>
      // CHECK: %[[EXTF_F32:.*]] = arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
      // CHECK-NEXT: %[[CEILF:.*]] = func.call @llvm.genx.rndd.v16f32(%[[EXTF_F32]]) : (vector<16xf32>) -> vector<16xf32>
      // CHECK-NEXT: %[[TRUNC_F16:.*]] = arith.truncf %[[CEILF]] : vector<16xf32> to vector<16xf16>
      %2 = math.floor %0 : vector<16xf16>
      gpu.return
    }

    // -----

    // CHECK-LABEL: gpu.func @exp_f16
    // HIGH_PRECISION-LABEL: gpu.func @exp_f16
    gpu.func @exp_f16(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      // HIGH_PRECISION: %[[VEC:.*]] = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<16xf16>
      %v1 = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<16xf16>
      // CHECK: %[[LOG2E_VEC:.*]] = arith.constant dense<1.44{{.*}}> : vector<16xf16>
      // CHECK-NEXT: %[[MULF:.*]] = arith.mulf {{.*}} %[[LOG2E_VEC]]
      // CHECK-NEXT: func.call @llvm.genx.exp.v8i32(%[[MULF]])
      // HIGH_PRECISION: %[[LOG2E_VEC:.*]] = arith.constant dense<1.44{{.*}}> : vector<16xf32>
      // HIGH_PRECISION: %[[VEC_F32:.*]] = arith.extf %[[VEC]] : vector<16xf16> to vector<16xf32>
      // HIGH_PRECISION: %[[MULF_F32:.*]] = arith.mulf %[[VEC_F32]], %[[LOG2E_VEC]] : vector<16xf32>
      // HIGH_PRECISION: %[[MULF:.*]] = arith.truncf %[[MULF_F32]] : vector<16xf32> to vector<16xf16>
      // HIGH_PRECISION: func.call @llvm.genx.exp.v8i32(%[[MULF]])
      %2 = math.exp %v1 : vector<16xf16>
      gpu.return
    }

    // -----

    // CHECK-LABEL: gpu.func @exp2_f16
    gpu.func @exp2_f16(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      // CHECK: %[[VEC:.*]] = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<16xf16>
      %v1 = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<16xf16>
      // CHECK-NEXT: func.call @llvm.genx.exp.v8i32(%[[VEC]])
      %2 = math.exp2 %v1 : vector<16xf16>
      gpu.return
    }
  }
}

// -----
