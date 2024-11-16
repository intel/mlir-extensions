// RUN: imex-opt -convert-arith-to-vc -verify-diagnostics  %s | FileCheck %s --check-prefixes=CHECK

module @arith_to_vc attributes {gpu.container_module} {
  gpu.module @arith_to_vc {
    // CHECK-LABEL: gpu.func @maximumf_f16
    gpu.func @maximumf_f16(%arg0: vector<16xf16>, %arg1: vector<16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      // CHECK: %[[VC_RES:.*]] = func.call @llvm.genx.fmax.v8i32(%arg0, %arg1) : (vector<16xf16>, vector<16xf16>) -> vector<16xf16>
      %res0 = arith.maximumf %arg0, %arg1 fastmath<nnan> : vector<16xf16>
      // CHECK-NEXT: %[[UNCONVERTED_RES:.*]] = arith.maximumf %arg0, %arg1 : vector<16xf16>
      %res1 = arith.maximumf %arg0, %arg1 : vector<16xf16>
      gpu.return
    }
  }
}
