// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {

  gpu.module @test_kernel {
    // CHECK: func.func private @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64(vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, vector<16xi1>, vector<16xi64>, vector<16xi32>, vector<16xi32>, i32, vector<16xi32>) -> vector<16xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64", linkage_type = <Import>>}
    gpu.func @test_atomiclsc(%arg0: memref<128xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK:  %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
      %mask = arith.constant dense<true> : vector<16xi1>

      // CHECK: %[[cst_0:.*]] = arith.constant dense<5.000000e-01> : vector<16xf32>
      %1 = arith.constant dense<0.5> : vector<16xf32>

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<128xf32> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst_1:.*]] = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi64>
      //CHECK: %[[r1:.*]] = vector.broadcast %[[r0]] : i64 to vector<16xi64>
      //CHECK: %[[r2:.*]] = arith.addi %[[r1]], %[[cst_1]] : vector<16xi64>
      //CHECK: %[[c19_i8:.*]] = arith.constant 19 : i8
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[c1_i16:.*]] = arith.constant 1 : i16
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[c3_i8:.*]] = arith.constant 3 : i8
      //CHECK: %[[undef:.*]] = spirv.Undef : vector<16xi32>
      //CHECK: %[[r3:.*]] = vector.bitcast %[[cst_0]] : vector<16xf32> to vector<16xi32>
      //CHECK: %[[r4:.*]] = func.call @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64(
      //CHECK-SAME: %[[cst]], %[[c19_i8]], %[[c1_i8]], %[[c1_i8]], %[[c1_i16]], %[[c0_i32]], %[[c3_i8]],
      //CHECK-SAME: %[[c1_i8]], %[[c1_i8]], %[[cst]], %[[r2]], %[[r3]], %[[undef]], %[[c0_i32]], %[[undef]])
      //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, vector<16xi1>, vector<16xi64>, vector<16xi32>, vector<16xi32>, i32, vector<16xi32>) -> vector<16xi32>
      %offsets = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %2 = xegpu.create_tdesc %arg0, %offsets : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %3 = xegpu.atomic_rmw "addf" %2, %mask, %1 : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
      gpu.return
    }
 }
}
