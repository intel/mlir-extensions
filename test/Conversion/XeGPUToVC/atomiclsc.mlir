// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {

  gpu.module @test_kernel {
    // CHECK: func.func private @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64(vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, vector<16xi1>, vector<16xi64>, vector<16xi32>, vector<16xi32>, i32, vector<16xi32>) -> vector<16xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64", linkage_type = <Import>>}
    gpu.func @test_atomiclsc(%arg0: memref<128xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

      //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
      //CHECK: %[[cst_0:.*]] = arith.constant dense<5.000000e-01> : vector<16xf32>
      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<128xf32> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst_1:.*]] = arith.constant dense<4> : vector<16xi64>
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[c1:.*]] = arith.constant 1 : index
      //CHECK: %[[c2:.*]] = arith.constant 2 : index
      //CHECK: %[[c3:.*]] = arith.constant 3 : index
      //CHECK: %[[c4:.*]] = arith.constant 4 : index
      //CHECK: %[[c5:.*]] = arith.constant 5 : index
      //CHECK: %[[c6:.*]] = arith.constant 6 : index
      //CHECK: %[[c7:.*]] = arith.constant 7 : index
      //CHECK: %[[c8:.*]] = arith.constant 8 : index
      //CHECK: %[[c9:.*]] = arith.constant 9 : index
      //CHECK: %[[c10:.*]] = arith.constant 10 : index
      //CHECK: %[[c11:.*]] = arith.constant 11 : index
      //CHECK: %[[c12:.*]] = arith.constant 12 : index
      //CHECK: %[[c13:.*]] = arith.constant 13 : index
      //CHECK: %[[c14:.*]] = arith.constant 14 : index
      //CHECK: %[[c15:.*]] = arith.constant 15 : index
      //CHECK: %[[r1:.*]] = vector.from_elements %[[c0]], %[[c1]], %[[c2]], %[[c3]], %[[c4]], %[[c5]], %[[c6]], %[[c7]], %[[c8]], %[[c9]], %[[c10]], %[[c11]], %[[c12]], %[[c13]], %[[c14]], %[[c15]] : vector<16xindex>
      //CHECK: %[[r2:.*]] = arith.index_castui %[[r1]] : vector<16xindex> to vector<16xi64>
      //CHECK: %[[r3:.*]] = arith.muli %[[r2]], %[[cst_1]] : vector<16xi64>
      //CHECK: %[[r4:.*]] = vector.broadcast %[[r0]] : i64 to vector<16xi64>
      //CHECK: %[[r5:.*]] = arith.addi %[[r4]], %[[r3]] : vector<16xi64>
      //CHECK: %[[c19_i8:.*]] = arith.constant 19 : i8
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[c1_i16:.*]] = arith.constant 1 : i16
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[c3_i8:.*]] = arith.constant 3 : i8
      //CHECK: %[[cst_2:.*]] = arith.constant dense<0> : vector<16xi32>
      //CHECK: %[[r6:.*]] = vector.bitcast %[[cst_0]] : vector<16xf32> to vector<16xi32>
      //CHECK: %[[r7:.*]] = func.call @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64(%[[cst]], %[[c19_i8]], %[[c1_i8]], %[[c1_i8]], %[[c1_i16]], %[[c0_i32]], %[[c3_i8]], %[[c1_i8]], %[[c1_i8]], %[[cst]], %[[r5]], %[[r6]], %[[cst_2]], %[[c0_i32]], %[[cst_2]]) : (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, vector<16xi1>, vector<16xi64>, vector<16xi32>, vector<16xi32>, i32, vector<16xi32>) -> vector<16xi32>

      %mask = arith.constant dense<true> : vector<16xi1>
      %1 = arith.constant dense<0.5> : vector<16xf32>
      %offsets = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %2 = xegpu.create_tdesc %arg0[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<128xf32> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %3 = xegpu.atomic_rmw "addf" %2, %mask, %1 : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
      gpu.return
    }
 }
}
