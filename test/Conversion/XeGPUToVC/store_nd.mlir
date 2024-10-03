// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s --check-prefixes=CHECK,LSC
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {

    // CHECK: gpu.func @test_store_nd(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_store_nd(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      // CHECK: %[[data:.*]] = arith.constant dense<1.000000e+00> : vector<8x16xf16>
      %c = arith.constant dense<1.0> : vector<8x16xf16>

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<8x16xf16> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r1:.*]] = vector.insert %[[r0]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r2:.*]] = vector.bitcast %[[r1]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
      //CHECK: %[[c7_i32:.*]] = arith.constant 7 : i32
      //CHECK: %[[r3:.*]] = vector.insert %[[c31_i32:.*]], %2 [2] : i32 into vector<16xi32>
      //CHECK: %[[r4:.*]] = vector.insert %[[c7_i32:.*]], %3 [3] : i32 into vector<16xi32>
      //CHECK: %[[r5:.*]] = vector.insert %[[c31_i32:.*]], %4 [4] : i32 into vector<16xi32>
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[r6:.*]] = vector.insert %[[c0_i32]], %[[r5]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r7:.*]] = vector.insert %[[c0_i32]], %[[r6]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c1807_i32:.*]] = arith.constant 1807 : i32
      //CHECK: %[[r8:.*]] = vector.insert %[[c1807_i32]], %[[r7]] [7] : i32 into vector<16xi32>
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>

      //LSC: %[[true:.*]] = arith.constant true
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[r9:.*]] = vector.from_elements %c0_i8, %c0_i8 : vector<2xi8>
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[c16_i16:.*]] = arith.constant 16 : i16
      //LSC: %[[c8_i16:.*]] = arith.constant 8 : i16
      //LSC: func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f16(%[[true]], %[[r9]], %[[c1_i8]], %[[c16_i16]], %[[c8_i16]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[data]]) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<8x16xf16>) -> ()
      xegpu.store_nd %c, %0 : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_store_nd_1d_strided_memref(%[[arg0:.*]]: memref<32x32xf32, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_store_nd_1d_strided_memref(%arg0: memref<32x32xf32, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %cst = arith.constant dense<1.000000e+00> : vector<16xf32>
      %c1 = arith.constant dense<1.0> : vector<16xf32>

      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<32x32xf32, strided<[64, 1]>> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %1 = vector.broadcast %0 : i64 to vector<1xi64>
      %tdesc_1d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf32, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<16xf32>

      //LSC: %cst_0 = arith.constant dense<true> : vector<1xi1>
      //LSC: %c4_i8 = arith.constant 4 : i8
      //LSC: %c0_i8 = arith.constant 0 : i8
      //LSC: %c1_i16 = arith.constant 1 : i16
      //LSC: %c0_i32 = arith.constant 0 : i32
      //LSC: %c3_i8 = arith.constant 3 : i8
      //LSC: %c6_i8 = arith.constant 6 : i8
      //LSC: %c1_i8 = arith.constant 1 : i8
      //LSC: func.call @llvm.genx.lsc.store.stateless.v1i1.v1i64.v16f32(%cst_0, %c4_i8, %c0_i8, %c0_i8, %c1_i16, %c0_i32, %c3_i8, %c6_i8, %c1_i8, %c0_i8, %1, %cst, %c0_i32) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi64>, vector<16xf32>, i32) -> ()
      xegpu.store_nd %c1, %tdesc_1d : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }

    // CHECK: gpu.func @test_store_nd_2d_strided_memref(%[[arg0:.*]]: memref<32x32xf16, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_store_nd_2d_strided_memref(%arg0: memref<32x32xf16, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %cst = arith.constant dense<1.000000e+00> : vector<8x16xf16>
      %c2 = arith.constant dense<1.0> : vector<8x16xf16>

      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<32x32xf16, strided<[64, 1]>> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %cst_0 = arith.constant dense<0> : vector<8xi64>
      //CHECK: %1 = vector.insert %0, %cst_0 [0] : i64 into vector<8xi64>
      //CHECK: %2 = vector.bitcast %1 : vector<8xi64> to vector<16xi32>
      //CHECK: %c63_i32 = arith.constant 63 : i32
      //CHECK: %c31_i32 = arith.constant 31 : i32
      //CHECK: %c127_i32 = arith.constant 127 : i32
      //CHECK: %3 = vector.insert %c63_i32, %2 [2] : i32 into vector<16xi32>
      //CHECK: %4 = vector.insert %c31_i32, %3 [3] : i32 into vector<16xi32>
      //CHECK: %5 = vector.insert %c127_i32, %4 [4] : i32 into vector<16xi32>
      //CHECK: %c0_i32 = arith.constant 0 : i32
      //CHECK: %6 = vector.insert %c0_i32, %5 [5] : i32 into vector<16xi32>
      //CHECK: %7 = vector.insert %c0_i32, %6 [6] : i32 into vector<16xi32>
      //CHECK: %c1807_i32 = arith.constant 1807 : i32
      //CHECK: %8 = vector.insert %c1807_i32, %7 [7] : i32 into vector<16xi32>
      %tdesc_2d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf16, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<8x16xf16>

      //LSC: %true = arith.constant true
      //LSC: %c0_i8 = arith.constant 0 : i8
      //LSC: %9 = vector.from_elements %c0_i8, %c0_i8 : vector<2xi8>
      //LSC: %c1_i8 = arith.constant 1 : i8
      //LSC: %c16_i16 = arith.constant 16 : i16
      //LSC: %c8_i16 = arith.constant 8 : i16
      //LSC: func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f16(%true, %9, %c1_i8, %c16_i16, %c8_i16, %8, %c0_i32, %c0_i32, %cst) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<8x16xf16>) -> ()
      xegpu.store_nd %c2, %tdesc_2d : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
}
