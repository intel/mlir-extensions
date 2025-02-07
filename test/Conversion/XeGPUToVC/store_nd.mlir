// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {

    // CHECK: gpu.func @test_store_nd(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_store_nd(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      // CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : vector<8x16xf16>
      // CHECK: %[[data:.*]] = vector.shape_cast %[[cst]] : vector<8x16xf16> to vector<128xf16>

      %c = arith.constant dense<1.0> : vector<8x16xf16>

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<8x16xf16> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r1:.*]] = vector.insert %[[r0]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r2:.*]] = vector.bitcast %[[r1]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
      //CHECK: %[[c7_i32:.*]] = arith.constant 7 : i32
      //CHECK: %[[r3:.*]] = vector.insert %[[c31_i32:.*]], %[[r2]] [2] : i32 into vector<16xi32>
      //CHECK: %[[r4:.*]] = vector.insert %[[c7_i32:.*]], %[[r3]] [3] : i32 into vector<16xi32>
      //CHECK: %[[r5:.*]] = vector.insert %[[c31_i32:.*]], %[[r4]] [4] : i32 into vector<16xi32>
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[r6:.*]] = vector.insert %[[c0_i32]], %[[r5]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r7:.*]] = vector.insert %[[c0_i32]], %[[r6]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c1807_i32:.*]] = arith.constant 1807 : i32
      //CHECK: %[[r8:.*]] = vector.insert %[[c1807_i32]], %[[r7]] [7] : i32 into vector<16xi32>
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>

      //CHECK: %[[true:.*]] = arith.constant true
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: %[[r9:.*]] = vector.from_elements %[[c0_i8]], %[[c0_i8]] : vector<2xi8>
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[c16_i16:.*]] = arith.constant 16 : i16
      //CHECK: %[[c8_i16:.*]] = arith.constant 8 : i16
      //CHECK: func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f16(%[[true]], %[[r9]], %[[c1_i8]], %[[c16_i16]], %[[c8_i16]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[data]]) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> ()
      xegpu.store_nd %c, %0 : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_store_nd_1d_strided_memref(%[[arg0:.*]]: memref<32x32xf32, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_store_nd_1d_strided_memref(%arg0: memref<32x32xf32, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : vector<16xf32>
      %c1 = arith.constant dense<1.0> : vector<16xf32>

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<32x32xf32, strided<[64, 1]>> -> index
      //CHECK: %[[R0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[R1:.*]] = vector.broadcast %[[R0]] : i64 to vector<1xi64>
      %tdesc_1d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf32, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<16xf32>

      //CHECK: %[[cst_0:.*]] = arith.constant dense<true> : vector<1xi1>
      //CHECK: %[[c4_i8:.*]] = arith.constant 4 : i8
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: %[[c1_i16:.*]] = arith.constant 1 : i16
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[c3_i8:.*]] = arith.constant 3 : i8
      //CHECK: %[[c6_i8:.*]] = arith.constant 6 : i8
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: func.call @llvm.genx.lsc.store.stateless.v1i1.v1i64.v16f32(%[[cst_0]], %[[c4_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c3_i8]], %[[c6_i8]], %[[c1_i8]], %[[c0_i8]], %[[R1]], %[[cst]], %[[c0_i32]]) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi64>, vector<16xf32>, i32) -> ()
      xegpu.store_nd %c1, %tdesc_1d : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }

    // CHECK: gpu.func @test_store_nd_2d_strided_memref(%[[arg0:.*]]: memref<32x32xf16, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_store_nd_2d_strided_memref(%arg0: memref<32x32xf16, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : vector<8x16xf16>
      //CHECK: %[[R0:.*]] = vector.shape_cast %[[cst]] : vector<8x16xf16> to vector<128xf16>
      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<32x32xf16, strided<[64, 1]>> -> index
      //CHECK: %[[R1:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst_0:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[R2:.*]] = vector.insert %[[R1]], %[[cst_0]] [0] : i64 into vector<8xi64>
      //CHECK: %[[R3:.*]] = vector.bitcast %[[R2]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c63_i32:.*]] = arith.constant 63 : i32
      //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
      //CHECK: %[[c127_i32:.*]] = arith.constant 127 : i32
      //CHECK: %[[R4:.*]] = vector.insert %[[c63_i32]], %[[R3]] [2] : i32 into vector<16xi32>
      //CHECK: %[[R5:.*]] = vector.insert %[[c31_i32]], %[[R4]] [3] : i32 into vector<16xi32>
      //CHECK: %[[R6:.*]] = vector.insert %[[c127_i32]], %[[R5]] [4] : i32 into vector<16xi32>
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[R7:.*]] = vector.insert %[[c0_i32]], %[[R6]] [5] : i32 into vector<16xi32>
      //CHECK: %[[R8:.*]] = vector.insert %[[c0_i32]], %[[R7]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c1807_i32:.*]] = arith.constant 1807 : i32
      //CHECK: %[[R9:.*]] = vector.insert %[[c1807_i32]], %[[R8]] [7] : i32 into vector<16xi32>
      //CHECK: %[[true:.*]] = arith.constant true
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: %[[R10:.*]] = vector.from_elements %[[c0_i8]], %[[c0_i8]] : vector<2xi8>
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[c16_i16:.*]] = arith.constant 16 : i16
      //CHECK: %[[c8_i16:.*]] = arith.constant 8 : i16
      //CHECK: func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f16(%[[true]], %[[R10]], %[[c1_i8]], %[[c16_i16]], %[[c8_i16]], %[[R9]], %[[c0_i32]], %[[c0_i32]], %[[R0]]) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> ()
      %c2 = arith.constant dense<1.0> : vector<8x16xf16>
      %tdesc_2d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf16, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %c2, %tdesc_2d : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
}
