// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s --check-prefixes=CHECK,LSC
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {

    // CHECK: gpu.func @test_load_nd_0(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_load_nd_0(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

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

      //LSC: %[[cst_0:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
      //LSC: %[[true:.*]] = arith.constant true
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[r9:.*]] = vector.from_elements %[[c0_i8]], %[[c0_i8]] : vector<2xi8>
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[c16_i16:.*]] = arith.constant 16 : i16
      //LSC: %[[c8_i16:.*]] = arith.constant 8 : i16
      //LSC: %[[r10:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.v128f16.v2i8(%[[true]], %[[r9]], %[[c1_i8]], %[[c16_i16]], %[[c8_i16]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[cst_0]]) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_load_nd_subbyte(%[[arg0:.*]]: memref<8x256xi1>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_load_nd_subbyte(%arg0: memref<8x256xi1>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x256xi1> -> !xegpu.tensor_desc<8x256xi1>
      // CHECK: %[[V10:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.v256i8.v2i8({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<256xi8>) -> vector<256xi8>
      // CHECK: %[[V11:.*]] = vector.bitcast %[[V10]] : vector<256xi8> to vector<2048xi1>
      // CHECK: %[[V12:.*]] = vector.shape_cast %[[V11]] : vector<2048xi1> to vector<8x256xi1>
      %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x256xi1> -> vector<8x256xi1>
      %cst0 = arith.constant 0 : index
      vector.store %1, %arg0[%cst0, %cst0] : memref<8x256xi1>, vector<8x256xi1>
      gpu.return
    }

    // CHECK: gpu.func @test_load_nd_1(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_load_nd_1(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
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
      //CHECK: %[[c1807_i32]] = arith.constant 1807 : i32
      //CHECK: %[[r8:.*]] = vector.insert %[[c1807_i32]], %[[r7]] [7] : i32 into vector<16xi32>
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>

      //LSC: %[[cst_0:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
      //LSC: %[[true:.*]] = arith.constant true
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[r9:.*]] = vector.from_elements %[[c0_i8]], %[[c0_i8]] : vector<2xi8>
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[c16_i16:.*]] = arith.constant 16 : i16
      //LSC: %[[c8_i16:.*]] = arith.constant 8 : i16
      //LSC: %[[r10:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v128f16.v2i8(%[[true]], %[[r9]], %[[c1_i8]], %[[c16_i16]], %[[c8_i16]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[cst_0]]) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %1 = xegpu.load_nd %0 <{packed}> : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
      // CHECK: gpu.return
      gpu.return
    }


    // CHECK: gpu.func @test_load_nd_2(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_load_nd_2(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<8x16xf16> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r1:.*]] = vector.insert %[[r0]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r2:.*]] = vector.bitcast %[[r1]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
      //CHECK: %[[c7_i32:.*]] = arith.constant 7 : i32
      //CHECK: %[[r3:.*]] = vector.insert %[[c31_i32]], %2 [2] : i32 into vector<16xi32>
      //CHECK: %[[r4:.*]] = vector.insert %[[c7_i32]], %3 [3] : i32 into vector<16xi32>
      //CHECK: %[[r5:.*]] = vector.insert %[[c31_i32]], %4 [4] : i32 into vector<16xi32>
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[r6:.*]] = vector.insert %[[c0_i32]], %[[r5]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r7:.*]] = vector.insert %[[c0_i32]], %[[r6]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c1807_i32:.*]] = arith.constant 1807 : i32
      //CHECK: %[[r8:.*]] = vector.insert %[[c1807_i32]], %[[r7]] [7] : i32 into vector<16xi32>
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>

      //LSC: %[[cst_0:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
      //LSC: %[[true:.*]] = arith.constant true
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[r9:.*]] = vector.from_elements %c0_i8, %c0_i8 : vector<2xi8>
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[c16_i16:.*]] = arith.constant 16 : i16
      //LSC: %[[c8_i16:.*]] = arith.constant 8 : i16
      //LSC: %[[r10:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.v128f16.v2i8(%[[true]], %[[r9]], %[[c1_i8]], %[[c16_i16]], %[[c8_i16]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[cst_0]]) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_load_nd_1d_strided_memref(%[[arg0:.*]]: memref<32x32xf32, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_load_nd_1d_strided_memref(%arg0: memref<32x32xf32, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<32x32xf32, strided<[64, 1]>> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[r1:.*]] = vector.broadcast %[[r0]] : i64 to vector<1xi64>

      //LSC: %[[cst:.*]] = arith.constant dense<true> : vector<1xi1>
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[c1_i16:.*]] = arith.constant 1 : i16
      //LSC: %[[c0_i32:.*]] = arith.constant 0 : i32
      //LSC: %[[c3_i8:.*]] = arith.constant 3 : i8
      //LSC: %[[c6_i8:.*]] = arith.constant 6 : i8
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[r2:.*]] = func.call @llvm.genx.lsc.load.stateless.v16f32.v1i1.v1i64(%[[cst]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c3_i8]], %[[c6_i8]], %[[c1_i8]], %[[c0_i8]], %[[r1]], %[[c0_i32]]) : (vector<1xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<1xi64>, i32) -> vector<16xf32>

      %tdesc_1d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf32, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<16xf32>
      %load_1d = xegpu.load_nd %tdesc_1d  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      gpu.return
    }

    // CHECK: gpu.func @test_load_nd_2d_strided_memref(%[[arg0:.*]]: memref<32x32xf16, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_load_nd_2d_strided_memref(%arg0: memref<32x32xf16, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<32x32xf16, strided<[64, 1]>> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r1:.*]] = vector.insert %[[r0]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r2:.*]] = vector.bitcast %[[r1]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c63_i32:.*]] = arith.constant 63 : i32
      //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
      //CHECK: %[[c127_i32:.*]] = arith.constant 127 : i32
      //CHECK: %[[r3:.*]] = vector.insert %[[c63_i32]], %[[r2]] [2] : i32 into vector<16xi32>
      //CHECK: %[[r4:.*]] = vector.insert %[[c31_i32]], %[[r3]] [3] : i32 into vector<16xi32>
      //CHECK: %[[r5:.*]] = vector.insert %[[c127_i32]], %[[r4]] [4] : i32 into vector<16xi32>
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[r6:.*]] = vector.insert %[[c0_i32]], %[[r5]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r7:.*]] = vector.insert %[[c0_i32]], %[[r6]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c1807_i32:.*]] = arith.constant 1807 : i32
      //CHECK: %[[r8:.*]] = vector.insert %[[c1807_i32]], %[[r7]] [7] : i32 into vector<16xi32>
      %tdesc_2d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf16, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<8x16xf16>

      //LSC: %[[cst_0:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
      //LSC: %[[true:.*]] = arith.constant true
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[r9:.*]] = vector.from_elements %c0_i8, %c0_i8 : vector<2xi8>
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[c16_i16:.*]] = arith.constant 16 : i16
      //LSC: %[[c8_i16:.*]] = arith.constant 8 : i16
      //LSC: %[[r10:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.v128f16.v2i8(%[[true]], %[[r9]], %[[c1_i8]], %[[c16_i16]], %[[c8_i16]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[cst_0]]) : (i1, vector<2xi8>, i8, i16, i16, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %load_2d = xegpu.load_nd %tdesc_2d : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      gpu.return
    }
  }
}
