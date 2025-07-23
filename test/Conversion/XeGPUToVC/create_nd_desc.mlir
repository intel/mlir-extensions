// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s --check-prefixes=CHECK
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {

    // CHECK: gpu.func @test_create_nd_tdesc_0(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_0(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %cst = arith.constant dense<0> : vector<8xi64>
      //CHECK: %1 = vector.insert %0, %cst [0] : i64 into vector<8xi64>
      //CHECK: %2 = vector.bitcast %1 : vector<8xi64> to vector<16xi32>
      //CHECK: %c31_i32 = arith.constant 31 : i32
      //CHECK: %c7_i32 = arith.constant 7 : i32
      //CHECK: %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<16xi32>
      //CHECK: %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<16xi32>
      //CHECK: %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<16xi32>
      //CHECK: %c0_i32 = arith.constant 0 : i32
      //CHECK: %6 = vector.insert %c0_i32, %5 [5] : i32 into vector<16xi32>
      //CHECK: %7 = vector.insert %c0_i32, %6 [6] : i32 into vector<16xi32>
      //CHECK: %c1807_i32 = arith.constant 1807 : i32
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_1(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_1(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %c0 = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %cst = arith.constant dense<0> : vector<8xi64>
      //CHECK: %1 = vector.insert %0, %cst [0] : i64 into vector<8xi64>
      //CHECK: %2 = vector.bitcast %1 : vector<8xi64> to vector<16xi32>
      //CHECK: %c31_i32 = arith.constant 31 : i32
      //CHECK: %c7_i32 = arith.constant 7 : i32
      //CHECK: %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<16xi32>
      //CHECK: %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<16xi32>
      //CHECK: %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<16xi32>
      //CHECK: %c0_i32 = arith.constant 0 : i32
      //CHECK: %6 = arith.index_castui %c0 : index to i32
      //CHECK: %7 = vector.insert %c0_i32, %5 [5] : i32 into vector<16xi32>
      //CHECK: %8 = vector.insert %6, %7 [6] : i32 into vector<16xi32>
      //CHECK: %c1807_i32 = arith.constant 1807 : i32
      %0 = xegpu.create_nd_tdesc %arg0[%c0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      //CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_2(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_2(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %c0 = arith.constant 0 : index
      %c0 = arith.constant 0 : index

      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %cst = arith.constant dense<0> : vector<8xi64>
      //CHECK: %1 = vector.insert %0, %cst [0] : i64 into vector<8xi64>
      //CHECK: %2 = vector.bitcast %1 : vector<8xi64> to vector<16xi32>
      //CHECK: %c31_i32 = arith.constant 31 : i32
      //CHECK: %c7_i32 = arith.constant 7 : i32
      //CHECK: %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<16xi32>
      //CHECK: %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<16xi32>
      //CHECK: %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<16xi32>
      //CHECK: %6 = arith.index_castui %c0 : index to i32
      //CHECK: %c0_i32 = arith.constant 0 : i32
      //CHECK: %7 = vector.insert %6, %5 [5] : i32 into vector<16xi32>
      //CHECK: %8 = vector.insert %c0_i32, %7 [6] : i32 into vector<16xi32>
      //CHECK: %c1807_i32 = arith.constant 1807 : i32

      %0 = xegpu.create_nd_tdesc %arg0[0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      //CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_3(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_3(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %c0 = arith.constant 0 : index
      %c0 = arith.constant 0 : index

      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %cst = arith.constant dense<0> : vector<8xi64>
      //CHECK: %1 = vector.insert %0, %cst [0] : i64 into vector<8xi64>
      //CHECK: %2 = vector.bitcast %1 : vector<8xi64> to vector<16xi32>
      //CHECK: %c31_i32 = arith.constant 31 : i32
      //CHECK: %c7_i32 = arith.constant 7 : i32
      //CHECK: %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<16xi32>
      //CHECK: %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<16xi32>
      //CHECK: %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<16xi32>
      //CHECK: %6 = arith.index_castui %c0 : index to i32
      //CHECK: %7 = vector.insert %6, %5 [5] : i32 into vector<16xi32>
      //CHECK: %8 = vector.insert %6, %7 [6] : i32 into vector<16xi32>
      //CHECK: %c1807_i32 = arith.constant 1807 : i32
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      //CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_4(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_4(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c1 = arith.constant 1 : index

      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      //CHECK: %c32 = arith.constant 32 : index
      //CHECK: %0 = arith.addi %intptr, %c32 : index
      //CHECK: %1 = arith.index_castui %0 : index to i64
      //CHECK: %c2_i64 = arith.constant 2 : i64
      //CHECK: %2 = arith.addi %1, %c2_i64 : i64
      %0 = xegpu.create_nd_tdesc %arg0[%c1, %c1] : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
      //CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_1d_strided_memref(%[[arg0:.*]]: memref<32x32xf16, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_1d_strided_memref(%arg0: memref<32x32xf16, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<32x32xf16, strided<[64, 1]>> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      %tdesc_1d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf16, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<16xf16>
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_2d_strided_memref(%[[arg0:.*]]: memref<32x32xf16, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_2d_strided_memref(%arg0: memref<32x32xf16, strided<[64,1], offset: 0>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
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

      %tdesc_2d = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x32xf16, strided<[64,1], offset: 0>> -> !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_1d_dynamic_strided_memref(%[[arg0:.*]]: memref<?x?xf16, strided<[?, ?], offset: ?>>, %[[arg1:.*]]: index, %[[arg2:.*]]: index, %[[arg3:.*]]: index, %[[arg4:.*]]: index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_1d_dynamic_strided_memref(%arg0: memref<?x?xf16, strided<[?,?], offset: ?>>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?x?xf16, strided<[?, ?], offset: ?>> -> index
      //CHECK: %[[c2:.*]] = arith.constant 2 : index
      //CHECK: %[[r0:.*]] = arith.muli %[[arg3]], %[[c2]] : index
      //CHECK: %[[r1:.*]] = arith.muli %[[r0]], %[[c0]] : index
      //CHECK: %[[r2:.*]] = arith.addi %[[intptr]], %[[r1]] : index
      //CHECK: %[[r3:.*]] = arith.index_castui %[[r2]] : index to i64
      %tdesc_1d = xegpu.create_nd_tdesc %arg0[%c0, %c0], shape: [%arg1, %arg2], strides: [%arg3, %arg4] : memref<?x?xf16, strided<[?,?], offset: ?>> -> !xegpu.tensor_desc<16xf16>
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_2d_dynamic_strided_memref(%[[arg0:.*]]: memref<?x?xf16, strided<[?, ?], offset: ?>>, %[[arg1:.*]]: index, %[[arg2:.*]]: index, %[[arg3:.*]]: index, %[[arg4:.*]]: index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_2d_dynamic_strided_memref(%arg0: memref<?x?xf16, strided<[?,?], offset: ?>>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?x?xf16, strided<[?, ?], offset: ?>> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %intptr : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r1:.*]] = vector.insert %[[r0]], %cst [0] : i64 into vector<8xi64>
      //CHECK: %[[r2:.*]] = vector.bitcast %[[r1]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[r3:.*]] = arith.index_castui %arg2 : index to i32
      //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
      //CHECK: %[[r4:.*]] = arith.muli %[[r3]], %c2_i32 : i32
      //CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
      //CHECK: %[[r5:.*]] = arith.subi %[[r4]], %c1_i32 : i32
      //CHECK: %[[r6:.*]] = arith.index_castui %arg1 : index to i32
      //CHECK: %[[r7:.*]] = arith.subi %[[r6]], %c1_i32 : i32
      //CHECK: %[[r8:.*]] = arith.index_castui %arg3 : index to i32
      //CHECK: %[[r9:.*]] = arith.muli %[[r8]], %c2_i32 : i32
      //CHECK: %[[r10:.*]] = arith.subi %[[r9]], %c1_i32 : i32
      //CHECK: %[[r11:.*]] = vector.insert %[[r5]], %[[r2]] [2] : i32 into vector<16xi32>
      //CHECK: %[[r12:.*]] = vector.insert %[[r7]], %[[r11]] [3] : i32 into vector<16xi32>
      //CHECK: %[[r13:.*]] = vector.insert %[[r10]], %[[r12]] [4] : i32 into vector<16xi32>
      //CHECK: %[[r14:.*]] = arith.index_castui %c0 : index to i32
      //CHECK: %[[r15:.*]] = vector.insert %[[r14]], %[[r13]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r16:.*]] = vector.insert %[[r14]], %[[r15]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c1807_i32:.*]] = arith.constant 1807 : i32
      %tdesc_2d = xegpu.create_nd_tdesc %arg0[%c0, %c0], shape: [%arg1, %arg2], strides: [%arg3, %arg4] : memref<?x?xf16, strided<[?,?], offset: ?>> -> !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
}
