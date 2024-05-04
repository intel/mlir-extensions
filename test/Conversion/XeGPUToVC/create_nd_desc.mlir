// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true' -cse %s | FileCheck %s --check-prefixes=CHECK
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false' -cse  %s | FileCheck %s --check-prefixes=CHECK
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {

    // CHECK: gpu.func @test_create_nd_tdesc_0(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_0(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      // CHECK: %cst = arith.constant dense<0> : vector<4xi64>
      // CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      // CHECK: %0 = arith.index_castui %intptr : index to i64
      // CHECK: %1 = vector.insert %0, %cst [0] : i64 into vector<4xi64>
      // CHECK: %2 = vector.bitcast %1 : vector<4xi64> to vector<8xi32>
      // CHECK: %c31_i32 = arith.constant 31 : i32
      // CHECK: %c7_i32 = arith.constant 7 : i32
      // CHECK: %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<8xi32>
      // CHECK: %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<8xi32>
      // CHECK: %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<8xi32>
      // CHECK: %c0_i32 = arith.constant 0 : i32
      // CHECK: %6 = vector.insert %c0_i32, %5 [5] : i32 into vector<8xi32>
      // CHECK: %7 = vector.insert %c0_i32, %6 [6] : i32 into vector<8xi32>
      // CHECK: %c1807_i32 = arith.constant 1807 : i32
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_1(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_1(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %c0 = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      //CHECK: %cst = arith.constant dense<0> : vector<4xi64>
      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %1 = vector.insert %0, %cst [0] : i64 into vector<4xi64>
      //CHECK: %2 = vector.bitcast %1 : vector<4xi64> to vector<8xi32>
      //CHECK: %c31_i32 = arith.constant 31 : i32
      //CHECK: %c7_i32 = arith.constant 7 : i32
      //CHECK: %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<8xi32>
      //CHECK: %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<8xi32>
      //CHECK: %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<8xi32>
      //CHECK: %c0_i32 = arith.constant 0 : i32
      //CHECK: %6 = arith.index_cast %c0 : index to i64
      //CHECK: %7 = arith.trunci %6 : i64 to i32
      //CHECK: %8 = vector.insert %c0_i32, %5 [5] : i32 into vector<8xi32>
      //CHECK: %9 = vector.insert %7, %8 [6] : i32 into vector<8xi32>
      //CHECK: %c1807_i32 = arith.constant 1807 : i32
      %0 = xegpu.create_nd_tdesc %arg0[%c0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      //CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_2(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_2(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %c0 = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      // %cst = arith.constant dense<0> : vector<4xi64>
      // %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      // %0 = arith.index_castui %intptr : index to i64
      // %1 = vector.insert %0, %cst [0] : i64 into vector<4xi64>
      // %2 = vector.bitcast %1 : vector<4xi64> to vector<8xi32>
      // %c31_i32 = arith.constant 31 : i32
      // %c7_i32 = arith.constant 7 : i32
      // %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<8xi32>
      // %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<8xi32>
      // %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<8xi32>
      // %6 = arith.index_cast %c0 : index to i64
      // %7 = arith.trunci %6 : i64 to i32
      // %c0_i32 = arith.constant 0 : i32
      // %8 = vector.insert %7, %5 [5] : i32 into vector<8xi32>
      // %9 = vector.insert %c0_i32, %8 [6] : i32 into vector<8xi32>
      // %c1807_i32 = arith.constant 1807 : i32
      %0 = xegpu.create_nd_tdesc %arg0[0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      //CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @test_create_nd_tdesc_3(%[[arg0:.*]]: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    gpu.func @test_create_nd_tdesc_3(%arg0: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      //CHECK: %c0 = arith.constant 0 : index
      %c0 = arith.constant 0 : index

      //CHECK: %cst = arith.constant dense<0> : vector<4xi64>
      //CHECK: %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      //CHECK: %0 = arith.index_castui %intptr : index to i64
      //CHECK: %1 = vector.insert %0, %cst [0] : i64 into vector<4xi64>
      //CHECK: %2 = vector.bitcast %1 : vector<4xi64> to vector<8xi32>
      //CHECK: %c31_i32 = arith.constant 31 : i32
      //CHECK: %c7_i32 = arith.constant 7 : i32
      //CHECK: %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<8xi32>
      //CHECK: %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<8xi32>
      //CHECK: %5 = vector.insert %c31_i32, %4 [4] : i32 into vector<8xi32>
      //CHECK: %6 = arith.index_cast %c0 : index to i64
      //CHECK: %7 = arith.trunci %6 : i64 to i32
      //CHECK: %8 = vector.insert %7, %5 [5] : i32 into vector<8xi32>
      //CHECK: %9 = vector.insert %7, %8 [6] : i32 into vector<8xi32>
      //CHECK: %c1807_i32 = arith.constant 1807 : i32
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      //CHECK: gpu.return
      gpu.return
    }
  }
}
