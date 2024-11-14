// Tests ops on nd vectors that should be linearized.

// RUN: imex-opt -convert-xegpu-to-vc -split-input-file %s | FileCheck %s --check-prefixes=CHECK
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {

    // CHECK-LABEL: gpu.func @test_index_cast
    // CHECK: %[[c1024:.*]] = arith.constant 1024 : i32
    // CHECK: %[[bid:.*]] = gpu.block_id x
    // CHECK: %[[cst:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
    // CHECK: %[[r0:.*]] = arith.index_cast %[[bid]] : index to i32
    // CHECK: %[[r1:.*]] = arith.muli %[[r0]], %[[c1024]] : i32
    // CHECK: %[[r2:.*]] = vector.splat %[[r1]] : vector<16xi32>
    // CHECK: %[[r3:.*]] = arith.addi %[[r2]], %[[r2]] : vector<16xi32>
    // CHECK: %[[r4:.*]] = arith.addi %[[r2]], %[[cst]] : vector<16xi32>
    // CHECK: %[[r5:.*]] = arith.index_cast %[[r3]] : vector<16xi32> to vector<16xindex>
    // CHECK: %[[r6:.*]] = arith.index_cast %[[r4]] : vector<16xi32> to vector<16xindex>
    // CHECK-NEXT: gpu.return
    gpu.func @test_index_cast() kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c1024_i32 = arith.constant 1024 : i32
      %block_id_x = gpu.block_id  x
      %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
      %23 = arith.index_cast %block_id_x : index to i32
      %24 = arith.muli %23, %c1024_i32 : i32
      %25 = vector.splat %24 : vector<16xi32>
      %26 = arith.addi %25, %25 : vector<16xi32>
      %27 = vector.shape_cast %26 : vector<16xi32> to vector<1x16xi32>
      %28 = arith.addi %25, %cst_0 : vector<16xi32>
      %29 = vector.shape_cast %28 : vector<16xi32> to vector<1x16xi32>
      %30 = arith.index_cast %27 : vector<1x16xi32> to vector<1x16xindex>
      %31 = arith.index_cast %29 : vector<1x16xi32> to vector<1x16xindex>

      gpu.return
    }
  }
}
