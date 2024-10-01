// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s

#scatter = #xegpu.scatter_tdesc_attr<memory_scope=global>
gpu.module @test_kernel {
  gpu.func @test_copy(%a: memref<16xf32>, %b: memref<16xf32>) kernel {

    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
    %mask = arith.constant dense<1> : vector<16xi1>

    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<16xf32> -> index
    //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
    //CHECK: %[[cst_0:.*]] = arith.constant dense<4> : vector<16xi64>
    //CHECK: %[[r1:.*]] = vector.from_elements {{.*}} : vector<16xindex>
    //CHECK: %[[r2:.*]] = arith.index_castui %[[r1]] : vector<16xindex> to vector<16xi64>
    //CHECK: %[[r3:.*]] = arith.muli %[[r2]], %[[cst_0]] : vector<16xi64>
    //CHECK: %[[r4:.*]] = vector.broadcast %[[r0]] : i64 to vector<16xi64>
    //CHECK: %[[r5:.*]] = arith.addi %[[r4]], %[[r3]] : vector<16xi64>
    %a_tdesc = xegpu.create_tdesc %a[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<16xf32> -> !xegpu.tensor_desc<16xf32, #scatter>

    //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
    //CHECK: %[[c1_i16:.*]] = arith.constant 1 : i16
    //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
    //CHECK: %[[c3_i8:.*]] = arith.constant 3 : i8
    //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8

    //CHECK: func.call @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64
    //CHECK-SAME: (%[[cst]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c3_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r5]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi64>, i32) -> ()
    xegpu.prefetch %a_tdesc : !xegpu.tensor_desc<16xf32, #scatter>
    %data = xegpu.load %a_tdesc, %mask : !xegpu.tensor_desc<16xf32, #scatter>, vector<16xi1> -> vector<16xf32>
    %b_tdesc = xegpu.create_tdesc %b[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<16xf32> -> !xegpu.tensor_desc<16xf32, #scatter>
    xegpu.store %data, %b_tdesc, %mask : vector<16xf32>, !xegpu.tensor_desc<16xf32, #scatter>, vector<16xi1>
    gpu.return
  }
}
