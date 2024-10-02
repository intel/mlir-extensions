// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s

#scatter = #xegpu.scatter_tdesc_attr<memory_scope=global>

gpu.module @test_kernel {
  //CHECK: gpu.func @test_copy(%[[arg0:.*]]: memref<16xf16>, %[[arg1:.*]]: memref<16xf16>) kernel
  gpu.func @test_copy(%a: memref<16xf16>, %b: memref<16xf16>) kernel {

    //CHECK: %[[mask:.*]] = arith.constant dense<true> : vector<16xi1>
    %mask = arith.constant dense<1> : vector<16xi1>

    //CHECK: %[[a_ptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<16xf16> -> index
    //CHECK: %[[r0:.*]] = arith.index_castui %[[a_ptr]] : index to i64
    //CHECK: %[[cst_0:.*]] = arith.constant dense<2> : vector<16xi64>
    //CHECK: %[[r1:.*]] = vector.from_elements {{.*}} : vector<16xindex>
    //CHECK: %[[r2:.*]] = arith.index_castui %[[r1]] : vector<16xindex> to vector<16xi64>
    //CHECK: %[[r3:.*]] = arith.muli %[[r2]], %[[cst_0]] : vector<16xi64>
    //CHECK: %[[r4:.*]] = vector.broadcast %[[r0]] : i64 to vector<16xi64>
    //CHECK: %[[r5:.*]] = arith.addi %[[r4]], %[[r3]] : vector<16xi64>
    %a_tdesc = xegpu.create_tdesc %a[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<16xf16> -> !xegpu.tensor_desc<16xf16, #scatter>

    //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
    //CHECK: %[[c1_i16:.*]] = arith.constant 1 : i16
    //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
    //CHECK: %[[c6_i8:.*]] = arith.constant 6 : i8
    //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
    //CHECK: %[[r6:.*]] = func.call @llvm.genx.lsc.load.stateless.v16i32.v16i1.v16i64
    //CHECK-SAME: (%[[mask]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r5]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi64>, i32) -> vector<16xi32>
    //CHECK: %[[r7:.*]] = arith.trunci %[[r6]] : vector<16xi32> to vector<16xi16>
    //CHECK: %[[r8:.*]] = vector.bitcast %[[r7]] : vector<16xi16> to vector<16xf16>
    %data = xegpu.load %a_tdesc, %mask : !xegpu.tensor_desc<16xf16, #scatter>, vector<16xi1> -> vector<16xf16>

    //CHECK: %[[b_ptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]] : memref<16xf16> -> index
    //CHECK: %[[r9:.*]] = arith.index_castui %[[b_ptr]] : index to i64
    //CHECK: %[[r10:.*]] = vector.broadcast %[[r9]] : i64 to vector<16xi64>
    //CHECK: %[[r11:.*]] = arith.addi %[[r10]], %[[r3]] : vector<16xi64>
    %b_tdesc = xegpu.create_tdesc %b[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<16xf16> -> !xegpu.tensor_desc<16xf16, #scatter>

    //CHECK: %[[r12:.*]] = vector.bitcast %[[r8]] : vector<16xf16> to vector<16xi16>
    //CHECK: %[[r13:.*]] = arith.extui %[[r12]] : vector<16xi16> to vector<16xi32>
    //CHECK: %[[c4_i8:.*]] = arith.constant 4 : i8
    //CHECK: func.call @llvm.genx.lsc.store.stateless.v16i1.v16i64.v16i32
    //CHECK-SAME: (%[[mask]], %[[c4_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r11]], %[[r13]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi64>, vector<16xi32>, i32) -> ()
    xegpu.store %data, %b_tdesc, %mask : vector<16xf16>, !xegpu.tensor_desc<16xf16, #scatter>, vector<16xi1>
    gpu.return
  }
}
