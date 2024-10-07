// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s

#global = #xegpu.scatter_tdesc_attr<memory_scope=global>
#slm = #xegpu.scatter_tdesc_attr<memory_scope=slm>

gpu.module @test_kernel {
  //CHECK: gpu.func @test_store_scatter(%[[arg0:.*]]: memref<16xf16>) kernel
  gpu.func @test_store_scatter(%mem: memref<16xf16>) kernel {

    //CHECK: %[[cst:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00,
    //CHECK-SAME: 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01]> : vector<16xf16>
    %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : vector<16xf16>

    //CHECK: %[[cst_0:.*]] = arith.constant dense<true> : vector<16xi1>
    %mask = arith.constant dense<1> : vector<16xi1>

    //CHECK: %[[alloc:.*]] = memref.alloc() : memref<16xf16, 3>
    %slm = memref.alloc() : memref<16xf16, 3>

    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[alloc]] : memref<16xf16, 3> -> index
    //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i32
    //CHECK: %[[cst_1:.*]] = arith.constant dense<2> : vector<16xi32>
    //CHECK: %[[r1:.*]] = vector.from_elements {{.*}} : vector<16xindex>
    //CHECK: %[[r2:.*]] = arith.index_castui %[[r1]] : vector<16xindex> to vector<16xi32>
    //CHECK: %[[r3:.*]] = arith.muli %[[r2]], %[[cst_1]] : vector<16xi32>
    //CHECK: %[[r4:.*]] = vector.broadcast %[[r0]] : i32 to vector<16xi32>
    //CHECK: %[[r5:.*]] = arith.addi %[[r4]], %[[r3]] : vector<16xi32>
    %slm_tdesc = xegpu.create_tdesc %slm[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<16xf16, 3> -> !xegpu.tensor_desc<16xf16, #slm>

    //CHECK: %[[r6:.*]] = vector.bitcast %[[cst]] : vector<16xf16> to vector<16xi16>
    //CHECK: %[[r7:.*]] = arith.extui %[[r6]] : vector<16xi16> to vector<16xi32>
    //CHECK: %[[c4_i8:.*]] = arith.constant 4 : i8
    //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
    //CHECK: %[[c1_i16:.*]] = arith.constant 1 : i16
    //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
    //CHECK: %[[c6_i8:.*]] = arith.constant 6 : i8
    //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
    //CHECK: func.call @llvm.genx.lsc.store.slm.v16i1.v16i32.v16i32
    //CHECK-SAME: (%[[cst_0]], %[[c4_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r5]], %[[r7]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi32>, vector<16xi32>, i32) -> ()
    xegpu.store %cst, %slm_tdesc, %mask : vector<16xf16>, !xegpu.tensor_desc<16xf16, #slm>, vector<16xi1>

    //CHECK: %[[r8:.*]] = func.call @llvm.genx.lsc.load.slm.v16i32.v16i1.v16i32
    //CHECK-SAME: (%[[cst_0]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r5]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi32>, i32) -> vector<16xi32>
    //CHECK: %[[r9:.*]] = arith.trunci %[[r8]] : vector<16xi32> to vector<16xi16>
    //CHECK: %[[r10:.*]] = vector.bitcast %[[r9]] : vector<16xi16> to vector<16xf16>
    %data = xegpu.load %slm_tdesc, %mask : !xegpu.tensor_desc<16xf16, #slm>, vector<16xi1> -> vector<16xf16>

    //CHECK: %[[intptr_2:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<16xf16> -> index

    //CHECK: %[[r11:.*]] = arith.index_castui %[[intptr_2]] : index to i64
    //CHECK: %[[cst_3:.*]] = arith.constant dense<2> : vector<16xi64>

    //CHECK: %[[r12:.*]] = arith.index_castui %[[r1]] : vector<16xindex> to vector<16xi64>
    //CHECK: %[[r13:.*]] = arith.muli %[[r12]], %[[cst_3]] : vector<16xi64>
    //CHECK: %[[r14:.*]] = vector.broadcast %[[r11]] : i64 to vector<16xi64>
    //CHECK: %[[r15:.*]] = arith.addi %[[r14]], %[[r13]] : vector<16xi64>
    %tdesc = xegpu.create_tdesc %mem[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<16xf16> -> !xegpu.tensor_desc<16xf16, #global>

    //CHECK: %[[r16:.*]] = vector.bitcast %[[r10]] : vector<16xf16> to vector<16xi16>
    //CHECK: %[[r17:.*]] = arith.extui %[[r16]] : vector<16xi16> to vector<16xi32>
    //CHECK: func.call @llvm.genx.lsc.store.stateless.v16i1.v16i64.v16i32
    //CHECK-SAME: (%[[cst_0]], %[[c4_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r15]], %[[r17]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi64>, vector<16xi32>, i32) -> ()
    xegpu.store %data, %tdesc, %mask : vector<16xf16>, !xegpu.tensor_desc<16xf16, #global>, vector<16xi1>

    gpu.return
  }
}
