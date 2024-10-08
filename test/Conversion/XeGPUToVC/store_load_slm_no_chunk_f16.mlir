// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s

#global = #xegpu.scatter_tdesc_attr<memory_space=global>
#slm = #xegpu.scatter_tdesc_attr<memory_space=slm>

gpu.module @test_kernel {
  //CHECK: gpu.func @test_store_scatter(%[[arg0:.*]]: memref<16xf16>) kernel
  gpu.func @test_store_scatter(%mem: memref<16xf16>) kernel {

    //CHECK: %[[cst:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00,
    //CHECK-SAME: 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01]> : vector<16xf16>
    %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : vector<16xf16>

    //CHECK: %[[cst_0:.*]] = arith.constant dense<true> : vector<16xi1>
    %mask = arith.constant dense<1> : vector<16xi1>

    %offsets = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>

    //CHECK: %[[alloc:.*]] = memref.alloc() : memref<16xf16, 3>
    %slm = memref.alloc() : memref<16xf16, 3>

    //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[alloc]] : memref<16xf16, 3> -> index
    //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i32
    //CHECK: %[[cst_1:.*]] = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]> : vector<16xi32>
    //CHECK: %[[r1:.*]] = vector.broadcast %[[r0]] : i32 to vector<16xi32>
    //CHECK: %[[r2:.*]] = arith.addi %[[r1]], %[[cst_1]] : vector<16xi32>
    %slm_tdesc = xegpu.create_tdesc %slm, %offsets : memref<16xf16, 3>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #slm>

    //CHECK: %[[r3:.*]] = vector.bitcast %[[cst]] : vector<16xf16> to vector<16xi16>
    //CHECK: %[[r4:.*]] = arith.extui %[[r3]] : vector<16xi16> to vector<16xi32>
    //CHECK: %[[c4_i8:.*]] = arith.constant 4 : i8
    //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
    //CHECK: %[[c1_i16:.*]] = arith.constant 1 : i16
    //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
    //CHECK: %[[c6_i8:.*]] = arith.constant 6 : i8
    //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
    //CHECK: func.call @llvm.genx.lsc.store.slm.v16i1.v16i32.v16i32
    //CHECK-SAME: (%[[cst_0]], %[[c4_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r2]], %[[r4]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi32>, vector<16xi32>, i32) -> ()
    xegpu.store %cst, %slm_tdesc, %mask : vector<16xf16>, !xegpu.tensor_desc<16xf16, #slm>, vector<16xi1>

    //CHECK: %[[r5:.*]] = func.call @llvm.genx.lsc.load.slm.v16i32.v16i1.v16i32
    //CHECK-SAME: (%[[cst_0]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r2]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi32>, i32) -> vector<16xi32>
    //CHECK: %[[r6:.*]] = arith.trunci %[[r5]] : vector<16xi32> to vector<16xi16>
    //CHECK: %[[r7:.*]] = vector.bitcast %[[r6]] : vector<16xi16> to vector<16xf16>
    %data = xegpu.load %slm_tdesc, %mask : !xegpu.tensor_desc<16xf16, #slm>, vector<16xi1> -> vector<16xf16>

    //CHECK: %[[intptr_2:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]] : memref<16xf16> -> index

    //CHECK: %[[r8:.*]] = arith.index_castui %[[intptr_2]] : index to i64
    //CHECK: %[[cst_3:.*]] = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]> : vector<16xi64>
    //CHECK: %[[r9:.*]] = vector.broadcast %[[r8]] : i64 to vector<16xi64>
    //CHECK: %[[r10:.*]] = arith.addi %[[r9]], %[[cst_3]] : vector<16xi64>
    %tdesc = xegpu.create_tdesc %mem, %offsets : memref<16xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #global>

    //CHECK: %[[r11:.*]] = vector.bitcast %[[r7]] : vector<16xf16> to vector<16xi16>
    //CHECK: %[[r12:.*]] = arith.extui %[[r11]] : vector<16xi16> to vector<16xi32>
    //CHECK: func.call @llvm.genx.lsc.store.stateless.v16i1.v16i64.v16i32
    //CHECK-SAME: (%[[cst_0]], %[[c4_i8]], %[[c0_i8]], %[[c0_i8]], %[[c1_i16]], %[[c0_i32]], %[[c6_i8]], %[[c1_i8]], %[[c1_i8]], %[[c0_i8]], %[[r10]], %[[r12]], %[[c0_i32]])
    //CHECK-SAME: (vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi64>, vector<16xi32>, i32) -> ()
    xegpu.store %data, %tdesc, %mask : vector<16xf16>, !xegpu.tensor_desc<16xf16, #global>, vector<16xi1>

    gpu.return
  }
}
