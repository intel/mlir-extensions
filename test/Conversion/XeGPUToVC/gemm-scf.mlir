// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s --check-prefixes=CHECK,LSC

module @gemm attributes {gpu.container_module} {
    gpu.module @test_kernel {
    gpu.func @test_kernel(%arg0: memref<1024x1016xf16>, %arg1: memref<1016x1016xf16>, %arg2: memref<1024x1016xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1024 = arith.constant 1024 : index
      %c1016 = arith.constant 1016 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = arith.muli %0, %c8 : index
      %3 = arith.muli %1, %c16 : index

      //CHECK: %[[C_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<1024x1016xf32> -> index
      //CHECK: %[[r2:.*]] = arith.index_castui %[[C_BASEPTR]] : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r3:.*]] = vector.insert %[[r2]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r4:.*]] = vector.bitcast %[[r3]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c4063_i32:.*]] = arith.constant 4063 : i32
      //CHECK: %[[c1023_i32:.*]] = arith.constant 1023 : i32
      //CHECK: %[[r5:.*]] = vector.insert %[[c4063_i32:.*]], %[[r4]] [2] : i32 into vector<16xi32>
      //CHECK: %[[r6:.*]] = vector.insert %[[c1023_i32:.*]], %[[r5]] [3] : i32 into vector<16xi32>
      //CHECK: %[[r7:.*]] = vector.insert %[[c4063_i32:.*]], %[[r6]] [4] : i32 into vector<16xi32>
      //CHECK: %[[r8:.*]] = arith.index_castui %{{.*}} : index to i32
      //CHECK: %[[r9:.*]] = arith.index_castui %{{.*}} : index to i32
      //CHECK: %[[r10:.*]] = vector.insert %[[r8]], %[[r7]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r11:.*]] = vector.insert %[[r9]], %[[r10]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c1807_i32:.*]] = arith.constant 1807 : i32
      //CHECK: %[[r12:.*]] = vector.insert %[[c1807_i32]], %[[r11]] [7] : i32 into vector<16xi32>
      %4 = xegpu.create_nd_tdesc  %arg2[%2, %3] : memref<1024x1016xf32> -> !xegpu.tensor_desc<8x16xf32>

      //LSC: %[[cst_0:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
      //LSC: %[[true:.*]] = arith.constant true
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[r13:.*]] = vector.from_elements %[[c0_i8]], %[[c0_i8]] : vector<2xi8>
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[c16_i8:.*]] = arith.constant 16 : i8
      //LSC: %[[c8_i8:.*]] = arith.constant 8 : i8
      //LSC: %[[c0_i32:.*]] = arith.constant 0 : i32
      //LSC: %[[CVALUE:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.v128f32.v2i8(%[[true]], %[[r13]], %[[c1_i8]], %[[c16_i8]], %[[c8_i8]], %[[r12]], %[[c0_i32]], %[[c0_i32]], %[[cst_0]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, vector<128xf32>) -> vector<128xf32>
      %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      // CHECK: %[[SCF_RESULT:.*]] = scf.for {{.*}} iter_args(%[[arg4:.*]] = %[[CVALUE]]) -> (vector<128xf32>)
      %6 = scf.for %arg3 = %c0 to %c1016 step %c16 iter_args(%arg4 = %5) -> (vector<8x16xf32>) {
        //CHECK: %[[intptr_1:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<1024x1016xf16> -> index
        //CHECK: %[[r18:.*]] = arith.index_castui %[[intptr_1]] : index to i64
        //CHECK: %[[r19:.*]] = vector.insert %[[r18]], %[[cst]] [0] : i64 into vector<8xi64>
        //CHECK: %[[r20:.*]] = vector.bitcast %[[r19]] : vector<8xi64> to vector<16xi32>
        //CHECK: %[[c2031_i32:.*]] = arith.constant 2031 : i32
        //CHECK: %[[r21:.*]] = vector.insert %[[c2031_i32]], %[[r20]] [2] : i32 into vector<16xi32>
        //CHECK: %[[r22:.*]] = vector.insert %[[c1023_i32]], %[[r21]] [3] : i32 into vector<16xi32>
        //CHECK: %[[r23:.*]] = vector.insert %[[c2031_i32]], %[[r22]] [4] : i32 into vector<16xi32>
        //CHECK: %[[r24:.*]] = arith.index_castui %{{.*}} : index to i32
        //CHECK: %[[r25:.*]] = vector.insert %[[r24]], %[[r23]] [5] : i32 into vector<16xi32>
        //CHECK: %[[r26:.*]] = vector.insert %[[r9]], %[[r25]] [6] : i32 into vector<16xi32>
        //CHECK: %[[r27:.*]] = vector.insert %[[c1807_i32]], %[[r26]] [7] : i32 into vector<16xi32>
        %7 = xegpu.create_nd_tdesc %arg0[%2, %arg3] : memref<1024x1016xf16> -> !xegpu.tensor_desc<8x16xf16>

        //CHECK: %[[intptr_2:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<1016x1016xf16> -> index
        //CHECK: %[[r28:.*]] = arith.index_castui %[[intptr_2]] : index to i64
        //CHECK: %[[r29:.*]] = vector.insert %[[r28]], %[[cst]] [0] : i64 into vector<8xi64>
        //CHECK: %[[r30:.*]] = vector.bitcast %[[r29]] : vector<8xi64> to vector<16xi32>
        //CHECK: %[[c1015_i32:.*]] = arith.constant 1015 : i32
        //CHECK: %[[r31:.*]] = vector.insert %[[c2031_i32]], %[[r30]] [2] : i32 into vector<16xi32>
        //CHECK: %[[r32:.*]] = vector.insert %[[c1015_i32]], %[[r31]] [3] : i32 into vector<16xi32>
        //CHECK: %[[r33:.*]] = vector.insert %[[c2031_i32]], %[[r32]] [4] : i32 into vector<16xi32>
        //CHECK: %[[r34:.*]] = vector.insert %[[r8]], %[[r33]] [5] : i32 into vector<16xi32>
        //CHECK: %[[r35:.*]] = vector.insert %[[r24]], %[[r34]] [6] : i32 into vector<16xi32>
        //CHECK: %[[c3855_i32:.*]] = arith.constant 3855 : i32
        //CHECK: %[[r36:.*]] = vector.insert %[[c3855_i32]], %[[r35]] [7] : i32 into vector<16xi32>
        %8 = xegpu.create_nd_tdesc %arg1[%arg3, %3] : memref<1016x1016xf16> -> !xegpu.tensor_desc<16x16xf16>

        //LSC: %[[cst_3:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
        //LSC: %[[r39:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.v128f16.v2i8(%[[true]], %[[r13]], %[[c1_i8]], %[[c16_i8]], %[[c8_i8]], %[[r27]], %[[c0_i32]], %[[c0_i32]], %[[cst_3]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
        %9 = xegpu.load_nd %7 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

        //LSC: %[[cst_4:.*]] = arith.constant dense<0.000000e+00> : vector<256xf16>
        //LSC: %[[r42:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v256f16.v2i8(%[[true]], %[[r13]], %[[c1_i8]], %[[c16_i8]], %[[c16_i8]], %[[r36]], %[[c0_i32]], %[[c0_i32]], %[[cst_4]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, vector<256xf16>) -> vector<256xf16>
        %10 = xegpu.load_nd %8 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        //CHECK: %[[r43:.*]] = vector.bitcast %[[r39]] : vector<128xf16> to vector<64xi32>
        //CHECK: %[[r44:.*]] = vector.bitcast %[[r42]] : vector<256xf16> to vector<128xi32>
        //CHECK: %[[c10_i32:.*]] = arith.constant 10 : i32
        //CHECK: %[[c8_i32:.*]] = arith.constant 8 : i32
        //CHECK: %[[DPAS_RES:.*]] = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%{{.*}}, %[[r44]], %[[r43]], {{.*}}) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %11 = xegpu.dpas %9, %10, %arg4 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // CHECK: scf.yield %[[DPAS_RES]] : vector<128xf32>
        scf.yield %11 : vector<8x16xf32>
      }

      // LSC: func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%[[true]], %[[r13]], %[[c1_i8]], %[[c16_i8]], %[[c8_i8]], %[[r12]], %[[c0_i32]], %[[c0_i32]], %[[SCF_RESULT]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      xegpu.store_nd %6, %4 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }

}
