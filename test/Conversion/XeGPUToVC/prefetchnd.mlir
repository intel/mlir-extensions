// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true' --cse %s | FileCheck %s --check-prefixes=CHECK,RAW
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false' --cse %s | FileCheck %s --check-prefixes=CHECK,LSC
module @gemm attributes {gpu.container_module} {

  gpu.module @test_kernel {

    //RAW: func.func private @llvm.genx.raw.sends2.noresult.i1.v16i32.v128f32(i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v16i32.v128f32", linkage_type = <Import>>}
    //RAW: func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    //RAW: func.func private @llvm.genx.raw.send2.v128i32.i1.v16i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<128xi32>) -> vector<128xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v16i32", linkage_type = <Import>>}
    //RAW: func.func private @llvm.genx.raw.send2.v64i32.i1.v16i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<64xi32>) -> vector<64xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.i1.v16i32", linkage_type = <Import>>}
    //RAW: func.func private @llvm.genx.raw.send2.noresult.i1.v16i32(i8, i8, i1, i8, i8, i32, i32, vector<16xi32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.noresult.i1.v16i32", linkage_type = <Import>>}

    gpu.func @test_prefetch(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<8x16xf16> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r1:.*]] = vector.insert %[[r0:.*]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r2:.*]] = vector.bitcast %[[r1:.*]] : vector<8xi64> to vector<16xi32>
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

      //CHECK: %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %arg1 : memref<16x16xf16> -> index
      //CHECK: %[[r9:.*]] = arith.index_castui %[[intptr_0]] : index to i64
      //CHECK: %[[r10:.*]] = vector.insert %[[r9]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r11:.*]] = vector.bitcast %[[r10]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c15_i32:.*]] = arith.constant 15 : i32
      //CHECK: %[[r12:.*]] = vector.insert %[[c31_i32]], %[[r11]] [2] : i32 into vector<16xi32>
      //CHECK: %[[r13:.*]] = vector.insert %[[c15_i32]], %[[r12]] [3] : i32 into vector<16xi32>
      //CHECK: %[[r14:.*]] = vector.insert %[[c31_i32]], %[[r13]] [4] : i32 into vector<16xi32>
      //CHECK: %[[r15:.*]] = vector.insert %[[c0_i32]], %[[r14]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r16:.*]] = vector.insert %[[c0_i32]], %[[r15]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c3855_i32:.*]] = arith.constant 3855 : i32
      //CHECK: %[[r17:.*]] = vector.insert %[[c3855_i32]], %[[r16]] [7] : i32 into vector<16xi32>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

      //CHECK: %[[intptr_1:.*]] = memref.extract_aligned_pointer_as_index %arg2 : memref<8x16xf32> -> index
      //CHECK: %[[r18:.*]] = arith.index_castui %[[intptr_1]] : index to i64
      //CHECK: %[[r19:.*]] = vector.insert %[[r18]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r20:.*]] = vector.bitcast %[[r19]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c63_i32:.*]] = arith.constant 63 : i32
      //CHECK: %[[r21:.*]] = vector.insert %[[c63_i32]], %[[r20]] [2] : i32 into vector<16xi32>
      //CHECK: %[[r22:.*]] = vector.insert %[[c7_i32]], %[[r21]] [3] : i32 into vector<16xi32>
      //CHECK: %[[r23:.*]] = vector.insert %[[c63_i32]], %[[r22]] [4] : i32 into vector<16xi32>
      //CHECK: %[[r24:.*]] = vector.insert %[[c0_i32]], %[[r23]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r25:.*]] = vector.insert %[[c0_i32]], %[[r24]] [6] : i32 into vector<16xi32>
      //CHECK: %[[r26:.*]] = vector.insert %[[c1807_i32]], %[[r25]] [7] : i32 into vector<16xi32>
      %2 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

      //RAW: %[[c0_i8:.*]] = arith.constant 0 : i8
      //RAW: %[[true:.*]] = arith.constant true
      //RAW: %[[c1_i8:.*]] = arith.constant 1 : i8
      //RAW: %[[c15_i8:.*]] = arith.constant 15 : i8
      //RAW: %[[c33686019_i32:.*]] = arith.constant 33686019 : i32
      //RAW: func.call @llvm.genx.raw.send2.noresult.i1.v16i32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c15_i8]], %[[c0_i32]], %[[c33686019_i32]], %[[r8]]) : (i8, i8, i1, i8, i8, i32, i32, vector<16xi32>) -> ()
      //RAW: func.call @llvm.genx.raw.send2.noresult.i1.v16i32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c15_i8]], %[[c0_i32]], %[[c33686019_i32]], %[[r17]]) : (i8, i8, i1, i8, i8, i32, i32, vector<16xi32>) -> ()

      //LSC: %[[cst_2:.*]] = arith.constant 0.000000e+00 : f16
      //LSC: %[[true:.*]] = arith.constant true
      //LSC: %[[c0_i8:.*]] = arith.constant 0 : i8
      //LSC: %[[r27:.*]] = vector.from_elements %[[c0_i8]], %[[c0_i8]] : vector<2xi8>
      //LSC: %[[c1_i8:.*]] = arith.constant 1 : i8
      //LSC: %[[c16_i8:.*]] = arith.constant 16 : i8
      //LSC: %[[c8_i8:.*]] = arith.constant 8 : i8
      //LSC: func.call @llvm.genx.lsc.prefetch.2d.ugm.desc.v2i8(%[[true]], %[[r27]], %[[c1_i8]], %[[c16_i8]], %[[c8_i8]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[cst_2]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, f16) -> ()
      //LSC: func.call @llvm.genx.lsc.prefetch.2d.ugm.desc.v2i8(%[[true]], %[[r27]], %[[c1_i8]], %[[c16_i8]], %[[c16_i8]], %[[r17]], %[[c0_i32]], %[[c0_i32]], %[[cst_2]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, f16) -> ()
      xegpu.prefetch_nd %0 : !xegpu.tensor_desc<8x16xf16>
      xegpu.prefetch_nd %1 : !xegpu.tensor_desc<16x16xf16>

      //RAW: %[[c4_i8:.*]] = arith.constant 4 : i8
      //RAW: %[[c37880323_i32:.*]] = arith.constant 37880323 : i32
      //RAW: %[[cst_2:.*]] = arith.constant dense<0> : vector<64xi32>
      //RAW: %[[r27:.*]] = func.call @llvm.genx.raw.send2.v64i32.i1.v16i32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c4_i8]], %[[c15_i8]], %[[c0_i32]], %[[c37880323_i32]], %[[r8]], %[[cst_2]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<64xi32>) -> vector<64xi32>
      //RAW: %[[A:.*]] = vector.bitcast %[[r27]] : vector<64xi32> to vector<128xf16>

      //LSC: %[[cst_3:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
      //LSC: %[[A:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.v128f16.v2i8(%[[true]], %[[r27]], %[[c1_i8]], %[[c16_i8]], %[[c8_i8]], %[[r8]], %[[c0_i32]], %[[c0_i32]], %[[cst_3]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, vector<128xf16>) -> vector<128xf16>
      %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

      //RAW: %[[c8_i8:.*]] = arith.constant 8 : i8
      //RAW: %[[c42074755_i32:.*]] = arith.constant 42074755 : i32
      //RAW: %[[cst_3:.*]] = arith.constant dense<0> : vector<128xi32>
      //RAW: %[[r29:.*]] = func.call @llvm.genx.raw.send2.v128i32.i1.v16i32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c8_i8]], %[[c15_i8]], %[[c0_i32]], %[[c42074755_i32]], %[[r17]], %[[cst_3]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<128xi32>) -> vector<128xi32>
      //RAW: %[[B:.*]] = vector.bitcast %[[r29]] : vector<128xi32> to vector<256xf16>

      //LSC: %[[cst_4:.*]] = arith.constant dense<0.000000e+00> : vector<256xf16>
      //LSC: %[[B:.*]] = func.call @llvm.genx.lsc.load.2d.ugm.desc.vnni.v256f16.v2i8(%[[true]], %[[r27]], %[[c1_i8]], %[[c16_i8]], %[[c16_i8]], %[[r17]], %[[c0_i32]], %[[c0_i32]], %[[cst_4]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, vector<256xf16>) -> vector<256xf16>
      %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

      //CHECK: %[[c134744586_i32:.*]] = arith.constant 134744586 : i32
      //CHECK: %[[r31:.*]] = vector.bitcast %[[A]] : vector<128xf16> to vector<64xi32>
      //CHECK: %[[r32:.*]] = vector.bitcast %[[B]] : vector<256xf16> to vector<128xi32>
      //CHECK: %[[DPAS:.*]] = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%[[r32]], %[[r31]], %[[c134744586_i32]]) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
      %5 = xegpu.dpas %3, %4 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>

      //RAW: %[[c33686535_i32:.*]] = arith.constant 33686535 : i32
      //RAW: func.call @llvm.genx.raw.sends2.noresult.i1.v16i32.v128f32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c8_i8]], %[[c15_i8]], %[[c0_i32]], %[[c33686535_i32]], %[[r26]], %[[DPAS]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<128xf32>) -> ()

      //LSC: func.call @llvm.genx.lsc.store.2d.ugm.desc.v2i8.v128f32(%[[true]], %[[r27]], %[[c1_i8]], %[[c16_i8]], %[[c8_i8]], %[[r26]], %[[c0_i32]], %[[c0_i32]], %[[DPAS]]) : (i1, vector<2xi8>, i8, i8, i8, vector<16xi32>, i32, i32, vector<128xf32>) -> ()
      xegpu.store_nd %5, %2 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      //CHECK: gpu.return
      gpu.return
    }

 }
}
