
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true' --cse %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
   gpu.module @module0 {
    //CHECK: func.func private @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v128f32(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.v16i1.v16i64.v128f32", linkage_type = <Import>>}
    //CHECK: func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    //CHECK: func.func private @llvm.genx.raw.send2.v128i32.i1.v16i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<128xi32>) -> vector<128xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v16i32", linkage_type = <Import>>}
    //CHECK: func.func private @llvm.genx.raw.send2.v64i32.v16i1.v16i64(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<64xi32>) -> vector<64xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.v16i1.v16i64", linkage_type = <Import>>}
    gpu.func @test_loadgather(%arg0: memref<128xf16>, %arg1: memref<16x16xf16>, %arg2: memref<128xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
         %offsets = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
         //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
         %mask = arith.constant dense<true> : vector<16xi1>

         //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<128xf16> -> index
         //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
         //CHECK: %[[cst_0:.*]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xi64>
         //CHECK: %[[r1:.*]] = vector.broadcast %[[r0]] : i64 to vector<16xi64>
         //CHECK: %[[r2:.*]] = arith.addi %[[r1]], %[[cst_0]] : vector<16xi64>
         %0 = xegpu.create_tdesc %arg0, %offsets : memref<128xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>

         //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
         //CHECK: %[[c4_i8:.*]] = arith.constant 4 : i8
         //CHECK: %[[c2_i8:.*]] = arith.constant 2 : i8
         //CHECK: %[[c15_i8:.*]] = arith.constant 15 : i8
         //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
         //CHECK: %[[c71447936_i32:.*]] = arith.constant 71447936 : i32
         //CHECK: %[[cst_1:.*]] = arith.constant dense<0> : vector<64xi32>
         //CHECK: %[[r3:.*]] = func.call @llvm.genx.raw.send2.v64i32.v16i1.v16i64(%[[c0_i8]], %[[c4_i8]], %[[cst]], %[[c2_i8]], %[[c4_i8]], %[[c15_i8]], %[[c0_i32]], %[[c71447936_i32]], %[[r2]], %[[cst_1]]) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<64xi32>) -> vector<64xi32>
         //CHECK: %[[r4:.*]] = vector.bitcast %[[r3]] : vector<64xi32> to vector<128xf16>
         %3 = xegpu.load %0, %mask {transpose} : !xegpu.tensor_desc<16x8xf16, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1> -> vector<8x16xf16>

         //CHECK: %[[intptr_2:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<16x16xf16> -> index
         //CHECK: %[[r5:.*]] = arith.index_castui %[[intptr_2]] : index to i64
         //CHECK: %[[cst_3:.*]] = arith.constant dense<0> : vector<8xi64>
         //CHECK: %[[r6:.*]] = vector.insert %[[r5]], %[[cst_3]] [0] : i64 into vector<8xi64>
         //CHECK: %[[r7:.*]] = vector.bitcast %[[r6]] : vector<8xi64> to vector<16xi32>
         //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
         //CHECK: %[[c15_i32:.*]] = arith.constant 15 : i32
         //CHECK: %[[r8:.*]] = vector.insert %[[c31_i32]], %[[r7]] [2] : i32 into vector<16xi32>
         //CHECK: %[[r9:.*]] = vector.insert %[[c15_i32]], %[[r8]] [3] : i32 into vector<16xi32>
         //CHECK: %[[r10:.*]] = vector.insert %[[c31_i32]], %[[r9]] [4] : i32 into vector<16xi32>
         //CHECK: %[[r11:.*]] = vector.insert %[[c0_i32]], %[[r10]] [5] : i32 into vector<16xi32>
         //CHECK: %[[r12:.*]] = vector.insert %[[c0_i32]], %[[r11]] [6] : i32 into vector<16xi32>
         //CHECK: %[[c3855_i32:.*]] = arith.constant 3855 : i32
         //CHECK: %[[r13:.*]] = vector.insert %[[c3855_i32]], %[[r12]] [7] : i32 into vector<16xi32>
         %1 = xegpu.create_nd_tdesc %arg1[0, 0] {boundary_check = true} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

         //CHECK: %[[true:.*]] = arith.constant true
         //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
         //CHECK: %[[c8_i8:.*]] = arith.constant 8 : i8
         //CHECK: %[[c42074755_i32:.*]] = arith.constant 42074755 : i32
         //CHECK: %[[cst_4:.*]] = arith.constant dense<0> : vector<128xi32>
         //CHECK: %[[r14:.*]] = func.call @llvm.genx.raw.send2.v128i32.i1.v16i32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c8_i8]], %[[c15_i8]], %[[c0_i32]], %[[c42074755_i32]], %[[r13]], %[[cst_4]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<128xi32>) -> vector<128xi32>
         //CHECK: %[[r15:.*]] = vector.bitcast %[[r14]] : vector<128xi32> to vector<256xf16>
         %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

         //CHECK: %[[c134744586_i32:.*]] = arith.constant 134744586 : i32
         //CHECK: %[[r16:.*]] = vector.bitcast %[[r4]] : vector<128xf16> to vector<64xi32>
         //CHECK: %[[r17:.*]] = vector.bitcast %[[r15]] : vector<256xf16> to vector<128xi32>
         //CHECK: %[[r18:.*]] = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%[[r17]], %[[r16]], %[[c134744586_i32]]) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
         %5 = xegpu.dpas %3, %4 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
         %offsets2 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>

         //CHECK: %[[intptr_5:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<128xf32> -> index
         //CHECK: %[[r19:.*]] = arith.index_castui %[[intptr_5]] : index to i64
         //CHECK: %[[cst_6:.*]] = arith.constant dense<[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480]> : vector<16xi64>
         //CHECK: %[[r20:.*]] = vector.broadcast %[[r19]] : i64 to vector<16xi64>
         //CHECK: %[[r21:.*]] = arith.addi %[[r20]], %[[cst_6]] : vector<16xi64>
         %2 = xegpu.create_tdesc %arg2, %offsets2 : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>

         //CHECK: %[[c67257732_i32:.*]] = arith.constant 67257732 : i32
         //CHECK: func.call @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v128f32(%c0_i8, %c4_i8, %cst, %c2_i8, %c8_i8, %c15_i8, %c0_i32, %c67257732_i32, %21, %18) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<128xf32>) -> ()
         xegpu.store %5, %2, %mask {transpose} : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1>

         //CHECK: gpu.return
         gpu.return
      }
   }
}
