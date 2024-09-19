// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true' --cse  %s | FileCheck %s --check-prefixes=CHECK,RAW
module @gemm attributes {gpu.container_module} {

  gpu.module @test_kernel {
    // CHECK: gpu.func @test_nd(%[[arg0:.*]]: memref<8x16xf16>, %[[arg1:.*]]: memref<16x16xf16>, %[[arg2:.*]]: memref<8x16xf32>)
    gpu.func @test_nd(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %[[reinterpret_cast:.*]] = memref.reinterpret_cast %[[arg0]] to offset: [0], sizes: [128], strides: [1] : memref<8x16xf16> to memref<128xf16>
      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[reinterpret_cast]] : memref<128xf16> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[r1:.*]] = vector.broadcast %[[r0]] : i64 to vector<1xi64>
      %arg00 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf16> to memref<128xf16>
      %0 = xegpu.create_nd_tdesc %arg00[0] : memref<128xf16> -> !xegpu.tensor_desc<128xf16>

      //CHECK: %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]] : memref<16x16xf16> -> index
      //CHECK: %[[r2:.*]] = arith.index_castui %[[intptr_0]] : index to i64
      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi64>
      //CHECK: %[[r3:.*]] = vector.insert %[[r2]], %[[cst]] [0] : i64 into vector<8xi64>
      //CHECK: %[[r4:.*]] = vector.bitcast %[[r3]] : vector<8xi64> to vector<16xi32>
      //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
      //CHECK: %[[c15_i32:.*]] = arith.constant 15 : i32
      //CHECK: %[[r5:.*]] = vector.insert %[[c31_i32]], %[[r4]] [2] : i32 into vector<16xi32>
      //CHECK: %[[r6:.*]] = vector.insert %[[c15_i32]], %[[r5]] [3] : i32 into vector<16xi32>
      //CHECK: %[[r7:.*]] = vector.insert %[[c31_i32]], %[[r6]] [4] : i32 into vector<16xi32>
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[r8:.*]] = vector.insert %[[c0_i32]], %[[r7]] [5] : i32 into vector<16xi32>
      //CHECK: %[[r9:.*]] = vector.insert %[[c0_i32]], %[[r8]] [6] : i32 into vector<16xi32>
      //CHECK: %[[c3855_i32:.*]] = arith.constant 3855 : i32
      //CHECK: %[[r10:.*]] = vector.insert %[[c3855_i32]], %[[r9]] [7] : i32 into vector<16xi32>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] {boundary_check = true} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

      //CHECK: %[[reinterpret_cast_1:.*]] = memref.reinterpret_cast %[[arg2]] to offset: [0], sizes: [128], strides: [1] : memref<8x16xf32> to memref<128xf32>
      %arg02 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf32> to memref<128xf32>

      //CHECK: %[[intptr_2:.*]] = memref.extract_aligned_pointer_as_index %[[reinterpret_cast_1]] : memref<128xf32> -> index
      //CHECK: %[[r11:.*]] = arith.index_castui %[[intptr_2]] : index to i64
      //CHECK: %[[r12:.*]] = vector.broadcast %[[r11]] : i64 to vector<1xi64>
      %2 = xegpu.create_nd_tdesc %arg02[0] : memref<128xf32> -> !xegpu.tensor_desc<128xf32>

      //RAW: %[[c0_i8:.*]] = arith.constant 0 : i8
      //RAW: %[[true:.*]] = arith.constant true
      //RAW: %[[c1_i8:.*]] = arith.constant 1 : i8
      //RAW: %[[c4_i8:.*]] = arith.constant 4 : i8
      //RAW: %[[c15_i8:.*]] = arith.constant 15 : i8
      //RAW: %[[c42133376_i32:.*]] = arith.constant 42133376 : i32
      //RAW: %[[cst_3:.*]] = arith.constant dense<0> : vector<32xi64>
      //RAW: %[[r13:.*]] = func.call @llvm.genx.raw.send2.v32i64.i1.v16i32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c4_i8]], %[[c15_i8]], %[[c0_i32]], %[[c42133376_i32]], %[[r1]], %[[cst_3]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<1xi64>, vector<32xi64>) -> vector<32xi64>
      //RAW: %[[r14:.*]] = vector.bitcast %[[r13]] : vector<32xi64> to vector<128xf16>
      %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<128xf16> -> vector<128xf16>

      //RAW: %[[c8_i8:.*]] = arith.constant 8 : i8
      //RAW: %[[c42074755_i32:.*]] = arith.constant 42074755 : i32
      //RAW: %[[cst_4:.*]] = arith.constant dense<0> : vector<128xi32>
      //RAW: %[[r15:.*]] = func.call @llvm.genx.raw.send2.v128i32.i1.v16i32(%c0_i8, %c0_i8, %true, %c1_i8, %c8_i8, %c15_i8, %c0_i32, %c42074755_i32, %10, %cst_4) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi32>, vector<128xi32>) -> vector<128xi32>
      //RAW: %[[r16:.*]] = vector.bitcast %[[r15]] : vector<128xi32> to vector<256xf16>

      %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

      //CHECK: %[[c134744586_i32:.*]] = arith.constant 134744586 : i32
      //CHECK: %[[r17:.*]] = vector.bitcast %[[r14]] : vector<128xf16> to vector<64xi32>
      //CHECK: %[[r18:.*]] = vector.bitcast %[[r16]] : vector<256xf16> to vector<128xi32>
      //CHECK: %[[r19:.*]] = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%[[r18]], %[[r17]], %[[c134744586_i32]]) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
      %6 = vector.shape_cast %3: vector<128xf16> to vector<8x16xf16>
      %5 = xegpu.dpas %6, %4 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %7 = vector.shape_cast %5: vector<8x16xf32> to vector<128xf32>

      //RAW: %[[c33748868_i32:.*]] = arith.constant 33748868 : i32
      //RAW: %[[r20:.*]] = vector.bitcast %[[r19]] : vector<128xf32> to vector<64xi64>
      //RAW: func.call @llvm.genx.raw.sends2.noresult.i1.v16i32.v64i64(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c8_i8]], %[[c15_i8]], %[[c0_i32]], %[[c33748868_i32]], %[[r12]], %[[r20]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<1xi64>, vector<64xi64>) -> ()
      xegpu.store_nd %7, %2 : vector<128xf32>, !xegpu.tensor_desc<128xf32>

      //CHECK: gpu.return
      gpu.return
    }
 }
}
