
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true'  %s | FileCheck %s
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false'  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
   gpu.module @module0 {
    // CHECK: func.func private @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v128f32(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.v16i1.v16i64.v128f32", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.raw.send2.v64i32.v16i1.v16i64(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<64xi32>) -> vector<64xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.v16i1.v16i64", linkage_type = <Import>>}
    gpu.func @test_loadgather(%arg0: memref<128xf16>, %arg1: memref<16x16xf16>, %arg2: memref<128xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
         // CHECK: %[[IN_OFFSET:.*]] = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
         %offsets = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
         // CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<16xi1>
         %mask = arith.constant dense<true> : vector<16xi1>

         // CHECK: %[[IN_EMPTY_PAYLOAD:.*]] = arith.constant dense<0> : vector<16xindex>
         // CHECK: %[[IN_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<128xf16> -> index
         // CHECK: %[[IN_PAYLOAD_BASEPTR:.*]] = vector.insert %[[IN_BASEPTR]], %[[IN_EMPTY_PAYLOAD]] [0] : index into vector<16xindex>
         // CHECK: %[[IN_PAYLOAD_BASEPTR_SHUFFLED:.*]] = vector.shuffle %[[IN_PAYLOAD_BASEPTR]], %[[IN_PAYLOAD_BASEPTR]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xindex>, vector<16xindex>
         // CHECK: %[[IN_ELEMENT_BYTEWIDTH:.*]] = arith.constant dense<2> : vector<16xindex>
         // CHECK: %[[IN_ELEMENTWISE_OFFSET:.*]] = arith.muli %[[IN_ELEMENT_BYTEWIDTH]], %[[IN_OFFSET]] : vector<16xindex>
         // CHECK: %[[IN_PAYLOAD:.*]] = arith.addi %[[IN_PAYLOAD_BASEPTR_SHUFFLED]], %[[IN_ELEMENTWISE_OFFSET]] : vector<16xindex>
         %0 = xegpu.create_tdesc %arg0, %offsets {chunk_size = 8} : memref<128xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>

         // CHECK: %[[OLD:.*]] =  arith.constant dense<0> : vector<64xi32>
         // CHECK: %[[LOAD_RES:.*]] = func.call @llvm.genx.raw.send2.v64i32.v16i1.v16i64({{.*}}, %[[MASK]], {{.*}}, %[[IN_PAYLOAD]], %[[OLD]]) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<64xi32>) -> vector<64xi32>
         %3 = xegpu.load %0, %mask : !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>, vector<16xi1> -> vector<16x8xf16>

         // CHECK: %[[LOADA_v128f16:.*]] = vector.bitcast %[[LOAD_RES]] : vector<64xi32> to vector<128xf16>
         %66 = vector.shape_cast %3: vector<16x8xf16> to vector<128xf16>
         %6 = vector.shape_cast %66: vector<128xf16> to vector<8x16xf16>

         // CHECK: %[[B_STRUCT:.*]]= arith.constant dense<0> : vector<4xi64>
         // CHECK: %[[B_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<16x16xf16> -> index
         // CHECK: %[[B_BASEADDR:.*]] = arith.index_castui %[[B_BASEPTR]] : index to i64
         // CHECK: %[[B_PAYLOAD_v4i64:.*]] = vector.insert %[[B_BASEADDR]], %[[B_STRUCT]][0] : i64 into vector<4xi64>
         // CHECK: %[[B_PAYLOAD_v8i32:.*]] = vector.bitcast %[[B_PAYLOAD_v4i64]] : vector<4xi64> to vector<8xi32>
         // CHECK: %[[p2:.*]] = vector.insert %{{.*}}, %[[B_PAYLOAD_v8i32]] [2] : i32 into vector<8xi32>
         // CHECK: %[[p3:.*]] = vector.insert %{{.*}}, %[[p2]] [3] : i32 into vector<8xi32>
         // CHECK: %[[p4:.*]] = vector.insert %{{.*}}, %[[p3]] [4] : i32 into vector<8xi32>
         // CHECK: %[[p5:.*]] = vector.insert %{{.*}}, %[[p4]] [5] : i32 into vector<8xi32>
         // CHECK: %[[p6:.*]] = vector.insert %{{.*}}, %[[p5]] [6] : i32 into vector<8xi32>
         // CHECK: %[[B_PAYLOAD:.*]] = vector.insert %{{.*}}, %[[p6]] [7] : i32 into vector<8xi32>
         %1 = xegpu.create_nd_tdesc %arg1[0, 0] {boundary_check = true} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

         %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

         // CHECK: %[[LOADA_v64i32:.*]] = vector.bitcast %[[LOADA_v128f16]] : vector<128xf16> to vector<64xi32>
         // CHECK: %[[C_ACC_v128f32:.*]] = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%{{.*}}, %[[LOADA_v64i32]], %{{.*}}) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
         %5 = xegpu.dpas %6, %4 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>

         // CHECK: %[[OUT_OFFSET:.*]] = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
         %offsets2 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>

         // CHECK: %[[OUT_EMPTY_PAYLOAD:.*]] = arith.constant dense<0> : vector<16xindex>
         // CHECK: %[[OUT_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<128xf32> -> index
         // CHECK: %[[OUT_PAYLOAD_BASEPTR:.*]] = vector.insert %[[OUT_BASEPTR]], %[[OUT_EMPTY_PAYLOAD]] [0] : index into vector<16xindex>
         // CHECK: %[[OUT_PAYLOAD_BASEPTR_SHUFFLED:.*]] = vector.shuffle %[[OUT_PAYLOAD_BASEPTR]], %[[OUT_PAYLOAD_BASEPTR]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xindex>, vector<16xindex>
         // CHECK: %[[OUT_ELEMENT_BYTEWIDTH:.*]] = arith.constant dense<4> : vector<16xindex>
         // CHECK: %[[OUT_ELEMENTWISE_OFFSET:.*]] = arith.muli %[[OUT_ELEMENT_BYTEWIDTH]], %[[OUT_OFFSET]] : vector<16xindex>
         // CHECK: %[[OUT_PAYLOAD:.*]] = arith.addi %[[OUT_PAYLOAD_BASEPTR_SHUFFLED]], %[[OUT_ELEMENTWISE_OFFSET]] : vector<16xindex>
         %2 = xegpu.create_tdesc %arg2, %offsets2 {chunk_size = 8} : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
         %7 = vector.shape_cast %5: vector<8x16xf32> to vector<128xf32>
         %8 = vector.shape_cast %7: vector<128xf32> to vector<16x8xf32>

         // CHECK: func.call @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v128f32({{.*}}, %[[MASK]], {{.*}}, %[[OUT_PAYLOAD]], %[[C_ACC_v128f32]]) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<128xf32>) -> ()
         xegpu.store %8, %2, %mask : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>, vector<16xi1>

         gpu.return
      }
   }
}
