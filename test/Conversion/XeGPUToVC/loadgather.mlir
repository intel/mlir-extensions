
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true'  %s | FileCheck %s
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false'  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
   gpu.module @module0 {
   // CHECK: func.func private @llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32(i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi64>, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32", linkage_type = <Import>>}
   // CHECK: func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
   // CHECK: func.func private @llvm.genx.raw.send2.v64i32.i1.v16i64(i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi64>, vector<64xi32>) -> vector<64xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.i1.v16i64", linkage_type = <Import>>}
  gpu.func @test_loadgather(%arg0: memref<128xf16>, %arg1: memref<16x16xf16>, %arg2: memref<128xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      // CHECK: %[[cst:.*]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
      %offsets = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>

      // CHECK: %[[A_STRUCT:.*]] = arith.constant dense<0> : vector<4xi64>
      // CHECK: %[[A_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<128xf16> -> index
      // CHECK: %[[A_BASEADDR:.*]] = arith.index_castui %[[A_BASEPTR]] : index to i64
      // CHECK: %[[A_PAYLOAD_v4i64:.*]] = vector.insert %[[A_BASEADDR]], %[[A_STRUCT]] [0] : i64 into vector<4xi64>
      // CHECK: %[[A_PAYLOAD_v8i32:.*]] = vector.bitcast %[[A_PAYLOAD_v4i64]] : vector<4xi64> to vector<8xi32>
       %0 = xegpu.create_tdesc %arg0, %offsets {chunk_size = 8} : memref<128xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>

       // CHECK: %[[cst_1:.*]] = arith.constant dense<true> : vector<128xi1>
       %cst = arith.constant dense<true> : vector<128xi1>

       %mask = vector.shape_cast %cst : vector<128xi1> to vector<16x8xi1>

       // CHECK: %[[LOADA_v128i32:.*]] = func.call @llvm.genx.raw.send2.v64i32.i1.v16i64(%c0_i8, %c4_i8, %true, %c2_i8, %c4_i8_2, %c15_i8, %c0_i32, %c71447936_i32, %8, %cst_4) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi64>, vector<64xi32>) -> vector<64xi32>
       %3 = xegpu.load %0, %mask : !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>, vector<16x8xi1> -> vector<16x8xf16>

       // CHECK: %[[LOADA_v128f16:.*]] = vector.bitcast %[[LOADA_v128i32]] : vector<64xi32> to vector<128xf16>
       %66 = vector.shape_cast %3: vector<16x8xf16> to vector<128xf16>

       // CHECK: %[[LOADA_v64i32:.*]] = vector.bitcast %[[LOADA_v128f16]] : vector<128xf16> to vector<64xi32>
       %6 = vector.shape_cast %66: vector<128xf16> to vector<8x8x2xf16>

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

       %4 = xegpu.load_nd %1 {vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

       // CHECK: %[[C_ACC_v128f32:.*]] = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%{{.*}}, %[[LOADA_v64i32]], %{{.*}}) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
       %5 = xegpu.dpas %6, %4 : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>

       // CHECK: %[[cst_17:.*]] = arith.constant dense<[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480]> : vector<16xindex>
       // CHECK: %[[C_OFFSETS:.*]] = builtin.unrealized_conversion_cast %[[cst_17]] : vector<16xindex> to vector<16xi64>

       %offsets2 = arith.constant dense<[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480]> : vector<16xindex>

      // CHECK: %[[C_STRUCT:.*]] = arith.constant dense<0> : vector<4xi64>
      // CHECK: %[[C_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<128xf32> -> index
      // CHECK: %[[C_BASE:.*]] = arith.index_castui %[[C_BASEPTR]] : index to i64
      // CHECK: %[[C_PAYLOAD:.*]] = vector.insert %[[C_BASE]], %[[C_STRUCT]] [0] : i64 into vector<4xi64>
      // CHECK: %[[C_PAYLOAD_v8i32:.*]] = vector.bitcast %[[C_PAYLOAD]] : vector<4xi64> to vector<8xi32>
       %2 = xegpu.create_tdesc %arg2, %offsets2 {chunk_size = 8} : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
       %7 = vector.shape_cast %5: vector<8x16xf32> to vector<128xf32>
       %8 = vector.shape_cast %7: vector<128xf32> to vector<16x8xf32>

       // CHECK: %[[C_PAYLOAD_v4i64:.*]] = vector.bitcast %[[C_PAYLOAD_v8i32]] : vector<8xi32> to vector<4xi64>
       // CHECK: %[[C_ADDR:.*]] = vector.extract %[[C_PAYLOAD_v4i64]][0] : i64 from vector<4xi64>
       // CHECK: %[[PAYLOAD_v16i64:.*]] = arith.constant dense<0> : vector<16xi64>
       // CHECK: %[[P0:.*]] = vector.insert %[[C_ADDR]], %[[PAYLOAD_v16i64]] [0] : i64 into vector<16xi64>
       // CHECK: %[[OFFSETS:.*]] = vector.shuffle %[[P0]], %[[P0]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi64>, vector<16xi64>
       // CHECK: %[[STORE_OFFSETS:.*]] = arith.addi %[[OFFSETS]], %[[C_OFFSETS]] : vector<16xi64>


       // CHECK: func.call @llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32({{.*}}, %[[STORE_OFFSETS]], %[[C_ACC_v128f32]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<16xi64>, vector<128xf32>) -> ()
       xegpu.store %8, %2, %mask : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>, vector<16x8xi1>


       gpu.return
    }
     }
}
