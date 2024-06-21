// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true'  %s | FileCheck %s --check-prefixes=CHECK,RAW
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false'  %s | FileCheck %s --check-prefixes=CHECK,LSC
module @gemm attributes {gpu.container_module} {

  gpu.module @test_kernel {

    // RAW: func.func private @llvm.genx.raw.sends2.noresult.i1.v8i32.v64i64(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi64>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v64i64", linkage_type = <Import>>}
    // LSC: func.func private @llvm.genx.lsc.store.stateless.i1.i64.v64i64(i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, vector<64xi64>, i32) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.store.stateless.i1.i64.v64i64", linkage_type = <Import>>}

    // CHECK: func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}

    // RAW: func.func private @llvm.genx.raw.send2.v128i32.i1.v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
    // LSC: func.func private @llvm.genx.lsc.load2d.stateless.v128i32.i1.i64(i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<128xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load2d.stateless.v128i32.i1.i64", linkage_type = <Import>>}

    // RAW: func.func private @llvm.genx.raw.send2.v32i64.i1.v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<32xi64>) -> vector<32xi64> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v32i64.i1.v8i32", linkage_type = <Import>>}
    // LSC: func.func private @llvm.genx.lsc.load.stateless.v32i64.i1.i64(i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<32xi64> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.load.stateless.v32i64.i1.i64", linkage_type = <Import>>}

    // CHECK: gpu.func @test_nd(%[[arg0:.*]]: memref<8x16xf16>, %[[arg1:.*]]: memref<16x16xf16>, %[[arg2:.*]]: memref<8x16xf32>)
    gpu.func @test_nd(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      %arg00 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf16> to memref<128xf16>

      // CHECK: %[[A_STRUCT:.*]] = arith.constant dense<0> : vector<4xi64>
      // CHECK: %[[A_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<128xf16> -> index
      // CHECK: %[[A_BASEADDR:.*]] = arith.index_castui %[[A_BASEPTR]] : index to i64
      // CHECK: %[[A_PAYLOAD_v4i64:.*]] = vector.insert %[[A_BASEADDR]], %[[A_STRUCT]] [0] : i64 into vector<4xi64>
      // CHECK: %[[A_PAYLOAD_v8i32:.*]] = vector.bitcast %[[A_PAYLOAD_v4i64]] : vector<4xi64> to vector<8xi32>
      %0 = xegpu.create_nd_tdesc %arg00[0] : memref<128xf16> -> !xegpu.tensor_desc<128xf16>

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

      %arg02 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf32> to memref<128xf32>

      // CHECK: %[[C_STRUCT:.*]] = arith.constant dense<0> : vector<4xi64>
      // CHECK: %[[C_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<128xf32> -> index
      // CHECK: %[[C_BASE:.*]] = arith.index_castui %[[C_BASEPTR]] : index to i64
      // CHECK: %[[C_PAYLOAD:.*]] = vector.insert %[[C_BASE]], %[[C_STRUCT]] [0] : i64 into vector<4xi64>
      // CHECK: %[[C_PAYLOAD_v8i32:.*]] = vector.bitcast %[[C_PAYLOAD]] : vector<4xi64> to vector<8xi32>
      %2 = xegpu.create_nd_tdesc %arg02[0] : memref<128xf32> -> !xegpu.tensor_desc<128xf32>


      // RAW: %[[LOAD2D_A_v32i64:.*]] =  func.call @llvm.genx.raw.send2.v32i64.i1.v8i32({{.*}}, %[[A_PAYLOAD_v8i32]], %{{.*}}) : ({{.*}}, vector<8xi32>, vector<32xi64>) -> vector<32xi64>

      // LSC: %[[A_v4i64:.*]] = vector.bitcast %[[A_PAYLOAD_v8i32]] : vector<8xi32> to vector<4xi64>
      // LSC: %[[BASE_A:.*]] = vector.extract %[[A_v4i64]][0] : i64 from vector<4xi64>
      // LSC: %[[LOAD2D_A_v32i64:.*]] = func.call @llvm.genx.lsc.load.stateless.v32i64.i1.i64({{.*}}, %[[BASE_A]], %{{.*}}) : ({{.*}}) -> vector<32xi64>
      // CHECK: %[[LOAD2D_A_v32i64_CAST:.*]] = vector.bitcast %[[LOAD2D_A_v32i64]] : vector<32xi64> to vector<128xf16>
      %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<128xf16> -> vector<128xf16>

      // RAW:  %[[LOAD2D_B_v128i32:.*]] = func.call @llvm.genx.raw.send2.v128i32.i1.v8i32({{.*}}, %[[B_PAYLOAD]], %{{.*}}) : ({{.*}}) -> vector<128xi32>

      // LSC: %[[B_v4i64:.*]] = vector.bitcast %[[B_PAYLOAD]] : vector<8xi32> to vector<4xi64>
      // LSC: %[[BASE_B:.*]] = vector.extract %[[B_v4i64]][0] : i64 from vector<4xi64>
      // LSC: %[[B_OFFSETX:.*]] = vector.extract %[[B_PAYLOAD]][5] : i32 from vector<8xi32>
      // LSC: %[[B_OFFSETY:.*]] = vector.extract %[[B_PAYLOAD]][6] : i32 from vector<8xi32>
      // LSC: %[[LOAD2D_B_v128i32:.*]] = func.call @llvm.genx.lsc.load2d.stateless.v128i32.i1.i64({{.*}}, %[[BASE_B]], {{.*}}, %[[B_OFFSETX]], %[[B_OFFSETY]]) : {{.*}} -> vector<128xi32>
      // CHECK: %[[LOAD2D_B_v128i32_CAST:.*]] = vector.bitcast %[[LOAD2D_B_v128i32]] : vector<128xi32> to vector<256xf16>
      %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %6 = vector.shape_cast %3: vector<128xf16> to vector<8x8x2xf16>

      // CHECK: %[[LOAD2D_A_v32i64_RECAST:.*]] = vector.bitcast %[[LOAD2D_A_v32i64_CAST]] : vector<128xf16> to vector<64xi32>
      // CHECK: %[[LOAD2D_B_v128i32_RECAST:.*]] = vector.bitcast %[[LOAD2D_B_v128i32_CAST]] : vector<256xf16> to vector<128xi32>
      // CHECK: %[[C_ACC_v128f32:.*]] = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%[[LOAD2D_B_v128i32_RECAST]], %[[LOAD2D_A_v32i64_RECAST]], %{{.*}}) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>

      %5 = xegpu.dpas %6, %4 : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %7 = vector.shape_cast %5: vector<8x16xf32> to vector<128xf32>

      // LSC: %[[C_RES_v4i64:.*]] = vector.bitcast %[[C_PAYLOAD_v8i32]] : vector<8xi32> to vector<4xi64>
      // LSC: %[[C_RES_BASE:.*]] = vector.extract %[[C_RES_v4i64]][0] : i64 from vector<4xi64>
      // CHECK: %[[C_ACC_v64i64:.*]] =  vector.bitcast %[[C_ACC_v128f32]] : vector<128xf32> to vector<64xi64>

      // RAW: func.call @llvm.genx.raw.sends2.noresult.i1.v8i32.v64i64({{.*}}, %[[C_PAYLOAD_v8i32]], %[[C_ACC_v64i64]]) : ({{.*}}) -> ()

      // LSC: func.call @llvm.genx.lsc.store.stateless.i1.i64.v64i64({{.*}}, %[[C_RES_BASE]], %[[C_ACC_v64i64]], %{{.*}}) : ({{.*}}, i64, vector<64xi64>, {{.*}}) -> ()
      xegpu.store_nd %7, %2 : vector<128xf32>, !xegpu.tensor_desc<128xf32>
      gpu.return
    }
 }
}
