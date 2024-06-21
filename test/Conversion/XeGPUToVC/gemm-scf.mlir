
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true'  %s | FileCheck %s --check-prefixes=CHECK,RAW
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false'  %s | FileCheck %s --check-prefixes=CHECK,LSC

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
      // CHECK: %[[C_STRUCT:.*]] = arith.constant dense<0> : vector<4xi64>
      // CHECK: %[[C_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<1024x1016xf32> -> index
      // CHECK: %[[C_BASEADDR:.*]] = arith.index_castui %[[C_BASEPTR]] : index to i64
      // CHECK: %[[C_PAYLOAD_v4i64:.*]] = vector.insert %[[C_BASEADDR]], %[[C_STRUCT]] [0] : i64 into vector<4xi64>
      // CHECK: %[[C_PAYLOAD_v8i32:.*]] = vector.bitcast %[[C_PAYLOAD_v4i64]] : vector<4xi64> to vector<8xi32>
      // CHECK: %[[C_P2:.*]] = vector.insert %{{.*}}, %[[C_PAYLOAD_v8i32]] [2] : i32 into vector<8xi32>
      // CHECK: %[[C_P3:.*]] = vector.insert %{{.*}}, %[[C_P2]] [3] : i32 into vector<8xi32>
      // CHECK: %[[C_P4:.*]] = vector.insert %{{.*}}, %[[C_P3]] [4] : i32 into vector<8xi32>
      // CHECK: %[[C_P5:.*]] = vector.insert %{{.*}}, %[[C_P4]] [5] : i32 into vector<8xi32>
      // CHECK: %[[C_P6:.*]] = vector.insert %{{.*}}, %[[C_P5]] [6] : i32 into vector<8xi32>
      // CHECK: %[[C_PAYLOAD:.*]] = vector.insert %{{.*}}, %[[C_P6]] [7] : i32 into vector<8xi32>
      %4 = xegpu.create_nd_tdesc  %arg2[%2, %3] : memref<1024x1016xf32> -> !xegpu.tensor_desc<8x16xf32>

       // RAW: %[[cst_2:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
       // RAW: %[[LOAD2D_C:.*]] = func.call @llvm.genx.raw.send2.v128f32.i1.v8i32({{.*}}, %[[C_PAYLOAD]], %[[cst_2]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf32>) -> vector<128xf32>

      // LSC: %[[C_v4i64:.*]] = vector.bitcast %[[C_PAYLOAD]] : vector<8xi32> to vector<4xi64>
      // LSC: %[[C_BASE:.*]] = vector.extract %[[C_v4i64]][0] : i64 from vector<4xi64>
      // LSC: %[[C_OFFSETX:.*]] = vector.extract %[[C_PAYLOAD]][5] : i32 from vector<8xi32>
      // LSC: %[[C_OFFSETY:.*]] = vector.extract %[[C_PAYLOAD]][6] : i32 from vector<8xi32>
      // LSC: %[[LOAD2D_C:.*]] = func.call @llvm.genx.lsc.load2d.stateless.v128f32.i1.i64({{.*}}, %[[C_BASE]], {{.*}}, %[[C_OFFSETX]], %[[C_OFFSETY]]) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<128xf32>
      %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      // CHECK: %[[SCF_RESULT:.*]] = scf.for {{.*}} iter_args(%[[arg4:.*]] = %[[LOAD2D_C]]) -> (vector<128xf32>)
      %6 = scf.for %arg3 = %c0 to %c1016 step %c16 iter_args(%arg4 = %5) -> (vector<8x16xf32>) {
        // CHECK: %[[A_STRUCT:.*]] = arith.constant dense<0> : vector<4xi64>
        // CHECK: %[[A_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<1024x1016xf16> -> index
        // CHECK: %[[A_BASEADDR:.*]] = arith.index_castui %[[A_BASEPTR]] : index to i64
        // CHECK: %[[A_PAYLOAD_v4i64:.*]] = vector.insert %[[A_BASEADDR]], %[[A_STRUCT]] [0] : i64 into vector<4xi64>
        // CHECK: %[[A_PAYLOAD_v8i32:.*]] = vector.bitcast %[[A_PAYLOAD_v4i64]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[A_p2:.*]] = vector.insert %{{.*}}, %[[A_PAYLOAD_v8i32]] [2] : i32 into vector<8xi32>
        // CHECK: %[[A_p3:.*]] = vector.insert %{{.*}}, %[[A_p2]] [3] : i32 into vector<8xi32>
        // CHECK: %[[A_p4:.*]] = vector.insert %{{.*}}, %[[A_p3]] [4] : i32 into vector<8xi32>
        // CHECK: %[[A_p5:.*]] = vector.insert %{{.*}}, %[[A_p4]] [5] : i32 into vector<8xi32>
        // CHECK: %[[A_p6:.*]] = vector.insert %{{.*}}, %[[A_p5]] [6] : i32 into vector<8xi32>
        // CHECK: %[[A_PAYLOAD:.*]] = vector.insert %{{.*}}, %[[A_p6]] [7] : i32 into vector<8xi32>
        %7 = xegpu.create_nd_tdesc %arg0[%2, %arg3] : memref<1024x1016xf16> -> !xegpu.tensor_desc<8x16xf16>


        // CHECK: %[[B_STRUCT:.*]]= arith.constant dense<0> : vector<4xi64>
        // CHECK: %[[B_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<1016x1016xf16> -> index
        // CHECK: %[[B_BASEADDR:.*]] = arith.index_castui %[[B_BASEPTR]] : index to i64
        // CHECK: %[[B_PAYLOAD_v4i64:.*]] = vector.insert %[[B_BASEADDR]], %[[B_STRUCT]][0] : i64 into vector<4xi64>
        // CHECK: %[[B_PAYLOAD_v8i32:.*]] = vector.bitcast %[[B_PAYLOAD_v4i64]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[B_p2:.*]] = vector.insert %{{.*}}, %[[B_PAYLOAD_v8i32]] [2] : i32 into vector<8xi32>
        // CHECK: %[[B_p3:.*]] = vector.insert %{{.*}}, %[[B_p2]] [3] : i32 into vector<8xi32>
        // CHECK: %[[B_p4:.*]] = vector.insert %{{.*}}, %[[B_p3]] [4] : i32 into vector<8xi32>
        // CHECK: %[[B_p5:.*]] = vector.insert %{{.*}}, %[[B_p4]] [5] : i32 into vector<8xi32>
        // CHECK: %[[B_p6:.*]] = vector.insert %{{.*}}, %[[B_p5]] [6] : i32 into vector<8xi32>
        // CHECK: %[[B_PAYLOAD:.*]] = vector.insert %{{.*}}, %[[B_p6]] [7] : i32 into vector<8xi32>
        %8 = xegpu.create_nd_tdesc %arg1[%arg3, %3] : memref<1016x1016xf16> -> !xegpu.tensor_desc<16x16xf16>

        // RAW: %[[LOAD2D_A_v64i32:.*]] = func.call @llvm.genx.raw.send2.v64i32.i1.v8i32({{.*}}, %[[A_PAYLOAD]], %{{.*}}) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32>

        // LSC: %[[A_v4i64:.*]] = vector.bitcast %[[A_PAYLOAD]] : vector<8xi32> to vector<4xi64>
        // LSC: %[[BASE_A:.*]] = vector.extract %[[A_v4i64]][0] : i64 from vector<4xi64>
        // LSC: %[[A_OFFSETX:.*]] = vector.extract %[[A_PAYLOAD]][5] : i32 from vector<8xi32>
        // LSC: %[[A_OFFSETY:.*]] = vector.extract %[[A_PAYLOAD]][6] : i32 from vector<8xi32>
        // LSC: %[[LOAD2D_A_v64i32:.*]] = func.call @llvm.genx.lsc.load2d.stateless.v64i32.i1.i64({{.*}}, %[[BASE_A]], {{.*}}, %[[A_OFFSETX]], %[[A_OFFSETY]]) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi32>
        // CHECK: %[[LOAD2D_A_v64i32_CAST:.*]] = vector.bitcast %[[LOAD2D_A_v64i32]] : vector<64xi32> to vector<128xf16>
        %9 = xegpu.load_nd %7 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

        // RAW: %[[LOAD2D_B_v128i32:.*]] = func.call @llvm.genx.raw.send2.v128i32.i1.v8i32({{.*}}, %[[B_PAYLOAD]], %{{.*}}) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>

        // LSC: %[[B_v4i64:.*]] = vector.bitcast %[[B_PAYLOAD]] : vector<8xi32> to vector<4xi64>
        // LSC: %[[BASE_B:.*]] = vector.extract %[[B_v4i64]][0] : i64 from vector<4xi64>
        // LSC: %[[B_OFFSETX:.*]] = vector.extract %[[B_PAYLOAD]][5] : i32 from vector<8xi32>
        // LSC: %[[B_OFFSETY:.*]] = vector.extract %[[B_PAYLOAD]][6] : i32 from vector<8xi32>
        // LSC: %[[LOAD2D_B_v128i32:.*]] = func.call @llvm.genx.lsc.load2d.stateless.v128i32.i1.i64({{.*}}, %[[BASE_B]], {{.*}}, %[[B_OFFSETX]], %[[B_OFFSETY]]) : {{.*}} -> vector<128xi32>
        // CHECK: %[[LOAD2D_B_v128i32_CAST:.*]] = vector.bitcast %[[LOAD2D_B_v128i32]] : vector<128xi32> to vector<256xf16>
        %10 = xegpu.load_nd %8 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        // CHECK: %[[LOAD2D_A_v64i32_RECAST:.*]] = vector.bitcast %[[LOAD2D_A_v64i32_CAST]] : vector<128xf16> to vector<64xi32>
        // CHECK: %[[LOAD2D_B_v128i32_RECAST:.*]] = vector.bitcast %[[LOAD2D_B_v128i32_CAST]] : vector<256xf16> to vector<128xi32>
        // CHECK: %[[DPAS_RES:.*]] = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%[[arg4]], %[[LOAD2D_B_v128i32_RECAST]], %[[LOAD2D_A_v64i32_RECAST]], {{.*}}) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %11 = xegpu.dpas %9, %10, %arg4 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // CHECK: scf.yield %[[DPAS_RES]] : vector<128xf32>
        scf.yield %11 : vector<8x16xf32>
      }

      // RAW: func.call @llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32({{.*}}, %[[C_PAYLOAD]], %[[SCF_RESULT]]) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf32>) -> ()

      // LSC: %[[C_RES_v4i64:.*]] = vector.bitcast %[[C_PAYLOAD]] : vector<8xi32> to vector<4xi64>
      // LSC: %[[C_RES_BASE:.*]] = vector.extract %[[C_RES_v4i64]][0] : i64 from vector<4xi64>
      // LSC: %[[C_RES_OOFSETX:.*]] = vector.extract %[[C_PAYLOAD]][5] : i32 from vector<8xi32>
      // LSC: %[[C_RES_OOFSETY:.*]] = vector.extract %[[C_PAYLOAD]][6] : i32 from vector<8xi32>
      // LSC:  func.call @llvm.genx.lsc.store2d.stateless.i1.i64.v128f32({{.*}}, %[[C_RES_BASE]], {{.*}}, %[[C_RES_OOFSETX]], %[[C_RES_OOFSETY]], %[[SCF_RESULT]]) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32, vector<128xf32>) -> ()
      xegpu.store_nd %6, %4 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }

}
