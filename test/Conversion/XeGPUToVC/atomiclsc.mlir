// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false'  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {

  gpu.module @test_kernel {
    // CHECK: func.func private @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64(vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi64>, vector<16xi32>, vector<16xi32>, i32, vector<16xi32>) -> vector<16xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64", linkage_type = <Import>>}

    gpu.func @test_atomiclsc(%arg0: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK:  %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
      %mask = arith.constant dense<true> : vector<16xi1>

      // CHECK: %[[cst_0:.*]] = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xindex>
      // CHECK: %[[OFFSETS:.*]] = builtin.unrealized_conversion_cast %[[cst_0]] : vector<16xindex> to vector<16xi64>

      %offsets = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xindex>

      // CHECK: %[[cst_1:.*]] = arith.constant dense<5.000000e-01> : vector<16xf32>
      %1 = arith.constant dense<0.5> : vector<16xf32>

      // CHECK: %[[STRUCT:.*]] = arith.constant dense<0> : vector<4xi64>
      // CHECK: %[[BASEPTR:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<8x16xf32> -> index
      // CHECK: %[[BASEADDR:.*]] = arith.index_castui %[[BASEPTR]] : index to i64
      // CHECK: %[[PAYLOAD_v4i64:.*]] = vector.insert %[[BASEADDR]], %[[STRUCT]] [0] : i64 into vector<4xi64>
      // CHECK: %[[PAYLOAD_v8i32:.*]] = vector.bitcast %[[PAYLOAD_v4i64]] : vector<4xi64> to vector<8xi32>
      %2 = xegpu.create_tdesc %arg0, %offsets {mode = vc, chunk_size_per_lane = 1} : memref<8x16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

      // CHECK: %[[cst_3:.*]] = arith.constant dense<true> : vector<16xi1>
      // CHECK: %[[TD_v4i64:.*]] = vector.bitcast %[[PAYLOAD_v8i32]] : vector<8xi32> to vector<4xi64>
      // CHECK: %[[BASE:.*]] = vector.extract %[[TD_v4i64]][0] : i64 from vector<4xi64>
      // CHECK: %[[PAYLOAD:.*]] = arith.constant dense<0> : vector<16xi64>
      // CHECK: %[[PAYLOAD0:.*]] = vector.insert %[[BASE]], %[[PAYLOAD]] [0] : i64 into vector<16xi64>
      // CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[PAYLOAD0]], %[[PAYLOAD0]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi64>, vector<16xi64>
      // CHECK: %[[ADDOFFSETS:.*]] = arith.addi %[[SHUFFLE]], %[[OFFSETS]] : vector<16xi64>
      // CHECK: %[[cst_8:.*]] = arith.constant dense<0> : vector<16xi32>
      // CHECK: %[[SRC0:.*]] = vector.bitcast %[[cst_1]] : vector<16xf32> to vector<16xi32>
      // CHECK: %[[ATOMIC_RES:.*]] = func.call @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64({{.*}}) : ({{vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<16xi64>, vector<16xi32>, vector<16xi32>, i32, vector<16xi32>}}) -> vector<16xi32>
      // CHECK: %{{.*}} = vector.bitcast %[[ATOMIC_RES]] : vector<16xi32> to vector<16xf32>
      %3 = xegpu.atomic_rmw "addf" %2, %mask, %1 {mode = vc} : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
      gpu.return
    }
 }
}
