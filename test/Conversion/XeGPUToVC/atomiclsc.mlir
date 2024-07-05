// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false'  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {

  gpu.module @test_kernel {
    // CHECK:    func.func private @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64(vector<16xi1>, i8, i8, i8, i16, i32, i8, i8, i8, vector<16xi1>, vector<16xindex>, vector<16xi32>, vector<16xi32>, i32, vector<16xi32>) -> vector<16xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64", linkage_type = <Import>>}


    gpu.func @test_atomiclsc(%arg0: memref<128xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK:  %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
      %mask = arith.constant dense<true> : vector<16xi1>

      // CHECK: %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %offsets = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>

      // CHECK: %[[cst_0:.*]] = arith.constant dense<5.000000e-01> : vector<16xf32>
      %1 = arith.constant dense<0.5> : vector<16xf32>

      // CHECK: %[[PAYLOAD:.*]] = arith.constant dense<0> : vector<16xindex>
      // CHECK: %[[BASEPTR:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<128xf32> -> index
      // CHECK: %[[VEC_BASEPTR_INSERTED:.*]] = vector.insert %[[BASEPTR]], %[[PAYLOAD]] [0] : index into vector<16xindex>
      // CHECK: %[[VEC_BASEPTR_SHUFFLED:.*]] = vector.shuffle %[[VEC_BASEPTR_INSERTED]], %[[VEC_BASEPTR_INSERTED]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xindex>, vector<16xindex>
      // CHECK: %[[ELEMENT_BYTEWIDTH:.*]] = arith.constant dense<4> : vector<16xindex>
      // CHECK: %[[OFFSETS_ADJUSTED:.*]] = arith.muli %[[ELEMENT_BYTEWIDTH]], %[[OFFSETS]] : vector<16xindex>
      // CHECK: %[[VEC_OFFSETS_APPLIED:.*]] = arith.addi %[[VEC_BASEPTR_SHUFFLED]], %[[OFFSETS_ADJUSTED]] : vector<16xindex>
      %2 = xegpu.create_tdesc %arg0, %offsets {chunk_size = 1} : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered = true>>

      // CHECK: %[[cst_3:.*]] = arith.constant dense<true> : vector<16xi1>
      // CHECK: %[[cst_8:.*]] = arith.constant dense<0> : vector<16xi32>
      // CHECK: %[[SRC0:.*]] = vector.bitcast %[[cst_0]] : vector<16xf32> to vector<16xi32>
      // CHECK: %[[ATOMIC_RES:.*]] = func.call @llvm.genx.lsc.xatomic.stateless.v16i32.v16i1.v16i64({{.*}}, %[[VEC_OFFSETS_APPLIED]], %[[SRC0]], %[[cst_8]], {{.*}}, %[[cst_8]]) : ({{.*}}) -> vector<16xi32>
      // CHECK: %{{.*}} = vector.bitcast %[[ATOMIC_RES]] : vector<16xi32> to vector<16xf32>
      %3 = xegpu.atomic_rmw "addf" %2, %mask, %1 : !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered = true>>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
      gpu.return
    }
 }
}
