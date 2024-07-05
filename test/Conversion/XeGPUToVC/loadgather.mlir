
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true'  %s | FileCheck %s
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false'  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
   gpu.module @module0 {
    // CHECK: func.func private @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v16i32(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<16xi32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.v16i1.v16i64.v16i32", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.raw.send2.v16i32.v16i1.v16i64(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<16xi32>) -> vector<16xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v16i32.v16i1.v16i64", linkage_type = <Import>>}
    gpu.func @test_loadgather(%in: memref<?xf16>, %out: memref<16x2xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %out_flat = memref.reinterpret_cast %out to offset: [0], sizes: [32], strides: [1] : memref<16x2xf16> to memref<32xf16>
      // CHECK: %[[OFFSET:.*]] = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 34, 38, 42, 46, 50, 54, 58]> : vector<16xindex>
      %offsets = arith.constant dense<[0,4,8,12,16,20,24,28,32,34,38,42,46,50,54,58]> : vector<16xindex>
      // CHECK: %[[MASK:.*]] = arith.constant dense<[true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<16xi1>
      %mask = arith.constant dense<[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]> : vector<16xi1>

      // CHECK: %[[IN_EMPTY_PAYLOAD:.*]] = arith.constant dense<0> : vector<16xindex>
      // CHECK: %[[IN_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<?xf16> -> index
      // CHECK: %[[IN_PAYLOAD_BASEPTR:.*]] = vector.insert %[[IN_BASEPTR]], %[[IN_EMPTY_PAYLOAD]] [0] : index into vector<16xindex>
      // CHECK: %[[IN_PAYLOAD_BASEPTR_SHUFFLED:.*]] = vector.shuffle %[[IN_PAYLOAD_BASEPTR]], %[[IN_PAYLOAD_BASEPTR]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xindex>, vector<16xindex>
      // CHECK: %[[IN_ELEMENT_BYTEWIDTH:.*]] = arith.constant dense<2> : vector<16xindex>
      // CHECK: %[[IN_ELEMENTWISE_OFFSET:.*]] = arith.muli %[[IN_ELEMENT_BYTEWIDTH]], %[[OFFSET]] : vector<16xindex>
      // CHECK: %[[IN_PAYLOAD:.*]] = arith.addi %[[IN_PAYLOAD_BASEPTR_SHUFFLED]], %[[IN_ELEMENTWISE_OFFSET]] : vector<16xindex>
      %tdesc_in = xegpu.create_tdesc %in, %offsets {chunk_size = 2} : memref<?xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf16, #xegpu.tdesc_attr<scattered = true>>

      // CHECK: %[[OUT_EMPTY_PAYLOAD:.*]] = arith.constant dense<0> : vector<16xindex>
      // CHECK: %[[OUT_BASEPTR:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<32xf16> -> index
      // CHECK: %[[OUT_PAYLOAD_BASEPTR:.*]] = vector.insert %[[OUT_BASEPTR]], %[[OUT_EMPTY_PAYLOAD]] [0] : index into vector<16xindex>
      // CHECK: %[[OUT_PAYLOAD_BASEPTR_SHUFFLED:.*]] = vector.shuffle %[[OUT_PAYLOAD_BASEPTR]], %[[OUT_PAYLOAD_BASEPTR]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xindex>, vector<16xindex>
      // CHECK: %[[OUT_ELEMENT_BYTEWIDTH:.*]] = arith.constant dense<2> : vector<16xindex>
      // CHECK: %[[OUT_ELEMENTWISE_OFFSET:.*]] = arith.muli %[[OUT_ELEMENT_BYTEWIDTH]], %[[OFFSET]] : vector<16xindex>
      // CHECK: %[[OUT_PAYLOAD:.*]] = arith.addi %[[OUT_PAYLOAD_BASEPTR_SHUFFLED]], %[[OUT_ELEMENTWISE_OFFSET]] : vector<16xindex>
      %tdesc_out = xegpu.create_tdesc %out_flat, %offsets {chunk_size = 2} : memref<32xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf16, #xegpu.tdesc_attr<scattered = true>>

      // CHECK: %[[OLD:.*]] =  arith.constant dense<0> : vector<16xi32>
      // CHECK: %[[LOAD_RES:.*]] = func.call @llvm.genx.raw.send2.v16i32.v16i1.v16i64({{.*}}, %[[MASK]], {{.*}}, %[[IN_PAYLOAD]], %[[OLD]]) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<16xi32>) -> vector<16xi32>
      %loaded = xegpu.load %tdesc_in, %mask : !xegpu.tensor_desc<16x2xf16, #xegpu.tdesc_attr<scattered = true>>, vector<16xi1> -> vector<16x2xf16>
      // CHECK: %[[POST_OP_ELEMENT_TYPE_CAST:.*]] = vector.bitcast %[[LOAD_RES]] : vector<16xi32> to vector<32xf16>

      // CHECK: %[[PRE_OP_ELEMENT_TYPE_CAST:.*]] = vector.bitcast %[[POST_OP_ELEMENT_TYPE_CAST]] : vector<32xf16> to vector<16xi32>
      // CHECK: func.call @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v16i32({{.*}}, %[[MASK]], {{.*}}, %[[OUT_PAYLOAD]], %[[PRE_OP_ELEMENT_TYPE_CAST]]) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xindex>, vector<16xi32>) -> ()
      xegpu.store %loaded, %tdesc_out, %mask : vector<16x2xf16>, !xegpu.tensor_desc<16x2xf16, #xegpu.tdesc_attr<scattered = true>>, vector<16xi1>

      gpu.return
    }
  }
}
