
// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=true' --cse %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
   gpu.module @module0 {
    //CHECK: func.func private @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v16i32(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<16xi32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.v16i1.v16i64.v16i32", linkage_type = <Import>>}
    //CHECK: func.func private @llvm.genx.raw.send2.v16i32.v16i1.v16i64(i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<16xi32>) -> vector<16xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v16i32.v16i1.v16i64", linkage_type = <Import>>}
    gpu.func @test_loadgather(%in: memref<?xf16>, %out: memref<16x2xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

      //CHECK: %[[reinterpret_cast:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [32], strides: [1] : memref<16x2xf16> to memref<32xf16>
      //CHECK: %[[cst:.*]] = arith.constant dense<[true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<16xi1>
      //CHECK: %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %{{.*}} : memref<?xf16> -> index
      //CHECK: %[[r0:.*]] = arith.index_castui %[[intptr]] : index to i64
      //CHECK: %[[cst_0:.*]] = arith.constant dense<2> : vector<16xi64>

      //CHECK: %[[r1:.*]] = vector.from_elements {{.*}} : vector<16xindex>
      //CHECK: %[[r2:.*]] = arith.index_castui %[[r1]] : vector<16xindex> to vector<16xi64>
      //CHECK: %[[r3:.*]] = arith.muli %[[r2]], %[[cst_0]] : vector<16xi64>
      //CHECK: %[[r4:.*]] = vector.broadcast %[[r0]] : i64 to vector<16xi64>
      //CHECK: %[[r5:.*]] = arith.addi %[[r4]], %[[r3]] : vector<16xi64>

      //CHECK: %[[intptr_1:.*]] = memref.extract_aligned_pointer_as_index %[[reinterpret_cast]] : memref<32xf16> -> index
      //CHECK: %[[r6:.*]] = arith.index_castui %[[intptr_1]] : index to i64
      //CHECK: %[[r7:.*]] = vector.broadcast %[[r6]] : i64 to vector<16xi64>
      //CHECK: %[[r8:.*]] = arith.addi %[[r7]], %[[r3]] : vector<16xi64>
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: %[[c4_i8:.*]] = arith.constant 4 : i8
      //CHECK: %[[c2_i8:.*]] = arith.constant 2 : i8
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[c15_i8:.*]] = arith.constant 15 : i8
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[c68289920_i32:.*]] = arith.constant 68289920 : i32
      //CHECK: %[[cst_2:.*]] = arith.constant dense<0> : vector<16xi32>
      //CHECK: %[[r9:.*]] = func.call @llvm.genx.raw.send2.v16i32.v16i1.v16i64(%[[c0_i8]], %[[c4_i8]], %[[cst]], %[[c2_i8]], %[[c1_i8]], %[[c15_i8]], %[[c0_i32]], %[[c68289920_i32]], %[[r5]], %[[cst_2]]) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<16xi32>) -> vector<16xi32>
      //CHECK: %[[r10:.*]] = vector.bitcast %[[r9]] : vector<16xi32> to vector<32xf16>
      //CHECK: %[[c67241348_i32:.*]] = arith.constant 67241348 : i32
      //CHECK: %[[r11:.*]] = vector.bitcast %[[r10]] : vector<32xf16> to vector<16xi32>
      //CHECK: func.call @llvm.genx.raw.sends2.noresult.v16i1.v16i64.v16i32(%[[c0_i8]], %[[c4_i8]], %[[cst]], %[[c2_i8]], %[[c1_i8]], %[[c15_i8]], %[[c0_i32]], %[[c67241348_i32]], %[[r8]], %[[r11]]) : (i8, i8, vector<16xi1>, i8, i8, i8, i32, i32, vector<16xi64>, vector<16xi32>) -> ()
      %out_flat = memref.reinterpret_cast %out to offset: [0], sizes: [32], strides: [1] : memref<16x2xf16> to memref<32xf16>
      %mask = arith.constant dense<[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]> : vector<16xi1>
      %tdesc_in = xegpu.create_tdesc %in[0,4,8,12,16,20,24,28,32,34,38,42,46,50,54,58] : memref<?xf16> -> !xegpu.tensor_desc<16x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
      %tdesc_out = xegpu.create_tdesc %out_flat[0,4,8,12,16,20,24,28,32,34,38,42,46,50,54,58] : memref<32xf16> -> !xegpu.tensor_desc<16x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
      %loaded = xegpu.load %tdesc_in, %mask {transpose} : !xegpu.tensor_desc<16x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1> -> vector<2x16xf16>
      xegpu.store %loaded, %tdesc_out, %mask {transpose} : vector<2x16xf16>, !xegpu.tensor_desc<16x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1>
      gpu.return
    }
  }
}
