// RUN: imex-opt -convert-xegpu-to-vc -cse  %s | FileCheck %s

gpu.module @load_store_bf16 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @load_store_bf16(%arg0: memref<4x2x128xbf16>, %arg1: memref<4x2x128xbf16>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 4, 2, 4>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c32 = arith.constant 32 : index
    %thread_id_x = gpu.thread_id x
    %thread_id_y = gpu.thread_id y
    %thread_id_z = gpu.thread_id z
    %0 = arith.muli %thread_id_z, %c32 : index
    %1 = xegpu.create_nd_tdesc %arg0[%thread_id_x, %thread_id_y, %0], shape: [4, 2, 128], strides: [256, 128, 1] : memref<4x2x128xbf16> -> !xegpu.tensor_desc<32xbf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>>

    // CHECK: func.call @llvm.genx.lsc.prefetch.stateless.v1i1.v1i64
    xegpu.prefetch_nd %1 : !xegpu.tensor_desc<32xbf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>>
    %2 = xegpu.create_nd_tdesc %arg0[%thread_id_x, %thread_id_y, %0],  shape: [4, 2, 128], strides: [256, 128, 1] : memref<4x2x128xbf16> -> !xegpu.tensor_desc<32xbf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>>

    // CHECK: %[[LOAD_VAL:.*]] = func.call @llvm.genx.lsc.load.stateless.v16i32.v1i1.v1i64
    // CHECK: %[[REAL_VAL:.*]] = vector.bitcast %[[LOAD_VAL]] : vector<16xi32> to vector<32xbf16>
    %3 = xegpu.load_nd %2 : !xegpu.tensor_desc<32xbf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>> -> vector<32xbf16>
    %4 = xegpu.create_nd_tdesc %arg1[%thread_id_x, %thread_id_y, %0], shape: [4, 2, 128], strides: [256, 128, 1] : memref<4x2x128xbf16> -> !xegpu.tensor_desc<32xbf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>>

    // CHECK: %[[STORE_VAL:.*]] = vector.bitcast %[[REAL_VAL]] : vector<32xbf16> to vector<16xi32>
    // CHECK: func.call @llvm.genx.lsc.store.stateless.v1i1.v1i64.v16i32
    // CHECK: %[[STORE_VAL]], %[[LAST_ARG:.*]]) :
    xegpu.store_nd %3, %4 : vector<32xbf16>, !xegpu.tensor_desc<32xbf16, #xegpu.block_tdesc_attr<memory_space = global, array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}
