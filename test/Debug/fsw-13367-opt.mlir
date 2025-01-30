gpu.module @bias_add attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @bias_add(%arg0: memref<16x64xf32>, %arg1: memref<1x1xf32>, %arg2: memref<16x64xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst = arith.constant dense<true> : vector<1xi1>
    %cst_0 = arith.constant dense<0> : vector<1xindex>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<1x16xf32>
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %thread_id_x = gpu.thread_id  x
    %0 = arith.remsi %thread_id_x, %c64 : index
    %1 = arith.divsi %thread_id_x, %c64 : index
    %2 = arith.muli %1, %c64 : index
    %3 = arith.addi %2, %0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] : memref<1x1xf32> to memref<1xf32>
    %4 = xegpu.create_tdesc %reinterpret_cast, %cst_0 : memref<1xf32>, vector<1xindex> -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>
    %5 = xegpu.load %4, %cst <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<memory_space =  global, chunk_size = 1 : i64>>, vector<1xi1> -> vector<1xf32>
    %6 = vector.shape_cast %5 : vector<1xf32> to vector<1x1xf32>
    %7 = vector.insert_strided_slice %6, %cst_1 {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %8 = vector.insert_strided_slice %6, %7 {offsets = [0, 1], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %9 = vector.insert_strided_slice %6, %8 {offsets = [0, 2], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %10 = vector.insert_strided_slice %6, %9 {offsets = [0, 3], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %11 = vector.insert_strided_slice %6, %10 {offsets = [0, 4], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %12 = vector.insert_strided_slice %6, %11 {offsets = [0, 5], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %13 = vector.insert_strided_slice %6, %12 {offsets = [0, 6], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %14 = vector.insert_strided_slice %6, %13 {offsets = [0, 7], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %15 = vector.insert_strided_slice %6, %14 {offsets = [0, 8], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %16 = vector.insert_strided_slice %6, %15 {offsets = [0, 9], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %17 = vector.insert_strided_slice %6, %16 {offsets = [0, 10], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %18 = vector.insert_strided_slice %6, %17 {offsets = [0, 11], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %19 = vector.insert_strided_slice %6, %18 {offsets = [0, 12], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %20 = vector.insert_strided_slice %6, %19 {offsets = [0, 13], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %21 = vector.insert_strided_slice %6, %20 {offsets = [0, 14], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %22 = vector.insert_strided_slice %6, %21 {offsets = [0, 15], strides = [1, 1]} : vector<1x1xf32> into vector<1x16xf32>
    %23 = arith.remsi %3, %c4 : index
    %24 = arith.divsi %3, %c4 : index
    %25 = arith.remsi %24, %c16 : index
    %26 = arith.remsi %23, %c4 : index
    %27 = arith.muli %26, %c16 : index
    %28 = xegpu.create_nd_tdesc %arg0[%25, %27] : memref<16x64xf32> -> !xegpu.tensor_desc<1x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %29 = xegpu.load_nd %28 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf32>
    %30 = arith.addf %22, %29 : vector<1x16xf32>
    %31 = xegpu.create_nd_tdesc %arg2[%25, %27] : memref<16x64xf32> -> !xegpu.tensor_desc<1x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %30, %31 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x16xf32>, !xegpu.tensor_desc<1x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}
