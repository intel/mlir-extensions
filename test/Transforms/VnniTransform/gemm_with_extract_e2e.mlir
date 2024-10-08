// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_1024x1024xf16 : memref<1024x1024xf16> = dense<0.0>
  memref.global "private" @__constant_1024x1024xf16_ : memref<1024x1024xf16> = dense<0.0>
  memref.global "private" @__constant_1024x1024xf32 : memref<1024x1024xf32> = dense<0.0>
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg0, %memref : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %arg1, %memref_0 : memref<1024x1024xf16> to memref<1024x1024xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_gemm blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %memref_1 : memref<1024x1024xf32>
  }

gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_gemm(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c32 : index
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = arith.addi %0, %c0 : index
    %3 = arith.addi %1, %c0 : index
    %4 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %c16 = arith.constant 16 : index
    %5 = arith.addi %1, %c16 : index
    %6 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %c8 = arith.constant 8 : index
    %7 = arith.addi %0, %c8 : index
    %8 = xegpu.create_nd_tdesc %arg2[%7, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %9 = xegpu.create_nd_tdesc %arg2[%7, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %10 = arith.addi %0, %c16 : index
    %11 = xegpu.create_nd_tdesc %arg2[%10, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %12 = xegpu.create_nd_tdesc %arg2[%10, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %c24 = arith.constant 24 : index
    %13 = arith.addi %0, %c24 : index
    %14 = xegpu.create_nd_tdesc %arg2[%13, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %15 = xegpu.create_nd_tdesc %arg2[%13, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %16 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %17 = xegpu.create_nd_tdesc %arg2[%2, %5] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %18 = xegpu.load_nd %16 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %19 = xegpu.load_nd %17 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %20 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
    %21 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
    %22:4 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %20, %arg5 = %21, %arg6 = %18, %arg7 = %19) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, vector<32x16xf32>, vector<32x16xf32>) {
      %31 = vector.extract_strided_slice %arg6 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %32 = vector.extract_strided_slice %arg6 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %33 = vector.extract_strided_slice %arg6 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %34 = vector.extract_strided_slice %arg6 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %35 = vector.extract_strided_slice %arg7 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %36 = vector.extract_strided_slice %arg7 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %37 = vector.extract_strided_slice %arg7 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %38 = vector.extract_strided_slice %arg7 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      %39 = xegpu.load_nd %arg4 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      %40 = vector.extract %39[0] : vector<32x16xf16> from vector<2x32x16xf16>
      %41 = vector.extract %39[1] : vector<32x16xf16> from vector<2x32x16xf16>
      %42 = vector.extract_strided_slice %40 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %43 = vector.extract_strided_slice %40 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %44 = vector.extract_strided_slice %40 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %45 = vector.extract_strided_slice %40 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %46 = vector.extract_strided_slice %41 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %47 = vector.extract_strided_slice %41 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %48 = vector.extract_strided_slice %41 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %49 = vector.extract_strided_slice %41 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>

      %50 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>

      %51 = vector.extract %50[0] : vector<32x16xf16> from vector<2x32x16xf16>
      %52 = vector.extract %50[1] : vector<32x16xf16> from vector<2x32x16xf16>

      %53 = vector.extract_strided_slice %51 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %54 = vector.extract_strided_slice %51 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %55 = vector.extract_strided_slice %52 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %56 = vector.extract_strided_slice %52 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>

      %61 = xegpu.dpas %42, %53, %31 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %62 = xegpu.dpas %46, %54, %61 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %63 = xegpu.dpas %42, %55, %35 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %64 = xegpu.dpas %46, %56, %63 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %65 = xegpu.dpas %43, %53, %32 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %66 = xegpu.dpas %47, %54, %65 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %67 = xegpu.dpas %43, %55, %36 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %68 = xegpu.dpas %47, %56, %67 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %69 = xegpu.dpas %44, %53, %33 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %70 = xegpu.dpas %48, %54, %69 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %71 = xegpu.dpas %44, %55, %37 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %72 = xegpu.dpas %48, %56, %71 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %73 = xegpu.dpas %45, %53, %34 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %74 = xegpu.dpas %49, %54, %73 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %75 = xegpu.dpas %45, %55, %38 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %76 = xegpu.dpas %49, %56, %75 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %77 = vector.shuffle %62, %66 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %78 = vector.shuffle %70, %74 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %79 = vector.shuffle %77, %78 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      %80 = vector.shuffle %64, %68 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %81 = vector.shuffle %72, %76 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      %82 = vector.shuffle %80, %81 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      %83 = xegpu.update_nd_offset %arg4, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
      %84 = xegpu.update_nd_offset %arg5, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
      scf.yield %83, %84, %79, %82 : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>, vector<32x16xf32>, vector<32x16xf32>
    }
    %23 = vector.extract_strided_slice %22#2 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %24 = vector.extract_strided_slice %22#2 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %25 = vector.extract_strided_slice %22#2 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %26 = vector.extract_strided_slice %22#2 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %27 = vector.extract_strided_slice %22#3 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %28 = vector.extract_strided_slice %22#3 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %29 = vector.extract_strided_slice %22#3 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %30 = vector.extract_strided_slice %22#3 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    xegpu.store_nd %23, %4 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %27, %6 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %24, %8 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %28, %9 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %25, %11 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %29, %12 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %26, %14 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xegpu.store_nd %30, %15 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    gpu.return
  }
}

  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1024x1024xf16 : memref<1024x1024xf16>
    %1 = memref.get_global @__constant_1024x1024xf16_ : memref<1024x1024xf16>
    %ref = memref.get_global @__constant_1024x1024xf32 : memref<1024x1024xf32>
    %init = arith.constant 0.0 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    // fill the top-left block 128x128
    // A matrix: row-major, start from 0.0, increase 0.01 per element
    // B matrix: A matrix + 1.0
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c128 step %c1 {
        %int0 = arith.index_cast %arg0 : index to i16
        %int1 = arith.index_cast %arg1 : index to i16
        %c128_i16 = arith.constant 128 : i16
        %idx0 = arith.muli %int0, %c128_i16 : i16
        %idx1 = arith.addi %int1, %idx0 : i16
        %fp = arith.uitofp %idx1 : i16 to f16
        %cst100 = arith.constant 100.0 : f16
        %val0 = arith.divf %fp, %cst100 : f16
        %cst1 = arith.constant 1.0 : f16
        %val1 = arith.addf %val0, %cst1 : f16
        memref.store %val0, %0[%arg0, %arg1] : memref<1024x1024xf16>
        memref.store %val1, %1[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    // caculate the result C matrix
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %acc = memref.load %ref[%arg0, %arg1] : memref<1024x1024xf32>
        %res = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %acc) -> f32 {
          %a = memref.load %0[%arg0, %arg2] : memref<1024x1024xf16>
          %b = memref.load %1[%arg2, %arg1] : memref<1024x1024xf16>
          %c = arith.mulf %a, %b : f16
          %cc = arith.extf %c : f16 to f32
          %ccc = arith.addf %cc, %arg3 : f32
          scf.yield %ccc : f32
        }
        memref.store %res, %ref[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }

    %2 = call @test(%0, %1) : (memref<1024x1024xf16>, memref<1024x1024xf16>) -> memref<1024x1024xf32>
    %cast = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    %cast_ref = memref.cast %ref : memref<1024x1024xf32> to memref<*xf32>
    //call @printMemrefF32(%cast_ref) : (memref<*xf32>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast, %cast_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
