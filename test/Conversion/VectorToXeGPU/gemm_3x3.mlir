// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/vector-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/vector-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map = affine_map<(d0, d1) -> (0)>
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3x3xi32_1 : memref<3x3xi32> = dense<1>
  memref.global "private" constant @__constant_3x3xi32_0 : memref<3x3xi32> = dense<[[10, 11, 12], [13, 14, 15], [16, 17, 18]]>
  memref.global "private" constant @__constant_3x3xi32 : memref<3x3xi32> = dense<[[1, 1, 1], [1, 1, 2], [3, 3, 3]]>
  func.func @main() {
    %0 = memref.get_global @__constant_3x3xi32 : memref<3x3xi32>
    %1 = memref.get_global @__constant_3x3xi32_0 : memref<3x3xi32>
    %2 = memref.get_global @__constant_3x3xi32_1 : memref<3x3xi32>
    %3 = call @test(%0, %1, %2) : (memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) -> memref<3x3xi32>
    %cast = memref.cast %3 : memref<3x3xi32> to memref<*xi32>
    call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [40,   43,   46]
    // CHECK-NEXT: [56,   60,   64]
    // CHECK-NEXT: [118,   127,   136]
    return
  }
  func.func private @printMemrefI32(memref<*xi32>)
  func.func @test(%arg0: memref<3x3xi32>, %arg1: memref<3x3xi32>, %arg2: memref<3x3xi32>) -> memref<3x3xi32> {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc  host_shared () : memref<3x3xi32>
    memref.copy %arg2, %memref : memref<3x3xi32> to memref<3x3xi32>
    %memref_0 = gpu.alloc  host_shared () : memref<3x3xi32>
    memref.copy %arg1, %memref_0 : memref<3x3xi32> to memref<3x3xi32>
    %memref_1 = gpu.alloc  host_shared () : memref<3x3xi32>
    memref.copy %arg0, %memref_1 : memref<3x3xi32> to memref<3x3xi32>
    %memref_2 = gpu.alloc  host_shared () : memref<3x3xi32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c3, %c1, %c1) threads in (%c1, %c1, %c1)  args(%c0 : index, %c16 : index, %c0 : index, %memref_2 : memref<3x3xi32>)
    %memref_3 = gpu.alloc  host_shared () : memref<3x3xi32>
    memref.copy %memref_2, %memref_3 : memref<3x3xi32> to memref<3x3xi32>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c3, %c1, %c1) threads in (%c1, %c1, %c1)  args(%c0 : index, %c16 : index, %c0 : index, %memref_1 : memref<3x3xi32>, %c0 : index, %memref_0 : memref<3x3xi32>, %memref_3 : memref<3x3xi32>)
    %memref_4 = gpu.alloc  host_shared () : memref<3x3xi32>
    gpu.launch_func  @test_kernel_1::@test_kernel blocks in (%c3, %c1, %c1) threads in (%c1, %c1, %c1)  args(%c0 : index, %c16 : index, %c0 : index, %memref_3 : memref<3x3xi32>, %memref : memref<3x3xi32>, %memref_4 : memref<3x3xi32>)
    gpu.dealloc  %memref_2 : memref<3x3xi32>
    gpu.dealloc  %memref_3 : memref<3x3xi32>
    %alloc = memref.alloc() : memref<3x3xi32>
    memref.copy %memref_4, %alloc : memref<3x3xi32> to memref<3x3xi32>
    gpu.dealloc  %memref_4 : memref<3x3xi32>
    gpu.dealloc  %memref_1 : memref<3x3xi32>
    gpu.dealloc  %memref_0 : memref<3x3xi32>
    gpu.dealloc  %memref : memref<3x3xi32>
    return %alloc : memref<3x3xi32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorComputeINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_INTEL_vector_compute, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<3x3xi32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<0> : vector<16xi32>
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = arith.addi %arg0, %0 : index
      %3 = arith.muli %arg1, %1 : index
      %4 = arith.addi %arg2, %3 : index
      vector.transfer_write %cst, %arg3[%2, %4] : vector<16xi32>, memref<3x3xi32>
      gpu.return
    }
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorComputeINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_INTEL_vector_compute, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<3x3xi32>, %arg4: index, %arg5: memref<3x3xi32>, %arg6: memref<3x3xi32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = arith.addi %arg0, %0 : index
      %3 = arith.muli %arg1, %1 : index
      %4 = arith.addi %arg2, %3 : index
      %5 = vector.transfer_read %arg3[%2, %arg4], %c0_i32 {permutation_map = #map} : memref<3x3xi32>, vector<16xi32>
      %6 = vector.transfer_read %arg5[%arg4, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %7 = vector.transfer_read %arg6[%2, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %8 = arith.muli %5, %6 : vector<16xi32>
      %9 = arith.addi %7, %8 : vector<16xi32>
      vector.transfer_write %9, %arg6[%2, %4] : vector<16xi32>, memref<3x3xi32>
      %10 = vector.transfer_read %arg3[%2, %c1], %c0_i32 {permutation_map = #map} : memref<3x3xi32>, vector<16xi32>
      %11 = vector.transfer_read %arg5[%c1, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %12 = vector.transfer_read %arg6[%2, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %13 = arith.muli %10, %11 : vector<16xi32>
      %14 = arith.addi %12, %13 : vector<16xi32>
      vector.transfer_write %14, %arg6[%2, %4] : vector<16xi32>, memref<3x3xi32>
      %15 = vector.transfer_read %arg3[%2, %c2], %c0_i32 {permutation_map = #map} : memref<3x3xi32>, vector<16xi32>
      %16 = vector.transfer_read %arg5[%c2, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %17 = vector.transfer_read %arg6[%2, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %18 = arith.muli %15, %16 : vector<16xi32>
      %19 = arith.addi %17, %18 : vector<16xi32>
      vector.transfer_write %19, %arg6[%2, %4] : vector<16xi32>, memref<3x3xi32>
      gpu.return
    }
  }
  gpu.module @test_kernel_1 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorComputeINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_INTEL_vector_compute, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<3x3xi32>, %arg4: memref<3x3xi32>, %arg5: memref<3x3xi32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0_i32 = arith.constant 0 : i32
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = arith.addi %arg0, %0 : index
      %3 = arith.muli %arg1, %1 : index
      %4 = arith.addi %arg2, %3 : index
      %5 = vector.transfer_read %arg3[%2, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %6 = vector.transfer_read %arg4[%2, %4], %c0_i32 : memref<3x3xi32>, vector<16xi32>
      %7 = arith.addi %5, %6 : vector<16xi32>
      vector.transfer_write %7, %arg5[%2, %4] : vector<16xi32>, memref<3x3xi32>
      gpu.return
    }
  }
}