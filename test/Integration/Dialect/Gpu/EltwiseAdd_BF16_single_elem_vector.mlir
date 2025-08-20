// RUN: IMEX_USE_IGC_VECTOR_BACK_END=1 %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/gpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner  -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: IMEX_USE_IGC_VECTOR_BACK_END=1 %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/gpu-to-llvm.pp \
// RUN:                                        --runner mlir-runner  -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @eltwise_add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_10x20xbf16 : memref<10x20xbf16> = dense<5.000000e-01>
  func.func @test(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>) -> memref<10x20xbf16> {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<10x20xbf16>
    memref.copy %arg1, %memref : memref<10x20xbf16> to memref<10x20xbf16>
    %memref_0 = gpu.alloc  host_shared () : memref<10x20xbf16>
    memref.copy %arg0, %memref_0 : memref<10x20xbf16> to memref<10x20xbf16>
    %memref_1 = gpu.alloc  host_shared () : memref<10x20xbf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1)  args(%memref_0 : memref<10x20xbf16>, %memref : memref<10x20xbf16>, %memref_1 : memref<10x20xbf16>)
    %alloc = memref.alloc() : memref<10x20xbf16>
    memref.copy %memref_1, %alloc : memref<10x20xbf16> to memref<10x20xbf16>
    gpu.dealloc  %memref_1 : memref<10x20xbf16>
    gpu.dealloc  %memref_0 : memref<10x20xbf16>
    gpu.dealloc  %memref : memref<10x20xbf16>
    return %alloc : memref<10x20xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL, Bfloat16ConversionINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute, SPV_INTEL_bfloat16_conversion]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>, %arg2: memref<10x20xbf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %cst = arith.constant dense<0.5> : vector<1xbf16>
      %0 = memref.load %arg0[%block_id_x, %block_id_y] : memref<10x20xbf16>
      %1 = memref.load %arg1[%block_id_x, %block_id_y] : memref<10x20xbf16>
      %vec_0 = vector.from_elements %0 : vector<1xbf16>
      %vec_1 = vector.from_elements %1 : vector<1xbf16>
      %2 = arith.addf %vec_0, %vec_1 : vector<1xbf16>
      %3 = arith.addf %2, %cst : vector<1xbf16>
      vector.store %3, %arg2[%block_id_x, %block_id_y] : memref<10x20xbf16>, vector<1xbf16>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_10x20xbf16 : memref<10x20xbf16>
    %1 = memref.get_global @__constant_10x20xbf16 : memref<10x20xbf16>
    %2 = call @test(%0, %1) : (memref<10x20xbf16>, memref<10x20xbf16>) -> memref<10x20xbf16>
    %cast = memref.cast %2 : memref<10x20xbf16> to memref<*xbf16>
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-200: 1.5
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    return
  }
   func.func private @printMemrefBF16(memref<*xbf16>)  attributes {llvm.emit_c_interface}
}
