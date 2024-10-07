// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#scatter = #xegpu.scatter_tdesc_attr<memory_scope=global>

module @gemm attributes {gpu.container_module} {
  func.func @test() -> memref<16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %out = gpu.alloc host_shared () : memref<16xf16>
    gpu.launch_func  @test_kernel::@test_store_scatter blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%out : memref<16xf16>)
    return %out : memref<16xf16>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_store_scatter(%mem: memref<16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : vector<16xf16>

      %mask = arith.constant dense<1> : vector<16xi1>
      %tdesc = xegpu.create_tdesc %mem[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : memref<16xf16> -> !xegpu.tensor_desc<16xf16, #scatter>
      xegpu.store %cst, %tdesc, %mask : vector<16xf16>, !xegpu.tensor_desc<16xf16, #scatter>, vector<16xi1>

      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %B = call @test() : () -> memref<16xf16>
    %cast = memref.cast %B : memref<16xf16> to memref<*xf16>
    //CHECK: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15]
    call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    return
  }

  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
}
