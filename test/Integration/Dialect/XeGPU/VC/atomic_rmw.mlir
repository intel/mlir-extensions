// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#scatter = #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>

module @gemm attributes {gpu.container_module} {
  func.func @test(%a: memref<16xf32>) -> memref<16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %a_gpu = gpu.alloc host_shared () : memref<16xf32>
    memref.copy %a, %a_gpu : memref<16xf32> to memref<16xf32>
    %out = gpu.alloc host_shared () : memref<16xf32>
    gpu.launch_func  @test_kernel::@test_atomic_rmw blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%a_gpu: memref<16xf32>, %out : memref<16xf32>)
    return %a_gpu : memref<16xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_atomic_rmw(%input: memref<16xf32>, %mem: memref<16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : vector<16xf32>
      %mask = arith.constant dense<1> : vector<16xi1>
      %offsets = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %in_tdesc = xegpu.create_tdesc %input, %offsets : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #scatter>
      %atomic_rmw = xegpu.atomic_rmw addf %in_tdesc, %mask, %cst : !xegpu.tensor_desc<16xf32, #scatter>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
      %out_tdesc = xegpu.create_tdesc %mem, %offsets : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #scatter>
      xegpu.store %atomic_rmw, %out_tdesc, %mask : vector<16xf32>, !xegpu.tensor_desc<16xf32, #scatter>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %a = memref.alloc() : memref<16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1_f32 = arith.constant 1.0 : f32
    scf.for %i = %c0 to %c16 step %c1 {
        memref.store %c1_f32, %a[%i] : memref<16xf32>
    }

    %B = call @test(%a) : (memref<16xf32>) -> memref<16xf32>
    %cast = memref.cast %B : memref<16xf32> to memref<*xf32>
    //CHECK: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15, 16]
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
