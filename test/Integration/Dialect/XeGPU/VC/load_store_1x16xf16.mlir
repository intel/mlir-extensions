// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1x32xf16>) -> memref<1x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<1x32xf16>
    memref.copy %arg0, %memref : memref<1x32xf16> to memref<1x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<1x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<1x32xf16>, %memref_1 : memref<1x32xf32>)
    gpu.dealloc  %memref : memref<1x32xf16>
    return %memref_1 : memref<1x32xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1x32xf16>, %arg1: memref<1x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %src_tdesc_0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<1x32xf16> -> !xegpu.tensor_desc<1x16xf16>
      %src_tdesc_1 = xegpu.create_nd_tdesc %arg0[0, 16] : memref<1x32xf16> -> !xegpu.tensor_desc<1x16xf16>

      %src_loaded_0 = xegpu.load_nd %src_tdesc_0 : !xegpu.tensor_desc<1x16xf16> -> vector<1x16xf16>
      %src_loaded_1 = xegpu.load_nd %src_tdesc_1 : !xegpu.tensor_desc<1x16xf16> -> vector<1x16xf16>

      %src_loaded_0_f32 = arith.extf %src_loaded_0: vector<1x16xf16> to vector<1x16xf32>
      %src_loaded_1_f32 = arith.extf %src_loaded_1: vector<1x16xf16> to vector<1x16xf32>

      %dest_tdesc_0 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<1x32xf32> -> !xegpu.tensor_desc<1x16xf32>
      %dest_tdesc_1 = xegpu.create_nd_tdesc %arg1[0, 16] : memref<1x32xf32> -> !xegpu.tensor_desc<1x16xf32>

      xegpu.store_nd %src_loaded_0_f32, %dest_tdesc_0 : vector<1x16xf32>, !xegpu.tensor_desc<1x16xf32>
      xegpu.store_nd %src_loaded_1_f32, %dest_tdesc_1 : vector<1x16xf32>, !xegpu.tensor_desc<1x16xf32>

      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<1x32xf16> // 1x32 to ensure surface pitch >= 64
    %A_random = memref.cast %A : memref<1x32xf16> to memref<*xf16>
    %c_gen_int = arith.constant 1 : i1
    %cf_lower = arith.constant -2.0 : f32
    %cf_upper = arith.constant 2.0 : f32
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    %B = call @test(%A) : (memref<1x32xf16>) -> memref<1x32xf32>
    %A_cast = memref.cast %A : memref<1x32xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<1x32xf32> to memref<*xf32>
    // call @printMemrefF16(%A_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%A_cast, %B_cast) : (memref<*xf16>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
