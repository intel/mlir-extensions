// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=opencl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A : memref<8x16xf16>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref_0 = gpu.alloc  host_shared () : memref<8x16xf16>
    memref.copy %A, %memref_0 : memref<8x16xf16> to memref<8x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    %memref_0_cast = memref.cast %memref_0 : memref<8x16xf16> to memref<?x?xf16>
    %memref_1_cast = memref.cast %memref_1 : memref<8x16xf32> to memref<?x?xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_0_cast : memref<?x?xf16>, %memref_1_cast : memref<?x?xf32>)
    gpu.dealloc %memref_0 : memref<8x16xf16>
    return %memref_1 : memref<8x16xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
   gpu.func @test_kernel(%arg0 : memref<?x?xf16>, %arg1: memref<?x?xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %1 = xegpu.create_nd_tdesc %arg0[0, 0], [%c8, %c16], [%c16, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>
      %2 = xegpu.load_nd %1 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %3 = vector.shape_cast %2 : vector<8x16xf16> to vector<128xf16>
      %5 = arith.extf %3 : vector<128xf16> to vector<128xf32>
      %4 = vector.shape_cast %5 : vector<128xf32> to vector<8x16xf32>
      %6 = xegpu.create_nd_tdesc %arg1[0, 0], [%c8, %c16], [%c16, %c1] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %4, %6 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf16>
    %A_random = memref.cast %A : memref<8x16xf16> to memref<*xf16>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32

    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    %B = call @test(%A) : (memref<8x16xf16>) -> memref<8x16xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf16> to memref<*xf16>
    // call @printMemrefF16(%A_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%A_cast, %B_cast) : (memref<*xf16>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<8x16xf16>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
