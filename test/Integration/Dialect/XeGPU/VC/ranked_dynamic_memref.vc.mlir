// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref_0 = gpu.alloc  host_shared () : memref<8x16xf32>
    memref.copy %A, %memref_0 : memref<8x16xf32> to memref<8x16xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    %memref_0_cast = memref.cast %memref_0 : memref<8x16xf32> to memref<?x?xf32>
    %memref_1_cast = memref.cast %memref_1 : memref<8x16xf32> to memref<?x?xf32>
    %dim0 = arith.constant 8 : index
    %dim1 = arith.constant 16 : index
    %stride0 = arith.constant 16 : index
    %stride1 = arith.constant 1 : index
    %x = arith.constant 0 : index
    %y = arith.constant 0 : index
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_0_cast : memref<?x?xf32>, %memref_1_cast : memref<?x?xf32>, %dim0 : index, %dim1 : index, %stride0 : index, %stride1 : index, %x : index, %y : index)
    gpu.dealloc %memref_0 : memref<8x16xf32>
    return %memref_1 : memref<8x16xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0 : memref<?x?xf32>, %arg1: memref<?x?xf32>, %dim0: index, %dim1: index, %stride0: index, %stride1: index, %x: index, %y: index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %1 = xegpu.create_nd_tdesc %arg0[%x, %y], shape: [%dim0, %dim1], strides: [%stride0, %stride1] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
      %2 = xegpu.load_nd %1 {l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %6 = xegpu.create_nd_tdesc %arg1[%x, %y], shape: [%dim0, %dim1], strides: [%stride0, %stride1] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %2, %6 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf32>
    %A_random = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32

    call @fillResource1DRandomF32(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %B = call @test(%A) : (memref<8x16xf32>) -> memref<8x16xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%A_cast, %B_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
