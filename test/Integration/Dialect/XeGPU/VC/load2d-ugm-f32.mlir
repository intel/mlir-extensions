// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  // memref.global "private" constant @__constant_8x16xf32 : memref<16x32xf32> = dense<1.0>
  memref.global "private" constant @__constant_8x16xf32 : memref<16x32xf32> = dense<1.0>

  func.func @test(%arg0: memref<16x32xf32>) -> memref<16x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<16x32xf32>
    memref.copy %arg0, %memref : memref<16x32xf32> to memref<16x32xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf32>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<16x32xf32>, %memref_1 : memref<16x32xf32>)

    gpu.dealloc  %memref : memref<16x32xf32>
    return %memref_1 : memref<16x32xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_copy(%arg0: memref<16x32xf32>, %arg1: memref<16x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %2 = xegpu.create_nd_tdesc %arg1[2, 2] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      xegpu.store_nd %3,%2 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf32 : memref<16x32xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %2 = call @test(%0) : (memref<16x32xf32>) -> memref<16x32xf32>

    %cast = memref.cast %2: memref<16x32xf32> to memref<*xf32>

    //CHECK-COUNT-2: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    //CHECK-COUNT-8: [0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    //CHECK-COUNT-6: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    call @printMemrefF32(%cast): (memref<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
