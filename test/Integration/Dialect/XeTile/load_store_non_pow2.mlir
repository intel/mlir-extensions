// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<384x64xf32>, %B: memref<384x64xf32>) -> memref<384x64xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %A_gpu = gpu.alloc  host_shared () : memref<384x64xf32>
    memref.copy %A, %A_gpu : memref<384x64xf32> to memref<384x64xf32>
    %B_gpu = gpu.alloc  host_shared () : memref<384x64xf32>
    memref.copy %B, %B_gpu : memref<384x64xf32> to memref<384x64xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<384x64xf32>, %B_gpu : memref<384x64xf32>)
    gpu.dealloc %A_gpu : memref<384x64xf32>
    return %B_gpu : memref<384x64xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<384x64xf32>, %B: memref<384x64xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      /// canonicalize
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y

      %a_tile = xetile.init_tile %A[%c0, %c0] : memref<384x64xf32> -> !xetile.tile<384x64xf32>
      %b_tile = xetile.init_tile %B[%c0, %c0] : memref<384x64xf32> -> !xetile.tile<384x64xf32>

      %a_value = xetile.load_tile %a_tile  : !xetile.tile<384x64xf32> -> vector<384x64xf32>
      xetile.store_tile %a_value, %b_tile  : vector<384x64xf32>, !xetile.tile<384x64xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_0_f32 = arith.constant 0.0 : f32
    %cf_2_f32 = arith.constant 2.0 : f32
    %cf_1 = arith.constant 1.0 : f16
    // TRY 385x64
    %A = memref.alloc() : memref<384x64xf32>
    %B = memref.alloc() : memref<384x64xf32>

    // fill A with 2, B with 0
    %A_nonzero = memref.cast %A : memref<384x64xf32> to memref<*xf32>
    %B_zeros = memref.cast %B : memref<384x64xf32> to memref<*xf32>
    call @fillResource1DF32(%A_nonzero, %cf_2_f32) : (memref<*xf32>, f32) -> ()
    call @fillResource1DF32(%B_zeros, %cf_0_f32) : (memref<*xf32>, f32) -> ()
    // Load from A, store to B
    %2 = call @test(%A, %B) : (memref<384x64xf32>, memref<384x64xf32>) -> memref<384x64xf32>

    %B_filled = memref.cast %2 : memref<384x64xf32> to memref<*xf32>
    // call @printMemrefF32(%A_nonzero) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%B_filled) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%A_nonzero, %B_filled) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<384x64xf32>
    memref.dealloc %B : memref<384x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
}
