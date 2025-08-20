// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_512xf32 : memref<512xf32> = dense<0.0>
  func.func @test(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %memref = gpu.alloc  host_shared () : memref<512xf32>
    memref.copy %arg0, %memref : memref<512xf32> to memref<512xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<512xf32>
    memref.copy %arg1, %memref_1 : memref<512xf32> to memref<512xf32>
    %memref_2 = gpu.alloc  host_shared () : memref<512xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%memref : memref<512xf32>, %memref_1 : memref<512xf32>, %memref_2 : memref<512xf32>)
    gpu.dealloc  %memref : memref<512xf32>
    gpu.dealloc  %memref_1 : memref<512xf32>
    return %memref_2 : memref<512xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %thread_id_x = gpu.thread_id x
      %c16 = arith.constant 16 : index
      cf.br ^bb1
    ^bb1:
      %t = arith.muli %thread_id_x, %c16 : index
      %0 = xegpu.create_nd_tdesc %arg1[%t]: memref<512xf32> -> !xegpu.tensor_desc<16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %2 = xegpu.create_nd_tdesc %arg0[%t] : memref<512xf32> -> !xegpu.tensor_desc<16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %4 = arith.addf %3, %1 : vector<16xf32>
      %5 = xegpu.create_nd_tdesc %arg2[%t] : memref<512xf32> -> !xegpu.tensor_desc<16xf32>
      xegpu.store_nd %4, %5  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -1. : f32
    %cf_upper = arith.constant 1. : f32

    %A = memref.alloc() : memref<512xf32>
    %A_random = memref.cast %A : memref<512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    %B = memref.alloc() : memref<512xf32>
    %B_random = memref.cast %B : memref<512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf32>, f32, f32, i1) -> ()

    // calculate the result of C vector
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %ref = memref.alloc() : memref<512xf32>
    scf.for %i = %c0 to %c512 step %c1 {
      %a = memref.load %A[%i] : memref<512xf32>
      %b = memref.load %B[%i] : memref<512xf32>
      %c = arith.addf %a, %b : f32
      memref.store %c, %ref[%i] : memref<512xf32>
    }

    %C = call @test(%A, %B) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>

    %C_cast = memref.cast %C : memref<512xf32> to memref<*xf32>
    %ref_cast = memref.cast %ref : memref<512xf32> to memref<*xf32>
    call @printMemrefF32(%ref_cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%ref_cast, %C_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
