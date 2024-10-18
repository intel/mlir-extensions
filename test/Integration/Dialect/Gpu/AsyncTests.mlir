// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/gpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/gpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @eltwise_add attributes {gpu.container_module} {
  func.func @fillRandom(%arg0: memref<4194304xf32>, %arg0_gpu: memref<4194304xf32>) -> () {
    %S0L = arith.constant 10.0 : f32
    %S0H = arith.constant 50.0 : f32
    %false = arith.constant 0 : i1

    %arg0_random = memref.cast %arg0 : memref<4194304xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%arg0_random, %S0L, %S0H, %false) : (memref<*xf32>, f32, f32, i1) -> ()

    memref.copy %arg0, %arg0_gpu : memref<4194304xf32> to memref<4194304xf32>

    return
  }

  func.func @fillZeros(%res: memref<4194304xf32>, %res_gpu: memref<4194304xf32>) -> () {
    %c0 = arith.constant 0.0 : f32

    %res_zeros = memref.cast %res : memref<4194304xf32> to memref<*xf32>
    call @fillResource1DF32(%res_zeros, %c0) : (memref<*xf32>, f32) -> ()

    memref.copy %res, %res_gpu : memref<4194304xf32> to memref<4194304xf32>

    return
  }

  gpu.module @eltwiseAdd_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @eltwiseAdd_kernel(%arg0: memref<4194304xf32>, %arg1: memref<4194304xf32>, %arg2: memref<4194304xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %global_id_x = gpu.global_id x
      %cst = arith.constant 0.5 : f32
      %0 = memref.load %arg0[%global_id_x] : memref<4194304xf32>
      %1 = memref.load %arg1[%global_id_x] : memref<4194304xf32>
      %2 = arith.addf %0, %1 : f32
      %3 = arith.addf %2, %cst : f32
      memref.store %3, %arg2[%global_id_x] : memref<4194304xf32>
      gpu.return
    }
  }

  // compute CPU reference (takes minutes)
  func.func @cpu_reference(%arg0: memref<4194304xf32>, %arg1: memref<4194304xf32>, %arg2: memref<4194304xf32>) {
    %c4194304 = arith.constant 4194304 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 5.000000e-01 : f32
    scf.for %i = %c0 to %c4194304 step %c1 {
      %0 = memref.load %arg0[%i] : memref<4194304xf32>
      %1 = memref.load %arg1[%i] : memref<4194304xf32>
      %2 = arith.addf %0, %1 : f32
      %3 = arith.addf %2, %cst : f32
      memref.store %3, %arg2[%i] : memref<4194304xf32>
    }
    return
  }

  func.func @main() {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c8192 = arith.constant 8192 : index

    %arg0 = memref.alloc() : memref<4194304xf32>
    %arg0_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillRandom(%arg0, %arg0_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    %arg1 = memref.alloc() : memref<4194304xf32>
    %arg1_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillRandom(%arg1, %arg1_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    %arg2 = memref.alloc() : memref<4194304xf32>
    %arg2_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillRandom(%arg2, %arg2_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    %arg3 = memref.alloc() : memref<4194304xf32>
    %arg3_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillRandom(%arg3, %arg3_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    %res0 = memref.alloc() : memref<4194304xf32>
    %res0_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillZeros(%res0, %res0_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    %res1 = memref.alloc() : memref<4194304xf32>
    %res1_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillZeros(%res1, %res1_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    %res2 = memref.alloc() : memref<4194304xf32>
    %res2_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillZeros(%res2, %res2_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    %res = memref.alloc() : memref<4194304xf32>
    %res_gpu = gpu.alloc  host_shared () : memref<4194304xf32>
    call @fillZeros(%res, %res_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    // Test1: Two async launches followed by sync launch that
    //        waits for events returned by the two async launches

    %e1 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg0_gpu: memref<4194304xf32>, %arg1_gpu: memref<4194304xf32>, %res0_gpu: memref<4194304xf32>)
    %e2 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg2_gpu: memref<4194304xf32>, %arg3_gpu: memref<4194304xf32>, %res1_gpu: memref<4194304xf32>)
    gpu.launch_func [%e1, %e2] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%res0_gpu: memref<4194304xf32>, %res1_gpu: memref<4194304xf32>, %res_gpu: memref<4194304xf32>)

    call @cpu_reference(%arg0, %arg1, %res0) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%arg2, %arg3, %res1) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%res0, %res1, %res) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()

    %cast_res = memref.cast %res : memref<4194304xf32> to memref<*xf32>
    %cast_res_gpu = memref.cast %res_gpu : memref<4194304xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_res, %cast_res_gpu) : (memref<*xf32>, memref<*xf32>) -> ()

    call @fillZeros(%res0, %res0_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%res1, %res1_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%res, %res_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    // Test2: An async launch followed by another async launch and
    //        finally a sync launch. Each launch waits on the event
    //        from the preceeding launch.

    %e3 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg0_gpu: memref<4194304xf32>, %arg1_gpu: memref<4194304xf32>, %res0_gpu: memref<4194304xf32>)
    %e4 = gpu.launch_func async [%e3] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg2_gpu: memref<4194304xf32>, %res0_gpu: memref<4194304xf32>, %res1_gpu: memref<4194304xf32>)
    gpu.launch_func [%e4] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg3_gpu: memref<4194304xf32>, %res1_gpu: memref<4194304xf32>, %res_gpu: memref<4194304xf32>)

    call @cpu_reference(%arg0, %arg1, %res0) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%arg2, %res0, %res1) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%arg3, %res1, %res) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()

    %cast_res_0 = memref.cast %res : memref<4194304xf32> to memref<*xf32>
    %cast_res_gpu_0 = memref.cast %res_gpu : memref<4194304xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_res_0, %cast_res_gpu_0) : (memref<*xf32>, memref<*xf32>) -> ()

    call @fillZeros(%res0, %res0_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%res1, %res1_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%res, %res_gpu) : (memref<4194304xf32>, memref<4194304xf32>) -> ()

    // Test3: An async launch followed by two async launches and
    //        finally a sync launch. The event from the first async launch
    //        is passed to the subsequent two async launches which wait on
    //        the same event. The last sync launch waits from two events
    //        from the preceeding two async launches.

    %e5 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg0_gpu: memref<4194304xf32>, %arg1_gpu: memref<4194304xf32>, %res0_gpu: memref<4194304xf32>)
    %e6 = gpu.launch_func async [%e5] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg2_gpu: memref<4194304xf32>, %res0_gpu: memref<4194304xf32>, %res1_gpu: memref<4194304xf32>)
    %e7 = gpu.launch_func async [%e5] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg3_gpu: memref<4194304xf32>, %res0_gpu: memref<4194304xf32>, %res2_gpu: memref<4194304xf32>)
    gpu.launch_func [%e6, %e7] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1) args(%res1_gpu: memref<4194304xf32>, %res2_gpu: memref<4194304xf32>, %res_gpu: memref<4194304xf32>)

    call @cpu_reference(%arg0, %arg1, %res0) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%arg2, %res0, %res1) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%arg3, %res0, %res2) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%res1, %res2, %res) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()

    %cast_res_1 = memref.cast %res : memref<4194304xf32> to memref<*xf32>
    %cast_res_gpu_1 = memref.cast %res_gpu : memref<4194304xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_res_1, %cast_res_gpu_1) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %arg0 : memref<4194304xf32>
    memref.dealloc  %arg1 : memref<4194304xf32>
    memref.dealloc  %arg2 : memref<4194304xf32>
    memref.dealloc  %arg3 : memref<4194304xf32>
    memref.dealloc  %res0 : memref<4194304xf32>
    memref.dealloc  %res1 : memref<4194304xf32>
    memref.dealloc  %res : memref<4194304xf32>

    gpu.dealloc  %arg0_gpu : memref<4194304xf32>
    gpu.dealloc  %arg1_gpu : memref<4194304xf32>
    gpu.dealloc  %arg2_gpu : memref<4194304xf32>
    gpu.dealloc  %arg3_gpu : memref<4194304xf32>
    gpu.dealloc  %res0_gpu : memref<4194304xf32>
    gpu.dealloc  %res1_gpu : memref<4194304xf32>
    gpu.dealloc  %res_gpu : memref<4194304xf32>

    return
  }

  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
