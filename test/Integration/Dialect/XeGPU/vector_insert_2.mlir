// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<8x16xf32> ) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf32>
    memref.copy %A, %memref : memref<8x16xf32> to memref<8x16xf32>
    %memref_2 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @module0::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf32>, %memref_2 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf32>
    return %memref_2 : memref<8x16xf32>
  }

    gpu.module @module0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<8x16xf32>, %Out: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load tile
      %a_tile0 = xegpu.create_nd_tdesc %A [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %val0 = xegpu.load_nd %a_tile0 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      // define const vector
      %cst = arith.constant dense<1.23> : vector<16xf32>
      // insert row at pos 7
      %val3 = vector.insert %cst, %val0 [7] : vector<16xf32> into vector<8x16xf32>
      // store
      %out_tile = xegpu.create_nd_tdesc %Out [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %val3, %out_tile  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1_f32 = arith.constant 1.0 : f32
    %c2_f32 = arith.constant 2.0 : f32
    %cst = arith.constant 1.23 : f32
    // random init
    %lower = arith.constant -3.0 : f32
    %upper = arith.constant 3.0 : f32
    %false = arith.constant 0 : i1
    %A = memref.alloc() : memref<8x16xf32>
    %Out_cpu = memref.alloc() : memref<8x16xf32>
    %A_random = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%A_random, %lower, %upper, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    // run GPU version
    %Out_gpu = call @test(%A) : (memref<8x16xf32>) -> memref<8x16xf32>
    %Out_gpu_cast = memref.cast %Out_gpu : memref<8x16xf32> to memref<*xf32>

    // run CPU version
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %A[%i, %j] : memref<8x16xf32>
        memref.store %v, %Out_cpu[%i, %j] : memref<8x16xf32>
      }
    }
    scf.for %i = %c0 to %c16 step %c1 {
      memref.store %cst, %Out_cpu[%c7, %i] : memref<8x16xf32>
    }

    %Out_cpu_cast = memref.cast %Out_cpu : memref<8x16xf32> to memref<*xf32>
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%Out_gpu_cast, %Out_cpu_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    // dealloc
    memref.dealloc %A : memref<8x16xf32>
    memref.dealloc %Out_cpu : memref<8x16xf32>
    // gpu dealloc
    gpu.dealloc %Out_gpu : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
