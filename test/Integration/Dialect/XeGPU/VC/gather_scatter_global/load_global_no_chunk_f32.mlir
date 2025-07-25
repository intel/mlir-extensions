// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#scatter = #xegpu.scatter_tdesc_attr<memory_space=global>
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<16xf32>) -> memref<16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index

    %in = gpu.alloc host_shared () : memref<16xf32>
    memref.copy %arg0, %in : memref<16xf32> to memref<16xf32>

    %out = gpu.alloc host_shared () : memref<16xf32>

    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%in : memref<16xf32>, %out : memref<16xf32>)

    gpu.dealloc  %in : memref<16xf32>
    return %out : memref<16xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_copy(%a: memref<16xf32>, %b: memref<16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

      %mask = arith.constant dense<1> : vector<16xi1>
      %offsets = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>

      // load from a using load_gather
      %a_tdesc = xegpu.create_tdesc %a, %offsets : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #scatter>
      xegpu.prefetch %a_tdesc : !xegpu.tensor_desc<16xf32, #scatter>
      %data = xegpu.load %a_tdesc, %mask : !xegpu.tensor_desc<16xf32, #scatter>, vector<16xi1> -> vector<16xf32>

      // store to b using store_scatter
      %b_tdesc = xegpu.create_tdesc %b, %offsets : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #scatter>
      xegpu.store %data, %b_tdesc, %mask : vector<16xf32>, !xegpu.tensor_desc<16xf32, #scatter>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1: index
    %c16 = arith.constant 16: index
    %A = memref.alloc() : memref<16xf32>

    scf.for %i = %c0 to %c16 step %c1 {
      %i32 = index.castu %i : index to i32
      %f32 = arith.sitofp %i32 : i32 to f32
      memref.store %f32, %A[%i] : memref<16xf32>
    }

    %B = call @test(%A) : (memref<16xf32>) -> memref<16xf32>
    %A_cast = memref.cast %A : memref<16xf32> to memref<*xf32>
    %B_cast = memref.cast %B : memref<16xf32> to memref<*xf32>

    // call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    //CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%A_cast, %B_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<16xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
