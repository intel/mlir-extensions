// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#scatter = #xegpu.scatter_tdesc_attr<memory_space=global>
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<16xf16>) -> memref<16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index

    %in = gpu.alloc host_shared () : memref<16xf16>
    memref.copy %arg0, %in : memref<16xf16> to memref<16xf16>

    %out = gpu.alloc host_shared () : memref<16xf16>

    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%in : memref<16xf16>, %out : memref<16xf16>)

    gpu.dealloc  %in : memref<16xf16>
    return %out : memref<16xf16>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_copy(%a: memref<16xf16>, %b: memref<16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

      %mask = arith.constant dense<1> : vector<16xi1>
      %offsets = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>

      // load from a using load_gather
      %a_tdesc = xegpu.create_tdesc %a, %offsets : memref<16xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #scatter>
      %data = xegpu.load %a_tdesc, %mask : !xegpu.tensor_desc<16xf16, #scatter>, vector<16xi1> -> vector<16xf16>

      // %v1 = vector.extract %data[4]: f16 from vector<16xf16>
      // gpu.printf "\ndata[4] : %f.\n" %v1: f16

      // store to b using store_scatter
      %b_tdesc = xegpu.create_tdesc %b, %offsets : memref<16xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #scatter>
      xegpu.store %data, %b_tdesc, %mask : vector<16xf16>, !xegpu.tensor_desc<16xf16, #scatter>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1: index
    %c16 = arith.constant 16: index
    %A = memref.alloc() : memref<16xf16>

    scf.for %i = %c0 to %c16 step %c1 {
      %i32 = index.castu %i : index to i32
      %f16 = arith.sitofp %i32 : i32 to f16
      memref.store %f16, %A[%i] : memref<16xf16>
    }

    %B = call @test(%A) : (memref<16xf16>) -> memref<16xf16>
    %B_cast = memref.cast %B : memref<16xf16> to memref<*xf16>
    //CHECK: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15]
    call @printMemrefF16(%B_cast) : (memref<*xf16>) -> ()
    memref.dealloc %A : memref<16xf16>
    return
  }

  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
}
