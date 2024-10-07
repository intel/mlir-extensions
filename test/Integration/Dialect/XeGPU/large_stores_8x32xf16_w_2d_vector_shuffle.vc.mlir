// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf16>) -> memref<8x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x32xf16>
    memref.copy %arg0, %memref : memref<8x32xf16> to memref<8x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x32xf16>, %memref_1 : memref<8x32xf16>)
    gpu.dealloc  %memref : memref<8x32xf16>
    return %memref_1 : memref<8x32xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
   gpu.func @test_kernel(%arg0: memref<8x32xf16>, %arg1: memref<8x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.create_nd_tdesc %arg0[0, 16] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %2 = xegpu.load_nd %0  {l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}
                  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %3 = xegpu.load_nd %1  {l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}
                  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %8 = vector.shuffle %2, %3 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
                                  : vector<8x16xf16>, vector<8x16xf16>
      %9 = vector.shape_cast %8 : vector<16x16xf16> to vector<256xf16>
      %11 = vector.shape_cast %9 : vector<256xf16> to vector<8x32xf16>
      %6 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %11, %6 : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x32xf16>
    %A_zeros = memref.cast %A : memref<8x32xf16> to memref<*xf16>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    call @fillResource1DRandomF16(%A_zeros, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    %B = call @test(%A) : (memref<8x32xf16>) -> memref<8x32xf16>
    %A_cast = memref.cast %A : memref<8x32xf16> to memref<*xf16>
    // call @printMemrefF16(%A_cast): (memref<*xf16>) -> ()
    // call @printMemrefF16(%B_cast): (memref<*xf16>) -> ()
    %B_copy = memref.alloc() : memref<8x32xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %v = memref.load %B[%i, %j] : memref<8x32xf16>
        %v_f32 = arith.extf %v : f16 to f32
        memref.store %v_f32, %B_copy[%i, %j] : memref<8x32xf32>
      }
    }
    %B_cast = memref.cast %B_copy : memref<8x32xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%A_cast, %B_cast) : (memref<*xf16>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<8x32xf16>
    memref.dealloc %B_copy : memref<8x32xf32>
    gpu.dealloc %B : memref<8x32xf16>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
