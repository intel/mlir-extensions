// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_16x32xf16 : memref<16x32xf16> = dense<5.000000e-01>
  func.func @test(%arg0: memref<16x32xf16>) -> memref<16x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<16x32xf16>
    memref.copy %arg0, %memref : memref<16x32xf16> to memref<16x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<16x32xf16>, %memref_1 : memref<16x32xf32>)
    gpu.dealloc  %memref : memref<16x32xf16>
    return %memref_1 : memref<16x32xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
   gpu.func @test_kernel(%arg0: memref<16x32xf16>, %arg1: memref<16x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
      %1 = xegpu.load_nd %0 {l1_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16xf16>
      %3 = arith.extf %1: vector<2x16x16xf16> to vector<2x16x16xf32>
      %4 = vector.extract %3[0]: vector<16x16xf32> from vector<2x16x16xf32>
      %5 = vector.extract %3[1]: vector<16x16xf32> from vector<2x16x16xf32>
      %6 = vector.shape_cast %4: vector<16x16xf32> to vector<2x8x16xf32>
      %7 = vector.shape_cast %5: vector<16x16xf32> to vector<2x8x16xf32>

      %8 = vector.extract %6[0]: vector<8x16xf32> from vector<2x8x16xf32>
      %9 = vector.extract %6[1]: vector<8x16xf32> from vector<2x8x16xf32>

      %10 = vector.extract %7[0]: vector<8x16xf32> from vector<2x8x16xf32>
      %11 = vector.extract %7[1]: vector<8x16xf32> from vector<2x8x16xf32>

      %12 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %13 = xegpu.create_nd_tdesc %arg1[0, 16] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %14 = xegpu.create_nd_tdesc %arg1[8, 0] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %15 = xegpu.create_nd_tdesc %arg1[8, 16] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>

      xegpu.store_nd %8, %12 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %10, %13 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %9, %14 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %11, %15 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

      // %16 = vector.extract %4[0, 0]: f32 from vector<16x16xf32>
      // %17 = vector.extract %5[0, 0]: f32 from vector<16x16xf32>
      // gpu.printf "\narray 0: %f, array 1: %f.\n" %16, %17: f32, f32
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<16x32xf16>
    %A_random = memref.cast %A : memref<16x32xf16> to memref<*xf16>
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    %B = call @test(%A) : (memref<16x32xf16>) -> memref<16x32xf32>
    %A_cast = memref.cast %A : memref<16x32xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<16x32xf32> to memref<*xf32>
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%A_cast, %B_cast) : (memref<*xf16>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
