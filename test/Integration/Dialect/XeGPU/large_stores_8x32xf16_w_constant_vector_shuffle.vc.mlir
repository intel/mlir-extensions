// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test() -> memref<8x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref_1 = gpu.alloc  host_shared () : memref<8x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_1 : memref<8x32xf16>)
    return %memref_1 : memref<8x32xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
   gpu.func @test_kernel(%arg1: memref<8x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %t0 = arith.constant dense<1.0> : vector<192xf16>
      %t1 = arith.constant dense<2.0> : vector<64xf16>
      %t3 = vector.shape_cast %t0 : vector<192xf16> to vector<12x16xf16>
      %t4 = vector.shape_cast %t1 : vector<64xf16> to vector<4x16xf16>

      // do a vector shuffle with two constant vectors that have different number of elements.
      %8 = vector.shuffle %t3, %t4 [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 9, 13, 10, 14, 11, 15]
                                  : vector<12x16xf16>, vector<4x16xf16>
      %9 = vector.shape_cast %8: vector<16x16xf16> to vector<256xf16>
      %11 = vector.shape_cast %9 : vector<256xf16> to vector<8x32xf16>
      %6 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %11, %6 : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %B = call @test() : () -> memref<8x32xf16>
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
    call @printMemrefF32(%B_cast): (memref<*xf32>) -> ()
    // CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2]

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
