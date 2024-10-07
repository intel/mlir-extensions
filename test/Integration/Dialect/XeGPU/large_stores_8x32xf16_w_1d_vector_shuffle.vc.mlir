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
      %4 = vector.shape_cast %2 : vector<8x16xf16> to vector<128xf16>
      %5 = vector.shape_cast %3 : vector<8x16xf16> to vector<128xf16>
      %8 = vector.shuffle %4, %5 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                  128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                  144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                                  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
                                  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                                  176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                                  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                  192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
                                  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                                  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                                  96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                                  224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
                                  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                                  240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
                                  : vector<128xf16>, vector<128xf16>
      %11 = vector.shape_cast %8 : vector<256xf16> to vector<8x32xf16>
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
