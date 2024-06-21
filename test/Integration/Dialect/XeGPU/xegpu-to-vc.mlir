// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp  \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp  \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module,
spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  memref.global "private" constant @__constant_8x16xf16 : memref<8x16xf16> = dense<5.000000e-01>
  memref.global "private" constant @__constant_16x16xf16 : memref<16x16xf16> = dense<1.099610e+00>
  func.func @test(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>) -> memref<8x16xf32> {
    %c1 = arith.constant 1 : index
    %memref_0 = gpu.alloc  host_shared () : memref<8x16xf16>
    memref.copy %arg0, %memref_0 : memref<8x16xf16> to memref<8x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x16xf16>
    memref.copy %arg1, %memref_1 : memref<16x16xf16> to memref<16x16xf16>
    %memref_c = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<8x16xf16>, %memref_1 : memref<16x16xf16>, %memref_c : memref<8x16xf32>)
    %result = memref.alloc() :  memref<8x16xf32>
    memref.copy %memref_c, %result: memref<8x16xf32> to memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf16>
    gpu.dealloc  %memref_1 : memref<16x16xf16>
    gpu.dealloc  %memref_c :memref<8x16xf32>

    return %result : memref<8x16xf32>
  }
  gpu.module @test_kernel {
   gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>}{

     %arg00 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf16> to memref<128xf16>
     %0 = xegpu.create_nd_tdesc %arg00[0]: memref<128xf16> -> !xegpu.tensor_desc<128xf16>
     %1 = xegpu.create_nd_tdesc %arg1[0, 0] {boundary_check = true} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
     %arg02 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf32> to memref<128xf32>
     %2 = xegpu.create_nd_tdesc %arg02[0] : memref<128xf32> -> !xegpu.tensor_desc<128xf32>
     %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<128xf16> -> vector<128xf16>
     %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
     %6 = vector.shape_cast %3: vector<128xf16> to vector<8x8x2xf16>
     %5 = xegpu.dpas %6, %4 : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
     %7 = vector.shape_cast %5: vector<8x16xf32> to vector<128xf32>
     xegpu.store_nd %7, %2 : vector<128xf32>, !xegpu.tensor_desc<128xf32>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_8x16xf16 : memref<8x16xf16>
    %1 = memref.get_global @__constant_16x16xf16 : memref<16x16xf16>
    %2 = call @test(%0, %1) : (memref<8x16xf16>, memref<16x16xf16>) -> memref<8x16xf32>
    %cast = memref.cast %2 : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

// CHECK: Unranked Memref base@{{(0x)?[-9a-f]*}}
// CHECK-SAME: rank = 2 offset = 0 sizes = [8, 16] strides = [16, 1] data =
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688]
