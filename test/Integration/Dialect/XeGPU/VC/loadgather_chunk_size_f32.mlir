// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1_4x16xf32 : memref<4x16xf32> = dense<1.1>
  memref.global "private" constant @__constant_3_4x16xf32 : memref<4x16xf32> = dense<3.0>

  func.func @test(%arg0: memref<4x16xf32>, %arg1: memref<4x16xf32>) -> memref<4x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<4x16xf32>
    memref.copy %arg0, %memref : memref<4x16xf32> to memref<4x16xf32>
    %memref1 = gpu.alloc  host_shared () : memref<4x16xf32>
    memref.copy %arg1, %memref1 : memref<4x16xf32> to memref<4x16xf32>

    %in = memref.reinterpret_cast %memref to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>
    %out = memref.reinterpret_cast %memref1 to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>

    %memref_dyn = memref.cast %in : memref<64xf32> to memref<?xf32>
    %memref1_dyn = memref.cast %out : memref<64xf32> to memref<?xf32>

    gpu.launch_func  @test_kernel::@test_scattered blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_dyn : memref<?xf32>, %memref1_dyn : memref<?xf32>)
    gpu.dealloc  %memref : memref<4x16xf32>
    return %memref1 : memref<4x16xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_scattered(%in: memref<?xf32>, %out: memref<?xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // We have 16 work items, each accesses 2 elements: {chunk_size = 2}, hence 16x2 tensor.
      // Valid offsets (%offsets for which %mask is 1) should not exceed 16*2=32.
      %offsets = arith.constant dense<[0,4,8,12,16,20,24,28,32,34,38,42,46,50,54,58]> : vector<16xindex>
      %mask = arith.constant dense<[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]> : vector<16xi1>
      %tdesc_in = xegpu.create_tdesc %in, %offsets : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
      %tdesc_out = xegpu.create_tdesc %out, %offsets : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
      %loaded = xegpu.load %tdesc_in, %mask : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1> -> vector<16x2xf32>
      xegpu.store %loaded, %tdesc_out, %mask : vector<16x2xf32>, !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1_4x16xf32 : memref<4x16xf32>
    %1 = memref.get_global @__constant_3_4x16xf32 : memref<4x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %2 = call @test(%0, %1) : (memref<4x16xf32>, memref<4x16xf32>) -> memref<4x16xf32>
    %cast = memref.cast %2 : memref<4x16xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = 0x{{.*}} rank = 2 offset = 0 sizes = [4, 16] strides = [16, 1] data =
    // CHECK-NEXT: [1.1,   1.1,   3,   3,   1.1,   1.1,   3,   3,   1.1,   1.1,   3,   3,   1.1,   1.1,   3,   3],
    // CHECK-NEXT: [1.1,   1.1,   3,   3,   1.1,   1.1,   3,   3,   1.1,   1.1,   3,   3,   1.1,   1.1,   3,   3],
    // CHECK-NEXT: [3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3],
    // CHECK-NEXT: [3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3]
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
