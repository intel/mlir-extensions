// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1x16xf32 : memref<1x16xf32> = dense<1.0>
  memref.global "private" constant @__constant_3x16xf32 : memref<1x16xf32> = dense<3.3>

  func.func @test(%arg0: memref<1x16xf32>, %arg1: memref<1x16xf32>) -> memref<1x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<1x16xf32>
    memref.copy %arg0, %memref : memref<1x16xf32> to memref<1x16xf32>
    %memref1 = gpu.alloc  host_shared () : memref<1x16xf32>
    memref.copy %arg1, %memref1 : memref<1x16xf32> to memref<1x16xf32>
    gpu.launch_func  @test_kernel::@test_scattered blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<1x16xf32>, %memref1 : memref<1x16xf32>)
    gpu.dealloc  %memref : memref<1x16xf32>
    return %memref1 : memref<1x16xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_scattered(%arg0: memref<1x16xf32>, %arg1: memref<1x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %offsets = arith.constant dense<[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]> : vector<16xindex>
      %mask = arith.constant dense<[1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]> : vector<16xi1>
      %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16], strides: [1] : memref<1x16xf32> to memref<16xf32>
      %2 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16], strides: [1] : memref<1x16xf32> to memref<16xf32>
      %tdesc1 = xegpu.create_tdesc %1, %offsets : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %tdesc2 = xegpu.create_tdesc %2, %offsets : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %loaded = xegpu.load %tdesc1, %mask : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
      xegpu.store %loaded, %tdesc2, %mask : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1x16xf32 : memref<1x16xf32>
    %1 = memref.get_global @__constant_3x16xf32 : memref<1x16xf32>
    %c0 = arith.constant 0 : index
    %2 = call @test(%0, %1) : (memref<1x16xf32>, memref<1x16xf32>) -> memref<1x16xf32>
    %vector_0 = vector.load %2[%c0,%c0] :memref<1x16xf32>, vector<16xf32>
    // CHECK: ( 1, 1, 1, 3.3, 1, 1, 1, 1, 3.3, 1, 1, 1, 1, 3.3, 1, 1 )
    vector.print %vector_0 : vector<16xf32>
    return
  }
}
