// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1_3x16xf32 : memref<3x16xf32> = dense<1.0>
  memref.global "private" constant @__constant_3_3x16xf32 : memref<3x16xf32> = dense<3.0>

  func.func @test(%arg0: memref<3x16xf32>, %arg1: memref<3x16xf32>) -> memref<3x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<3x16xf32>
    memref.copy %arg0, %memref : memref<3x16xf32> to memref<3x16xf32>
    %memref1 = gpu.alloc  host_shared () : memref<3x16xf32>
    memref.copy %arg1, %memref1 : memref<3x16xf32> to memref<3x16xf32>

    // Spirv has no lowering for memref.subview
    %in = memref.reinterpret_cast %memref to offset: [0], sizes: [48], strides: [1] : memref<3x16xf32> to memref<48xf32>
    %out = memref.reinterpret_cast %memref1 to offset: [0], sizes: [48], strides: [1] : memref<3x16xf32> to memref<48xf32>

    %memref_dyn = memref.cast %in : memref<48xf32> to memref<?xf32>
    %memref1_dyn = memref.cast %out : memref<48xf32> to memref<?xf32>

    gpu.launch_func  @test_kernel::@test_scattered blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_dyn : memref<?xf32>, %memref1_dyn : memref<?xf32>)
    gpu.dealloc  %memref : memref<3x16xf32>
    return %memref1 : memref<3x16xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_scattered(%arg0: memref<?xf32>, %arg1: memref<?xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // This test emulates 2D load with user defined padding
      // We load rows with %row_mask that has 0's to not cross the boundary.
      // We pad the values that were not loaded (as per %row_mask) with %user_val.
      // We store full (padded) rows with %store_mask.
      %c0 = arith.constant 0 : index
      %store_mask = arith.constant dense<[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]> : vector<16xi1>
      %user_val = arith.constant dense<22.33> : vector<16xf32>
      %row_mask = arith.constant dense<[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]> : vector<16xi1>
      %offset_step = arith.constant dense<16>: vector<16xindex>

      // Spirv has no lowering for memref.reinterpret_cast with different sizes (doesn't work: memref<3x16xf32> to memref<16xf32>)
      // Each row has a tdesc with offsets that determine linearized memref's values to be loaded
      %offsets_row1 = arith.constant dense<[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]> : vector<16xindex>
      %row_1_in_td = xegpu.create_tdesc %arg0, %offsets_row1 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %row_1_out_td = xegpu.create_tdesc %arg1, %offsets_row1 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %row_1_loaded = xegpu.load %row_1_in_td, %row_mask : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
      %row_1_store = arith.select %row_mask, %row_1_loaded, %user_val : vector<16xi1>, vector<16xf32>
      xegpu.store %row_1_store, %row_1_out_td, %store_mask : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>

      %row_2_in_td = xegpu.update_offset %row_1_in_td, %offset_step : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
      %row_2_out_td = xegpu.update_offset %row_1_out_td, %offset_step : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
      %row_2_loaded = xegpu.load %row_2_in_td, %row_mask : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
      %row_2_store = arith.select %row_mask, %row_2_loaded, %user_val : vector<16xi1>, vector<16xf32>
      xegpu.store %row_2_store, %row_2_out_td, %store_mask : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>

      // The entire row is out of bounds
      %row_3_out_td = xegpu.update_offset %row_2_out_td, %offset_step : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
      xegpu.store %user_val, %row_3_out_td, %store_mask : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1_3x16xf32 : memref<3x16xf32>
    %1 = memref.get_global @__constant_3_3x16xf32 : memref<3x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %2 = call @test(%0, %1) : (memref<3x16xf32>, memref<3x16xf32>) -> memref<3x16xf32>
    %cast = memref.cast %2 : memref<3x16xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = 0x{{.*}} rank = 2 offset = 0 sizes = [3, 16] strides = [16, 1] data =
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   22.33,   22.33,   22.33],
    // CHECK-NEXT: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   22.33,   22.33,   22.33],
    // CHECK-NEXT: [22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33,   22.33]
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
