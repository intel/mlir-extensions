// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1_3x16xf32 : memref<3x16xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_3_3x16xf32 : memref<3x16xf32> = dense<3.000000e+00>
  func.func @test(%arg0: memref<3x16xf32>, %arg1: memref<3x16xf32>) -> memref<3x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    // Spirv has no lowering for memref.subview
    %memref = gpu.alloc  () : memref<3x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<3x16xf32>, memref<3x16xf32>
    %memref_0 = gpu.alloc  () : memref<3x16xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<3x16xf32>, memref<3x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %memref to offset: [0], sizes: [48], strides: [1] : memref<3x16xf32> to memref<48xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %memref_0 to offset: [0], sizes: [48], strides: [1] : memref<3x16xf32> to memref<48xf32>
    %cast = memref.cast %reinterpret_cast : memref<48xf32> to memref<?xf32>
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<48xf32> to memref<?xf32>
    gpu.launch_func  @test_kernel::@test_scattered blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%cast : memref<?xf32>, %cast_2 : memref<?xf32>)
    gpu.dealloc  %memref : memref<3x16xf32>
    %alloc = memref.alloc() : memref<3x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<3x16xf32>, memref<3x16xf32>
    gpu.dealloc  %memref_0 : memref<3x16xf32>
    return %alloc : memref<3x16xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_scattered(%arg0: memref<?xf32>, %arg1: memref<?xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // This test emulates 2D load with user defined padding
      // We load rows with %row_mask that has 0's to not cross the boundary.
      // We pad the values that were not loaded (as per %row_mask) with %user_val.
      // We store full (padded) rows with %store_mask.
      %c0 = arith.constant 0 : index
      // Spirv has no lowering for memref.reinterpret_cast with different sizes (doesn't work: memref<3x16xf32> to memref<16xf32>)
      // Each row has a tdesc with offsets that determine linearized memref's values to be loaded
      // The entire row is out of bounds
      %cst = arith.constant dense<true> : vector<16xi1>
      %cst_0 = arith.constant dense<2.233000e+01> : vector<16xf32>
      %cst_1 = arith.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<16xi1>
      %cst_2 = arith.constant dense<16> : vector<16xindex>
      %cst_3 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %0 = xegpu.create_tdesc %arg0, %cst_3 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %1 = xegpu.create_tdesc %arg1, %cst_3 : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %2 = xegpu.load %0, %cst_1  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
      %3 = arith.select %cst_1, %2, %cst_0 : vector<16xi1>, vector<16xf32>
      xegpu.store %3, %1, %cst  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      %4 = xegpu.update_offset %0, %cst_2 : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
      %5 = xegpu.update_offset %1, %cst_2 : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
      %6 = xegpu.load %4, %cst_1  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
      %7 = arith.select %cst_1, %6, %cst_0 : vector<16xi1>, vector<16xf32>
      xegpu.store %7, %5, %cst  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      %8 = xegpu.update_offset %5, %cst_2 : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
      xegpu.store %cst_0, %8, %cst  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1_3x16xf32 : memref<3x16xf32>
    %1 = memref.get_global @__constant_3_3x16xf32 : memref<3x16xf32>
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

