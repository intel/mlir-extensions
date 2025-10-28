// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1_4x16xf32 : memref<4x16xf32> = dense<1.100000e+00>
  memref.global "private" constant @__constant_3_4x16xf32 : memref<4x16xf32> = dense<3.000000e+00>
  func.func @test(%arg0: memref<4x16xf32>, %arg1: memref<4x16xf32>) -> memref<4x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<4x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<4x16xf32>, memref<4x16xf32>
    %memref_0 = gpu.alloc  () : memref<4x16xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<4x16xf32>, memref<4x16xf32>
    %reinterpret_cast = memref.reinterpret_cast %memref to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %memref_0 to offset: [0], sizes: [64], strides: [1] : memref<4x16xf32> to memref<64xf32>
    %cast = memref.cast %reinterpret_cast : memref<64xf32> to memref<?xf32>
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<64xf32> to memref<?xf32>
    gpu.launch_func  @test_kernel::@test_scattered blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%cast : memref<?xf32>, %cast_2 : memref<?xf32>)
    gpu.dealloc  %memref : memref<4x16xf32>
    %alloc = memref.alloc() : memref<4x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<4x16xf32>, memref<4x16xf32>
    gpu.dealloc  %memref_0 : memref<4x16xf32>
    return %alloc : memref<4x16xf32>
  }
  gpu.module @test_kernel  {
      // We have 16 work items, each accesses 2 elements: {chunk_size = 2}, hence 16x2 tensor.
      // Valid offsets (%offsets for which %mask is 1) should not exceed 16*2=32.
    gpu.func @test_scattered(%arg0: memref<?xf32>, %arg1: memref<?xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 34, 38, 42, 46, 50, 54, 58]> : vector<16xindex>
      %cst_0 = arith.constant dense<[true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<16xi1>
      %0 = xegpu.create_tdesc %arg0, %cst : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
      %1 = xegpu.create_tdesc %arg1, %cst : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
      %2 = xegpu.load %0, %cst_0  : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1> -> vector<16x2xf32>
      xegpu.store %2, %1, %cst_0  : vector<16x2xf32>, !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1_4x16xf32 : memref<4x16xf32>
    %1 = memref.get_global @__constant_3_4x16xf32 : memref<4x16xf32>
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
