// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1x16xf32 : memref<1x16xf32> = dense<1.100000e+00>
  memref.global "private" constant @__constant_3x16xf32 : memref<1x16xf32> = dense<3.000000e+00>
  func.func @test(%arg0: memref<1x16xf32>, %arg1: memref<1x16xf32>) -> memref<1x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<1x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<1x16xf32>, memref<1x16xf32>
    %memref_0 = gpu.alloc  () : memref<1x16xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<1x16xf32>, memref<1x16xf32>
    gpu.launch_func  @test_kernel::@test_scattered blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1x16xf32>, %memref_0 : memref<1x16xf32>)
    gpu.dealloc  %memref : memref<1x16xf32>
    %alloc = memref.alloc() : memref<1x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<1x16xf32>, memref<1x16xf32>
    gpu.dealloc  %memref_0 : memref<1x16xf32>
    return %alloc : memref<1x16xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_scattered(%arg0: memref<1x16xf32>, %arg1: memref<1x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %cst_0 = arith.constant dense<true> : vector<16xi1>
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16], strides: [1] : memref<1x16xf32> to memref<16xf32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16], strides: [1] : memref<1x16xf32> to memref<16xf32>
      %0 = xegpu.create_tdesc %reinterpret_cast, %cst : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %1 = xegpu.create_tdesc %reinterpret_cast_1, %cst : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %2 = xegpu.load %0, %cst_0  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
      xegpu.store %2, %1, %cst_0  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_1x16xf32 : memref<1x16xf32>
    %1 = memref.get_global @__constant_3x16xf32 : memref<1x16xf32>
    %2 = call @test(%0, %1) : (memref<1x16xf32>, memref<1x16xf32>) -> memref<1x16xf32>
    // CHECK: ( 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 )
    %3 = vector.load %2[%c0, %c0] : memref<1x16xf32>, vector<16xf32>
    vector.print %3 : vector<16xf32>
    return
  }
}
