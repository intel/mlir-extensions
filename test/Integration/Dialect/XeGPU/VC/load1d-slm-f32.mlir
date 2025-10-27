// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_8x16xf32 : memref<32xf32> = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]>
  func.func @test(%arg0: memref<32xf32>) -> memref<32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<32xf32>
    gpu.memcpy  %memref, %arg0 : memref<32xf32>, memref<32xf32>
    %memref_0 = gpu.alloc  () : memref<32xf32>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<32xf32>, %memref_0 : memref<32xf32>)
    gpu.dealloc  %memref : memref<32xf32>
    %alloc = memref.alloc() : memref<32xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<32xf32>, memref<32xf32>
    gpu.dealloc  %memref_0 : memref<32xf32>
    return %alloc : memref<32xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_copy(%arg0: memref<32xf32>, %arg1: memref<32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c4 = arith.constant 4 : index
      %0 = xegpu.create_nd_tdesc %arg0[%c4] : memref<32xf32> -> !xegpu.tensor_desc<16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %alloc = memref.alloc() : memref<16xf32, 3>
      %2 = xegpu.create_nd_tdesc %alloc[0] : memref<16xf32, 3> -> !xegpu.tensor_desc<16xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
      xegpu.store_nd %1, %2  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<16xf32, #xegpu.block_tdesc_attr<memory_space =  slm>> -> vector<16xf32>
      %4 = xegpu.create_nd_tdesc %arg1[3] : memref<32xf32> -> !xegpu.tensor_desc<16xf32>
      xegpu.store_nd %3, %4  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf32 : memref<32xf32>
    //CHECK: [0,  0,  0,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    %1 = call @test(%0) : (memref<32xf32>) -> memref<32xf32>
    %cast = memref.cast %1 : memref<32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

