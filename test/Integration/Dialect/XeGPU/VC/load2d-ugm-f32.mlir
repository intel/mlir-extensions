// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  // memref.global "private" constant @__constant_8x16xf32 : memref<16x32xf32> = dense<1.0>
  memref.global "private" constant @__constant_8x16xf32 : memref<16x32xf32> = dense<1.000000e+00>
  func.func @test(%arg0: memref<16x32xf32>) -> memref<16x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<16x32xf32>
    gpu.memcpy  %memref, %arg0 : memref<16x32xf32>, memref<16x32xf32>
    %memref_0 = gpu.alloc  () : memref<16x32xf32>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16x32xf32>, %memref_0 : memref<16x32xf32>)
    gpu.dealloc  %memref : memref<16x32xf32>
    %alloc = memref.alloc() : memref<16x32xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<16x32xf32>, memref<16x32xf32>
    gpu.dealloc  %memref_0 : memref<16x32xf32>
    return %alloc : memref<16x32xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_copy(%arg0: memref<16x32xf32>, %arg1: memref<16x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.create_nd_tdesc %arg1[2, 2] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      xegpu.store_nd %2, %1  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf32 : memref<16x32xf32>
    //CHECK-COUNT-2: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    //CHECK-COUNT-8: [0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    //CHECK-COUNT-6: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    %1 = call @test(%0) : (memref<16x32xf32>) -> memref<16x32xf32>
    %cast = memref.cast %1 : memref<16x32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

