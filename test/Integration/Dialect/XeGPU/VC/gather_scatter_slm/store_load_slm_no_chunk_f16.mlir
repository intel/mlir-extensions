// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test() -> memref<16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<16xf16>
    gpu.launch_func  @test_kernel::@test_store_scatter blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16xf16>)
    %alloc = memref.alloc() : memref<16xf16>
    gpu.memcpy  %alloc, %memref : memref<16xf16>, memref<16xf16>
    gpu.dealloc  %memref : memref<16xf16>
    return %alloc : memref<16xf16>
  }
  gpu.module @test_kernel  {
      // store the cst into slm and load it back;
      // store data to global memory
    gpu.func @test_store_scatter(%arg0: memref<16xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : vector<16xf16>
      %cst_0 = arith.constant dense<true> : vector<16xi1>
      %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %alloc = memref.alloc() : memref<16xf16, 3>
      %0 = xegpu.create_tdesc %alloc, %cst_1 : memref<16xf16, 3>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  slm>>
      xegpu.store %cst, %0, %cst_0  : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  slm>>, vector<16xi1>
      %1 = xegpu.load %0, %cst_0  : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<memory_space =  slm>>, vector<16xi1> -> vector<16xf16>
      %2 = xegpu.create_tdesc %arg0, %cst_1 : memref<16xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<>>
      xegpu.store %1, %2, %cst_0  : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    //CHECK: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15]
    %0 = call @test() : () -> memref<16xf16>
    %cast = memref.cast %0 : memref<16xf16> to memref<*xf16>
    call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
}

