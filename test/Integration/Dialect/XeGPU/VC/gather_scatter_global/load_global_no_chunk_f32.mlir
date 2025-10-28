// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<16xf32>) -> memref<16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<16xf32>
    gpu.memcpy  %memref, %arg0 : memref<16xf32>, memref<16xf32>
    %memref_0 = gpu.alloc  () : memref<16xf32>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16xf32>, %memref_0 : memref<16xf32>)
    gpu.dealloc  %memref : memref<16xf32>
    %alloc = memref.alloc() : memref<16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<16xf32>, memref<16xf32>
    gpu.dealloc  %memref_0 : memref<16xf32>
    return %alloc : memref<16xf32>
  }
  gpu.module @test_kernel  {
      // load from a using load_gather
      // store to b using store_scatter
    gpu.func @test_copy(%arg0: memref<16xf32>, %arg1: memref<16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<true> : vector<16xi1>
      %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
      %0 = xegpu.create_tdesc %arg0, %cst_0 : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      xegpu.prefetch %0  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      %1 = xegpu.load %0, %cst  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
      %2 = xegpu.create_tdesc %arg1, %cst_0 : memref<16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
      xegpu.store %1, %2, %cst  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %alloc = memref.alloc() : memref<16xf32>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %1 = index.castu %arg0 : index to i32
      %2 = arith.sitofp %1 : i32 to f32
      memref.store %2, %alloc[%arg0] : memref<16xf32>
    }
    // call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()
    //CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc) : (memref<16xf32>) -> memref<16xf32>
    %cast = memref.cast %alloc : memref<16xf32> to memref<*xf32>
    %cast_0 = memref.cast %0 : memref<16xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_0) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
