// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<16x8xf32>) -> memref<16x8xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<16x8xf32>
    gpu.memcpy  %memref, %arg0 : memref<16x8xf32>, memref<16x8xf32>
    %memref_0 = gpu.alloc  () : memref<16x8xf32>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16x8xf32>, %memref_0 : memref<16x8xf32>)
    gpu.dealloc  %memref : memref<16x8xf32>
    %alloc = memref.alloc() : memref<16x8xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<16x8xf32>, memref<16x8xf32>
    gpu.dealloc  %memref_0 : memref<16x8xf32>
    return %alloc : memref<16x8xf32>
  }
  gpu.module @test_kernel  {
      // load from a using load_gather
      // store to b using store_nd, used to check the implicit order issues with load_gather and store_scatter.
      // %c0 = arith.constant 0 : index
      // %b_tdesc = xegpu.create_nd_tdesc %b[%c0, %c0] : memref<16x8xf32> -> !xegpu.tensor_desc<16x8xf32>
      // xegpu.store_nd %data, %b_tdesc : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32>
      // store to b using store_scatter
    gpu.func @test_copy(%arg0: memref<16x8xf32>, %arg1: memref<16x8xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<true> : vector<16xi1>
      %cst_0 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<16x8xf32> to memref<128xf32>
      %0 = xegpu.create_tdesc %reinterpret_cast, %cst_0 : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
      xegpu.prefetch %0  : !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
      %1 = xegpu.load %0, %cst  : !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>, vector<16xi1> -> vector<16x8xf32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1] : memref<16x8xf32> to memref<128xf32>
      %2 = xegpu.create_tdesc %reinterpret_cast_1, %cst_0 : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
      xegpu.store %1, %2, %cst  : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>, vector<16xi1>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %alloc = memref.alloc() : memref<16x8xf32>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %1 = arith.muli %arg0, %c8 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = index.castu %2 : index to i32
        %4 = arith.sitofp %3 : i32 to f32
        memref.store %4, %alloc[%arg0, %arg1] : memref<16x8xf32>
      }
    }
    // call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()
    //CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc) : (memref<16x8xf32>) -> memref<16x8xf32>
    %cast = memref.cast %alloc : memref<16x8xf32> to memref<*xf32>
    %cast_0 = memref.cast %0 : memref<16x8xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_0) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<16x8xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

