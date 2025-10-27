// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_512xf32 : memref<512xf32> = dense<0.000000e+00>
  func.func @test(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  () : memref<512xf32>
    gpu.memcpy  %memref, %arg0 : memref<512xf32>, memref<512xf32>
    %memref_0 = gpu.alloc  () : memref<512xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<512xf32>, memref<512xf32>
    %memref_1 = gpu.alloc  () : memref<512xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref : memref<512xf32>, %memref_0 : memref<512xf32>, %memref_1 : memref<512xf32>)
    gpu.dealloc  %memref : memref<512xf32>
    gpu.dealloc  %memref_0 : memref<512xf32>
    %alloc = memref.alloc() : memref<512xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<512xf32>, memref<512xf32>
    gpu.dealloc  %memref_1 : memref<512xf32>
    return %alloc : memref<512xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %thread_id_x = gpu.thread_id  x
      %c16 = arith.constant 16 : index
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %0 = arith.muli %thread_id_x, %c16 : index
      %1 = xegpu.create_nd_tdesc %arg1[%0] : memref<512xf32> -> !xegpu.tensor_desc<16xf32>
      %2 = xegpu.load_nd %1  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %3 = xegpu.create_nd_tdesc %arg0[%0] : memref<512xf32> -> !xegpu.tensor_desc<16xf32>
      %4 = xegpu.load_nd %3  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %5 = arith.addf %4, %2 : vector<16xf32>
      %6 = xegpu.create_nd_tdesc %arg2[%0] : memref<512xf32> -> !xegpu.tensor_desc<16xf32>
      xegpu.store_nd %5, %6  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // calculate the result of C vector
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %false = arith.constant false
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() : memref<512xf32>
    %cast = memref.cast %alloc : memref<512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast, %cst, %cst_0, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    %alloc_1 = memref.alloc() : memref<512xf32>
    %cast_2 = memref.cast %alloc_1 : memref<512xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast_2, %cst, %cst_0, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    %alloc_3 = memref.alloc() : memref<512xf32>
    scf.for %arg0 = %c0 to %c512 step %c1 {
      %1 = memref.load %alloc[%arg0] : memref<512xf32>
      %2 = memref.load %alloc_1[%arg0] : memref<512xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %alloc_3[%arg0] : memref<512xf32>
    }
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_1) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %cast_4 = memref.cast %0 : memref<512xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_3 : memref<512xf32> to memref<*xf32>
    call @printMemrefF32(%cast_5) : (memref<*xf32>) -> ()
    call @printMemrefF32(%cast_4) : (memref<*xf32>) -> ()
    call @printAllcloseF32(%cast_5, %cast_4) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

