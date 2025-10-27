// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<0.000000e+00>
  func.func @test(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x16xf32>, memref<8x16xf32>
    %memref_0 = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<8x16xf32>, memref<8x16xf32>
    %memref_1 = gpu.alloc  () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c1, %c1)  args(%memref : memref<8x16xf32>, %memref_0 : memref<8x16xf32>, %memref_1 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf32>
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref_1 : memref<8x16xf32>
    return %alloc : memref<8x16xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %thread_id_x = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %0 = xegpu.create_nd_tdesc %arg1[%thread_id_x, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %2 = xegpu.create_nd_tdesc %arg0[%thread_id_x, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %4 = math.ceil %1 : vector<16xf32>
      %5 = math.floor %3 : vector<16xf32>
      %6 = arith.addf %4, %5 : vector<16xf32>
      %7 = xegpu.create_nd_tdesc %arg2[%thread_id_x, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<16xf32>
      xegpu.store_nd %6, %7  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // calculate the result C matrix
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %cst = arith.constant -5.000000e-01 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %alloc = memref.alloc() : memref<8x16xf32>
    %cast = memref.cast %alloc : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast, %cst, %cst_0, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    %alloc_1 = memref.alloc() : memref<8x16xf32>
    %cast_2 = memref.cast %alloc_1 : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast_2, %cst, %cst_0, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    %alloc_3 = memref.alloc() : memref<8x16xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %1 = memref.load %alloc[%arg0, %arg1] : memref<8x16xf32>
        %2 = memref.load %alloc_1[%arg0, %arg1] : memref<8x16xf32>
        %3 = math.ceil %1 : f32
        %4 = math.floor %2 : f32
        %5 = arith.addf %3, %4 : f32
        memref.store %5, %alloc_3[%arg0, %arg1] : memref<8x16xf32>
      }
    }
    // call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_1) : (memref<8x16xf32>, memref<8x16xf32>) -> memref<8x16xf32>
    %cast_4 = memref.cast %0 : memref<8x16xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_3 : memref<8x16xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_5, %cast_4) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

