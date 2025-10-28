// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<384x64xf32>, %arg1: memref<384x64xf32>) -> memref<384x64xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<384x64xf32>
    gpu.memcpy  %memref, %arg0 : memref<384x64xf32>, memref<384x64xf32>
    %memref_0 = gpu.alloc  () : memref<384x64xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<384x64xf32>, memref<384x64xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<384x64xf32>, %memref_0 : memref<384x64xf32>)
    gpu.dealloc  %memref : memref<384x64xf32>
    %alloc = memref.alloc() : memref<384x64xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<384x64xf32>, memref<384x64xf32>
    gpu.dealloc  %memref_0 : memref<384x64xf32>
    return %alloc : memref<384x64xf32>
  }
  gpu.module @test_kernel  {
      /// canonicalize
    gpu.func @test_kernel(%arg0: memref<384x64xf32>, %arg1: memref<384x64xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = xetile.init_tile %arg0[%c0, %c0] : memref<384x64xf32> -> !xetile.tile<384x64xf32>
      %1 = xetile.init_tile %arg1[%c0, %c0] : memref<384x64xf32> -> !xetile.tile<384x64xf32>
      %2 = xetile.load_tile %0 : !xetile.tile<384x64xf32> -> vector<384x64xf32>
      xetile.store_tile %2,  %1 : vector<384x64xf32>, !xetile.tile<384x64xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // TRY 385x64
    // fill A with 2, B with 0
    // Load from A, store to B
    // call @printMemrefF32(%A_nonzero) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%B_filled) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() : memref<384x64xf32>
    %alloc_1 = memref.alloc() : memref<384x64xf32>
    %cast = memref.cast %alloc : memref<384x64xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_1 : memref<384x64xf32> to memref<*xf32>
    call @fillResource1DF32(%cast, %cst_0) : (memref<*xf32>, f32) -> ()
    call @fillResource1DF32(%cast_2, %cst) : (memref<*xf32>, f32) -> ()
    %0 = call @test(%alloc, %alloc_1) : (memref<384x64xf32>, memref<384x64xf32>) -> memref<384x64xf32>
    %cast_3 = memref.cast %0 : memref<384x64xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_3) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<384x64xf32>
    memref.dealloc %alloc_1 : memref<384x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
}
