// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @softmax attributes {gpu.container_module} {
  func.func @broadcast_test() -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<1024x1024xf32>
    gpu.launch_func  @kernel::@softmax_dim_0 blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf32>)
    %alloc = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy  %alloc, %memref : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc  %memref : memref<1024x1024xf32>
    return %alloc : memref<1024x1024xf32>
  }
    // the kernel is a 16x32 block broadcast. each thread is assigned with a 16x32 block, and broadcast value from vector<16x1xf32> to vector<16x32xf32> along dim-1 independently.
  gpu.module @kernel  {
    gpu.func @softmax_dim_0(%arg0: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %cst = arith.constant dense<3.000000e+00> : vector<16x1xf32>
      %2 = xetile.broadcast %cst [1] : vector<16x1xf32> -> vector<16x32xf32>
      %3 = xetile.init_tile %arg0[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<16x32xf32>
      xetile.store_tile %2,  %3 : vector<16x32xf32>, !xetile.tile<16x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 3.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // compute b for reference
    // step 1: exp
    %alloc = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // call @printMemrefF32(%cast_b) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_b_ref) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @broadcast_test() : () -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_0 = memref.cast %alloc : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_0) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

