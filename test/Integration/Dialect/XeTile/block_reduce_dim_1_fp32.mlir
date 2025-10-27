// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @softmax attributes {gpu.container_module} {
  func.func @reduce_test(%arg0: memref<1024x32xf32>) -> memref<1x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<1024x32xf32>
    gpu.memcpy  %memref, %arg0 : memref<1024x32xf32>, memref<1024x32xf32>
    %memref_0 = gpu.alloc  () : memref<1x1024xf32>
    gpu.launch_func  @kernel::@reduce_dim_1 blocks in (%c64, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x32xf32>, %memref_0 : memref<1x1024xf32>)
    gpu.dealloc  %memref : memref<1024x32xf32>
    %alloc = memref.alloc() : memref<1x1024xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<1x1024xf32>, memref<1x1024xf32>
    gpu.dealloc  %memref_0 : memref<1x1024xf32>
    return %alloc : memref<1x1024xf32>
  }
        // the kernel is a 16x32 block reduction. each thread is assigned with a 16x32 block, and do reduction along dim-1 independently.
  gpu.module @kernel  {
    gpu.func @reduce_dim_1(%arg0: memref<1024x32xf32>, %arg1: memref<1x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<3.200000e+00> : vector<16xf32>
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<1024x32xf32> -> !xetile.tile<16x32xf32>
      %3 = xetile.load_tile %2 : !xetile.tile<16x32xf32> -> vector<16x32xf32>
      %4 = xetile.reduction <add>, %3 [1] : vector<16x32xf32> -> vector<16x1xf32>
      %5 = xetile.init_tile %arg1[0, %0] : memref<1x1024xf32> -> !xetile.tile<1x16xf32>
      %6 = vector.shape_cast %4 : vector<16x1xf32> to vector<1x16xf32>
      xetile.store_tile %6,  %5 : vector<1x16xf32>, !xetile.tile<1x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    // intialize matrix A ; A[i, j] = j
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+02 : f32
    %alloc = memref.alloc() : memref<1024x32xf32>
    %alloc_1 = memref.alloc() : memref<1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f32
        %3 = arith.divf %2, %cst_0 : f32
        memref.store %3, %alloc[%arg0, %arg1] : memref<1024x32xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      %1 = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%arg2 = %cst) -> (f32) {
        %2 = memref.load %alloc[%arg0, %arg1] : memref<1024x32xf32>
        %3 = arith.addf %arg2, %2 : f32
        scf.yield %3 : f32
      }
      memref.store %1, %alloc_1[%arg0] : memref<1024xf32>
    }
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @reduce_test(%alloc) : (memref<1024x32xf32>) -> memref<1x1024xf32>
    %cast = memref.cast %0 : memref<1x1024xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_1 : memref<1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_2) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x32xf32>
    memref.dealloc %alloc_1 : memref<1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

