// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @softmax attributes {gpu.container_module} {
  func.func @reduce_test(%arg0: memref<16x512xf32>) -> memref<512xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %memref = gpu.alloc  () : memref<16x512xf32>
    gpu.memcpy  %memref, %arg0 : memref<16x512xf32>, memref<16x512xf32>
    %memref_0 = gpu.alloc  () : memref<512xf32>
    gpu.launch_func  @kernel::@reduce_dim_1 blocks in (%c1, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16x512xf32>, %memref_0 : memref<512xf32>)
    gpu.dealloc  %memref : memref<16x512xf32>
    %alloc = memref.alloc() : memref<512xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<512xf32>, memref<512xf32>
    gpu.dealloc  %memref_0 : memref<512xf32>
    return %alloc : memref<512xf32>
  }
    // the kernel is a 16x32 block reduction. each thread is assigned with a 16x32 block, and do reduction along dim-0 independently.
  gpu.module @kernel  {
    gpu.func @reduce_dim_1(%arg0: memref<16x512xf32>, %arg1: memref<512xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c16 : index
      %2 = xegpu.create_nd_tdesc %arg0[%0, %1] : memref<16x512xf32> -> !xegpu.tensor_desc<16x16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<16x16xf32> -> vector<16x16xf32>
      %4 = vector.multi_reduction <add>, %3, %cst [0] : vector<16x16xf32> to vector<16xf32>
      %5 = xegpu.create_nd_tdesc %arg1[%1] : memref<512xf32> -> !xegpu.tensor_desc<16xf32>
      xegpu.store_nd %4, %5  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    // intialize matrix A ; A[i, j] = j
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+02 : f32
    %alloc = memref.alloc() : memref<16x512xf32>
    %alloc_1 = memref.alloc() : memref<512xf32>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c512 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f32
        %3 = arith.divf %2, %cst_0 : f32
        memref.store %3, %alloc[%arg0, %arg1] : memref<16x512xf32>
      }
    }
    scf.for %arg0 = %c0 to %c512 step %c1 {
      %1 = scf.for %arg1 = %c0 to %c16 step %c1 iter_args(%arg2 = %cst) -> (f32) {
        %2 = memref.load %alloc[%arg1, %arg0] : memref<16x512xf32>
        %3 = arith.addf %arg2, %2 : f32
        scf.yield %3 : f32
      }
      memref.store %1, %alloc_1[%arg0] : memref<512xf32>
    }
    // call @printMemrefF32(%cast_b): (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_b_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @reduce_test(%alloc) : (memref<16x512xf32>) -> memref<512xf32>
    %cast = memref.cast %0 : memref<512xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_1 : memref<512xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_2) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<16x512xf32>
    memref.dealloc %alloc_1 : memref<512xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
