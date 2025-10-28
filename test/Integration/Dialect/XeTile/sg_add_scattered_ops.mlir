// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) -> memref<1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<1024xf32>
    gpu.memcpy  %memref, %arg0 : memref<1024xf32>, memref<1024xf32>
    %memref_0 = gpu.alloc  () : memref<1024xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<1024xf32>, memref<1024xf32>
    %memref_1 = gpu.alloc  () : memref<1024xf32>
    gpu.launch_func  @test_kernel::@add_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024xf32>, %memref_0 : memref<1024xf32>, %memref_1 : memref<1024xf32>)
    gpu.dealloc  %memref : memref<1024xf32>
    gpu.dealloc  %memref_0 : memref<1024xf32>
    %alloc = memref.alloc() : memref<1024xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024xf32>, memref<1024xf32>
    gpu.dealloc  %memref_1 : memref<1024xf32>
    return %alloc : memref<1024xf32>
  }
  gpu.module @test_kernel {
    gpu.func @add_kernel(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      // %c_init_tile = xetile.init_tile %C[0, 0] : memref<1024xf32> -> !xetile.tile<1x32xf32>
        // load A and B tiles
        // xetile.store_tile %c_value, %c_tile : vector<1x32xf32>, !xetile.tile<1x32xf32>
        // %c_next_tile = xetile.update_tile_offset %c_tile, [%c0, %c32] : !xetile.tile<1x32xf32>
      %cst = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : vector<1x32xindex>
      %cst_0 = arith.constant dense<32> : vector<1x32xindex>
      %cst_1 = arith.constant dense<true> : vector<1x32xi1>
      %0 = xetile.init_tile %arg0, %cst : memref<1024xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
      %1 = xetile.init_tile %arg1, %cst : memref<1024xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
      %2 = xetile.init_tile %arg2, %cst : memref<1024xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
      %3:3 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %0, %arg5 = %1, %arg6 = %2) -> (!xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>) {
        %4 = xetile.load %arg4, %cst_1 : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
        %5 = xetile.load %arg5, %cst_1 : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
        %6 = arith.addf %4, %5 : vector<1x32xf32>
        xetile.store %6, %arg6, %cst_1 : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
        %7 = xetile.update_tile_offset %arg4, %cst_0 : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
        %8 = xetile.update_tile_offset %arg5, %cst_0 : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
        %9 = xetile.update_tile_offset %arg6, %cst_0 : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
        scf.yield %7, %8, %9 : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
      }
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // intialize matrix A ;
    %alloc = memref.alloc() : memref<1024xf32>
    %alloc_0 = memref.alloc() : memref<1024xf32>
    %alloc_1 = memref.alloc() : memref<1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      %1 = index.castu %arg0 : index to i32
      %2 = arith.uitofp %1 : i32 to f32
      memref.store %2, %alloc[%arg0] : memref<1024xf32>
      memref.store %2, %alloc_0[%arg0] : memref<1024xf32>
    }
    // compute C for reference
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      %1 = memref.load %alloc[%arg0] : memref<1024xf32>
      %2 = memref.load %alloc_0[%arg0] : memref<1024xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %alloc_1[%arg0] : memref<1024xf32>
    }
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_0) : (memref<1024xf32>, memref<1024xf32>) -> memref<1024xf32>
    %cast = memref.cast %0 : memref<1024xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_1 : memref<1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_2) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024xf32>
    memref.dealloc %alloc_0 : memref<1024xf32>
    memref.dealloc %alloc_1 : memref<1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
