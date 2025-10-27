// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-fallback-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @narrow_tile attributes {gpu.container_module} {
  func.func @test(%arg0: memref<64x2xf32>) -> memref<64x2xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<64x2xf32>
    gpu.memcpy  %memref, %arg0 : memref<64x2xf32>, memref<64x2xf32>
    %memref_0 = gpu.alloc  () : memref<64x2xf32>
    gpu.launch_func  @test_module::@test_scf_for blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<64x2xf32>, %memref_0 : memref<64x2xf32>)
    %alloc = memref.alloc() : memref<64x2xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<64x2xf32>, memref<64x2xf32>
    gpu.dealloc  %memref : memref<64x2xf32>
    gpu.dealloc  %memref_0 : memref<64x2xf32>
    return %alloc : memref<64x2xf32>
  }
  gpu.module @test_module {
    gpu.func @test_scf_for(%arg0: memref<64x2xf32>, %arg1: memref<64x2xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %0 = xetile.init_tile %arg0[0, 0] : memref<64x2xf32> -> !xetile.tile<16x2xf32, #xetile.tile_attr<>>
      %1 = xetile.init_tile %arg1[0, 0] : memref<64x2xf32> -> !xetile.tile<16x2xf32, #xetile.tile_attr<>>
      %2:2 = scf.for %arg2 = %c0 to %c64 step %c16 iter_args(%arg3 = %0, %arg4 = %1) -> (!xetile.tile<16x2xf32, #xetile.tile_attr<>>, !xetile.tile<16x2xf32, #xetile.tile_attr<>>) {
        %3 = xetile.load_tile %arg3 : !xetile.tile<16x2xf32, #xetile.tile_attr<>> -> vector<16x2xf32>
        xetile.store_tile %3,  %arg4 : vector<16x2xf32>, !xetile.tile<16x2xf32, #xetile.tile_attr<>>
        %4 = xetile.update_tile_offset %arg3, [%c16, %c0] : !xetile.tile<16x2xf32, #xetile.tile_attr<>>
        %5 = xetile.update_tile_offset %arg4, [%c16, %c0] : !xetile.tile<16x2xf32, #xetile.tile_attr<>>
        scf.yield %4, %5 : !xetile.tile<16x2xf32, #xetile.tile_attr<>>, !xetile.tile<16x2xf32, #xetile.tile_attr<>>
      }
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x2xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %1 = index.castu %arg0 : index to i32
      %2 = arith.uitofp %1 : i32 to f32
      memref.store %2, %alloc[%arg0, %c0] : memref<64x2xf32>
      memref.store %2, %alloc[%arg0, %c1] : memref<64x2xf32>
    }
    // CHECK: [ALLCLOSE: TRUE]
    //call @printMemrefF32(%cast_A) : (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    %0 = call @test(%alloc) : (memref<64x2xf32>) -> memref<64x2xf32>
    %cast = memref.cast %alloc : memref<64x2xf32> to memref<*xf32>
    %cast_0 = memref.cast %0 : memref<64x2xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_0, %cast) : (memref<*xf32>, memref<*xf32>) -> ()
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  //func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

