// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// NOTES : This test simply load a tile from A and store it to SLM, and load it back from SLM
// and store it to B, to verify the correctness of SLM support.
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<64x64xf16>) -> memref<64x64xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<64x64xf16>
    gpu.memcpy  %memref, %arg0 : memref<64x64xf16>, memref<64x64xf16>
    %memref_0 = gpu.alloc  () : memref<64x64xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<64x64xf16>, %memref_0 : memref<64x64xf16>)
    gpu.dealloc  %memref : memref<64x64xf16>
    %alloc = memref.alloc() : memref<64x64xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<64x64xf16>, memref<64x64xf16>
    gpu.dealloc  %memref_0 : memref<64x64xf16>
    return %alloc : memref<64x64xf16>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %0 = arith.muli %thread_id_x, %c8 : index
      %1 = arith.muli %thread_id_y, %c16 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
      %3 = xetile.load_tile %2 : !xetile.tile<8x16xf16> -> vector<8x16xf16>
      %alloc = memref.alloc() : memref<8192xi8, 3>
      %view = memref.view %alloc[%c0][] : memref<8192xi8, 3> to memref<64x64xf16, 3>
      %4 = xetile.init_tile %view[%0, %1] : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>>
      xetile.store_tile %3,  %4 : vector<8x16xf16>, !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>>
      %5 = xetile.init_tile %view[%0, %1] : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>>
      %6 = xetile.load_tile %5 : !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>> -> vector<8x16xf16>
      %7 = xetile.init_tile %arg1[%0, %1] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
      xetile.store_tile %6,  %7 : vector<8x16xf16>, !xetile.tile<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    // intialize matrix A ; A[i, j] = j
    %alloc = memref.alloc() : memref<64x64xf16>
    %alloc_0 = memref.alloc() : memref<64x64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<64x64xf16>
        %3 = arith.uitofp %1 : i16 to f32
        memref.store %3, %alloc_0[%arg0, %arg1] : memref<64x64xf32>
      }
    }
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc) : (memref<64x64xf16>) -> memref<64x64xf16>
    %cast = memref.cast %0 : memref<64x64xf16> to memref<*xf16>
    %cast_1 = memref.cast %alloc_0 : memref<64x64xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_1) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<64x64xf16>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
