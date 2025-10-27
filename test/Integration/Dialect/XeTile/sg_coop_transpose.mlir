// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<32x32xf16>) -> memref<32x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %memref = gpu.alloc  () : memref<32x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<32x32xf16>, memref<32x32xf16>
    %memref_0 = gpu.alloc  () : memref<32x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c2, %c1)  args(%memref : memref<32x32xf16>, %memref_0 : memref<32x32xf16>)
    gpu.dealloc  %memref : memref<32x32xf16>
    %alloc = memref.alloc() : memref<32x32xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<32x32xf16>, memref<32x32xf16>
    gpu.dealloc  %memref_0 : memref<32x32xf16>
    return %alloc : memref<32x32xf16>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %0 = arith.muli %thread_id_x, %c8 : index
      %1 = arith.muli %thread_id_y, %c16 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<32x32xf16> -> !xetile.tile<8x16xf16>
      %3 = xetile.load_tile %2 : !xetile.tile<8x16xf16> -> vector<8x16xf16>
      %alloc = memref.alloc() : memref<2048xi8, 3>
      %view = memref.view %alloc[%c0][] : memref<2048xi8, 3> to memref<32x32xf16, 3>
      %transpose = memref.transpose %view (d0, d1) -> (d1, d0) : memref<32x32xf16, 3> to memref<32x32xf16, strided<[1, 32]>, 3>
      %4 = xetile.init_tile %transpose[%0, %1] : memref<32x32xf16, strided<[1, 32]>, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>
      xetile.store_tile %3,  %4 : vector<8x16xf16>, !xetile.tile<8x16xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>
      gpu.barrier
      %5 = xetile.init_tile %view[%0, %1] : memref<32x32xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>>
      %6 = xetile.load_tile %5 : !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>> -> vector<8x16xf16>
      %7 = xetile.init_tile %arg1[%0, %1] : memref<32x32xf16> -> !xetile.tile<8x16xf16>
      xetile.store_tile %6,  %7 : vector<8x16xf16>, !xetile.tile<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    // intialize matrix A ; A[i, j] = j
    %alloc = memref.alloc() : memref<32x32xf16>
    %alloc_0 = memref.alloc() : memref<32x32xf32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = arith.muli %arg0, %c32 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = index.castu %2 : index to i16
        %4 = arith.uitofp %3 : i16 to f16
        memref.store %4, %alloc[%arg0, %arg1] : memref<32x32xf16>
        %5 = index.castu %2 : index to i32
        %6 = arith.uitofp %5 : i32 to f32
        memref.store %6, %alloc_0[%arg1, %arg0] : memref<32x32xf32>
      }
    }
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc) : (memref<32x32xf16>) -> memref<32x32xf16>
    %cast = memref.cast %0 : memref<32x32xf16> to memref<*xf16>
    %cast_1 = memref.cast %alloc_0 : memref<32x32xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_1) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<32x32xf16>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

