// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// NOTES : This test load a tile from A, and then do a transpose on it,
// and store it back to B, using 16 threads in a workgroup. Each thread
// loads a 16x16 block from A, and transpose it. And then share the result
// with other threads via convert_layout. Finally each thread will store
// a 8x32 block to B.
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<64x64xf16>) -> memref<64x64xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc  () : memref<64x64xf16>
    gpu.memcpy  %memref, %arg0 : memref<64x64xf16>, memref<64x64xf16>
    %memref_0 = gpu.alloc  () : memref<64x64xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)  args(%memref : memref<64x64xf16>, %memref_0 : memref<64x64xf16>)
    gpu.dealloc  %memref : memref<64x64xf16>
    %alloc = memref.alloc() : memref<64x64xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<64x64xf16>, memref<64x64xf16>
    gpu.dealloc  %memref_0 : memref<64x64xf16>
    return %alloc : memref<64x64xf16>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %0 = xetile.init_tile %arg0[%c0, %c0] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [16, 16]>>>
      %1 = xetile.load_tile %0 : !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [16, 16]>>> -> vector<64x64xf16>
      %2 = xetile.transpose %1, [1, 0] {map = #xetile.wg_map<sg_layout = [4, 4], sg_data = [16, 16]>} : vector<64x64xf16> -> vector<64x64xf16>
      %3 = xetile.convert_layout %2 {wg_map_result = #xetile.wg_map<sg_layout = [8, 2], sg_data = [8, 32]>} : vector<64x64xf16>
      %4 = xetile.init_tile %arg1[%c0, %c0] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 2], sg_data = [8, 32]>>>
      xetile.store_tile %3,  %4 : vector<64x64xf16>, !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 2], sg_data = [8, 32]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    // intialize matrix A ; A[i, j] = j
        // %mul = arith.muli %i, %c64 : index
        // %add = arith.addi %mul, %j : index
        // %t = index.castu %add : index to i16
    %alloc = memref.alloc() : memref<64x64xf16>
    %alloc_0 = memref.alloc() : memref<64x64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<64x64xf16>
        %3 = arith.extf %2 : f16 to f32
        memref.store %3, %alloc_0[%arg1, %arg0] : memref<64x64xf32>
      }
    }
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc) : (memref<64x64xf16>) -> memref<64x64xf16>
    %cast = memref.cast %0 : memref<64x64xf16> to memref<*xf16>
    %cast_1 = memref.cast %alloc_0 : memref<64x64xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_1) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<64x64xf16>
    memref.dealloc %alloc_0 : memref<64x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
