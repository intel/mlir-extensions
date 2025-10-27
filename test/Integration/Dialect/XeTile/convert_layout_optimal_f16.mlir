// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @conv_layout attributes {gpu.container_module} {
  func.func @convert_layout(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>) -> memref<64x64xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<64x64xf16>
    gpu.memcpy  %memref, %arg0 : memref<64x64xf16>, memref<64x64xf16>
    %memref_0 = gpu.alloc  () : memref<64x64xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<64x64xf16>, memref<64x64xf16>
    %memref_1 = gpu.alloc  () : memref<64x64xf16>
    gpu.launch_func  @kernel::@test_convert_layout blocks in (%c1, %c1, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<64x64xf16>, %memref_0 : memref<64x64xf16>, %memref_1 : memref<64x64xf16>)
    gpu.dealloc  %memref : memref<64x64xf16>
    gpu.dealloc  %memref_0 : memref<64x64xf16>
    %alloc = memref.alloc() : memref<64x64xf16>
    gpu.memcpy  %alloc, %memref_1 : memref<64x64xf16>, memref<64x64xf16>
    gpu.dealloc  %memref_1 : memref<64x64xf16>
    return %alloc : memref<64x64xf16>
  }
  gpu.module @kernel  {
    gpu.func @test_convert_layout(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>, %arg2: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = xetile.init_tile %arg0[%block_id_x, %block_id_y] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [16, 2], sg_data = [4, 32]>>>
      %1 = xetile.load_tile %0 : !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [16, 2], sg_data = [4, 32]>>> -> vector<64x64xf16>
      %2 = xetile.convert_layout %1 {wg_map_result = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 16]>, wg_map_source = #xetile.wg_map<sg_layout = [16, 2], sg_data = [4, 32]>} : vector<64x64xf16>
      %3 = xetile.init_tile %arg1[%block_id_x, %block_id_y] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>>
      %4 = xetile.load_tile %3 : !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>> -> vector<64x64xf16>
      %5 = arith.addf %4, %2 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 16]>} : vector<64x64xf16>
      %6 = xetile.init_tile %arg2[%block_id_x, %block_id_y] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>>
      xetile.store_tile %5,  %6 : vector<64x64xf16>, !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    // intialize matrix A, B ; A[i, j] = 1
    %cst = arith.constant 1.000000e+00 : f16
    %cst_0 = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() : memref<64x64xf16>
    %alloc_1 = memref.alloc() : memref<64x64xf16>
    %alloc_2 = memref.alloc() : memref<64x64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c1 {
        memref.store %cst, %alloc[%arg0, %arg1] : memref<64x64xf16>
        memref.store %cst, %alloc_1[%arg0, %arg1] : memref<64x64xf16>
        memref.store %cst_0, %alloc_2[%arg0, %arg1] : memref<64x64xf32>
      }
    }
    // call @printMemrefF32(%cast_c): (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_c_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @convert_layout(%alloc, %alloc_1) : (memref<64x64xf16>, memref<64x64xf16>) -> memref<64x64xf16>
    %cast = memref.cast %0 : memref<64x64xf16> to memref<*xf16>
    %cast_3 = memref.cast %alloc_2 : memref<64x64xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_3) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<64x64xf16>
    memref.dealloc %alloc_1 : memref<64x64xf16>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

