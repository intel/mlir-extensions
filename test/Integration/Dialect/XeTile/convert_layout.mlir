// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @conv_layout attributes {gpu.container_module} {
  func.func @convert_layout(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<64x64xf32>
    gpu.memcpy  %memref, %arg0 : memref<64x64xf32>, memref<64x64xf32>
    %memref_0 = gpu.alloc  () : memref<64x64xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<64x64xf32>, memref<64x64xf32>
    %memref_1 = gpu.alloc  () : memref<64x64xf32>
    gpu.launch_func  @kernel::@test_convert_layout blocks in (%c1, %c1, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<64x64xf32>, %memref_0 : memref<64x64xf32>, %memref_1 : memref<64x64xf32>)
    gpu.dealloc  %memref : memref<64x64xf32>
    gpu.dealloc  %memref_0 : memref<64x64xf32>
    %alloc = memref.alloc() : memref<64x64xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<64x64xf32>, memref<64x64xf32>
    gpu.dealloc  %memref_1 : memref<64x64xf32>
    return %alloc : memref<64x64xf32>
  }
  gpu.module @kernel  {
    gpu.func @test_convert_layout(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c1 : index
      %1 = arith.muli %block_id_y, %c1 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>>
      %3 = xetile.load_tile %2 : !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>> -> vector<64x64xf32>
      %4 = xetile.init_tile %arg1[%0, %1] : memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>>
      %5 = xetile.load_tile %4 : !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>> -> vector<64x64xf32>
      %6 = xetile.convert_layout %3 {wg_map_result = #xetile.wg_map<sg_layout = [4, 8], sg_data = [16, 8]>, wg_map_source = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 16]>} : vector<64x64xf32>
      %7 = arith.addf %5, %6 {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [16, 8]>} : vector<64x64xf32>
      %8 = xetile.init_tile %arg2[%0, %1] : memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>>
      xetile.store_tile %7,  %8 : vector<64x64xf32>, !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    // intialize matrix A, B ; A[i, j] = 1
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() : memref<64x64xf32>
    %alloc_1 = memref.alloc() : memref<64x64xf32>
    %alloc_2 = memref.alloc() : memref<64x64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c1 {
        memref.store %cst, %alloc[%arg0, %arg1] : memref<64x64xf32>
        memref.store %cst, %alloc_1[%arg0, %arg1] : memref<64x64xf32>
        memref.store %cst_0, %alloc_2[%arg0, %arg1] : memref<64x64xf32>
      }
    }
    //call @printMemrefF32(%cast_c): (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_c_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @convert_layout(%alloc, %alloc_1) : (memref<64x64xf32>, memref<64x64xf32>) -> memref<64x64xf32>
    %cast = memref.cast %0 : memref<64x64xf32> to memref<*xf32>
    %cast_3 = memref.cast %alloc_2 : memref<64x64xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_3) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<64x64xf32>
    memref.dealloc %alloc_1 : memref<64x64xf32>
    memref.dealloc %alloc_2 : memref<64x64xf32>
    memref.dealloc  %0 : memref<64x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

