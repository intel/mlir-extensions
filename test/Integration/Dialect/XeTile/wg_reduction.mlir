// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @reduction attributes {gpu.container_module} {
  func.func @reduce_test(%arg0: memref<256x1024xf32>) -> memref<1x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<256x1024xf32>
    gpu.memcpy  %memref, %arg0 : memref<256x1024xf32>, memref<256x1024xf32>
    %memref_0 = gpu.alloc  () : memref<1x1024xf32>
    gpu.launch_func  @kernel::@test_reduction blocks in (%c1, %c8, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<256x1024xf32>, %memref_0 : memref<1x1024xf32>)
    gpu.dealloc  %memref : memref<256x1024xf32>
    %alloc = memref.alloc() : memref<1x1024xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<1x1024xf32>, memref<1x1024xf32>
    gpu.dealloc  %memref_0 : memref<1x1024xf32>
    return %alloc : memref<1x1024xf32>
  }
  gpu.module @kernel  {
    gpu.func @test_reduction(%arg0: memref<256x1024xf32>, %arg1: memref<1x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 : index
      %1 = arith.muli %block_id_y, %c128 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<256x1024xf32> -> !xetile.tile<256x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
      %3 = xetile.load_tile %2 : !xetile.tile<256x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>> -> vector<256x128xf32>
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} dense<0.000000e+00> : vector<8x128xf32>
      %4 = vector.shape_cast %3 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32> to vector<8x32x128xf32>
      %5 = vector.multi_reduction <add>, %4, %cst {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} [1] : vector<8x32x128xf32> to vector<8x128xf32>
      %6 = xetile.convert_layout %5 {wg_map_result = #xetile.wg_map<sg_layout = [1, 32], sg_data = [8, 4]>} : vector<8x128xf32>
      %cst_0 = arith.constant {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} dense<0.000000e+00> : vector<128xf32>
      %7 = vector.multi_reduction <add>, %6, %cst_0 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} [0] : vector<8x128xf32> to vector<128xf32>
      %8 = vector.shape_cast %7 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} : vector<128xf32> to vector<1x128xf32>
      %9 = xetile.init_tile %arg1[%c0, %1] : memref<1x1024xf32> -> !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>>>
      xetile.store_tile %8,  %9 : vector<1x128xf32>, !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    // intialize matrix A ; A[i, j] = 1
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() : memref<256x1024xf32>
    %alloc_1 = memref.alloc() : memref<1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst_0, %alloc[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      %1 = scf.for %arg1 = %c0 to %c256 step %c1 iter_args(%arg2 = %cst) -> (f32) {
        %2 = memref.load %alloc[%arg1, %arg0] : memref<256x1024xf32>
        %3 = arith.addf %arg2, %2 : f32
        scf.yield %3 : f32
      }
      memref.store %1, %alloc_1[%arg0] : memref<1024xf32>
    }
    //call @printMemrefF32(%cast_b): (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_b_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @reduce_test(%alloc) : (memref<256x1024xf32>) -> memref<1x1024xf32>
    %cast = memref.cast %0 : memref<1x1024xf32> to memref<*xf32>
    %cast_2 = memref.cast %alloc_1 : memref<1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_2) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<256x1024xf32>
    memref.dealloc %alloc_1 : memref<1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
