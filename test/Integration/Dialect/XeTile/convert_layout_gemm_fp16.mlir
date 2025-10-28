// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @conv_layout attributes {gpu.container_module} {
  func.func @test_convert_layout_gemm(%arg0: memref<8x32xf16>, %arg1: memref<32x32xf16>) -> memref<8x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %memref = gpu.alloc  () : memref<8x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf16>, memref<8x32xf16>
    %memref_0 = gpu.alloc  () : memref<32x32xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<32x32xf16>, memref<32x32xf16>
    %memref_1 = gpu.alloc  () : memref<8x32xf32>
    gpu.launch_func  @kernel::@test_convert_layout_gemm blocks in (%c1, %c1, %c1) threads in (%c2, %c1, %c1)  args(%memref : memref<8x32xf16>, %memref_0 : memref<32x32xf16>, %memref_1 : memref<8x32xf32>)
    gpu.dealloc  %memref : memref<8x32xf16>
    gpu.dealloc  %memref_0 : memref<32x32xf16>
    %alloc = memref.alloc() : memref<8x32xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<8x32xf32>, memref<8x32xf32>
    gpu.dealloc  %memref_1 : memref<8x32xf32>
    return %alloc : memref<8x32xf32>
  }
  gpu.module @kernel  {
    // this test performs a simple matrix multiplication on 8x32xf16 and 32x32xf16 with a workgroup of 2 threads, which resulting a 8x32xf32 matrix.
    // Each thread will compute 8x16xf32 matrix, which 8x32xf16 * 32x16xf16. a is shared, each thread will load 8x16xf16 from memory, and using convert
    // layout to share the data.
    gpu.func @test_convert_layout_gemm(%arg0: memref<8x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<8x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c1 : index
      %1 = arith.muli %block_id_y, %c1 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<8x32xf16> -> !xetile.tile<8x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [8, 16]>>>
      %3 = xetile.load_tile %2 : !xetile.tile<8x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [8, 16]>>> -> vector<8x32xf16>
      %4 = xetile.convert_layout %3 {wg_map_result = #xetile.wg_map<sg_layout = [1, 2], sg_data = [8, 32]>, wg_map_source = #xetile.wg_map<sg_layout = [1, 2], sg_data = [8, 16]>} : vector<8x32xf16>
      %5 = xetile.init_tile %arg1[%0, %1] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 16]>>>
      %6 = xetile.load_tile %5 : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 16]>>> -> vector<32x32xf16>
      %7 = xetile.tile_mma %4, %6 {wg_map_a = #xetile.wg_map<sg_layout = [1, 2], sg_data = [8, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [1, 2], sg_data = [32, 16]>, wg_map_c = #xetile.wg_map<sg_layout = [1, 2], sg_data = [8, 16]>} : vector<8x32xf16>, vector<32x32xf16> -> vector<8x32xf32>
      %8 = xetile.init_tile %arg2[%0, %1] : memref<8x32xf32> -> !xetile.tile<8x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [8, 16]>>>
      xetile.store_tile %7,  %8 : vector<8x32xf32>, !xetile.tile<8x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [8, 16]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    // intialize matrix A;
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+02 : f16
    %alloc = memref.alloc() : memref<8x32xf16>
    %alloc_1 = memref.alloc() : memref<32x32xf16>
    %alloc_2 = memref.alloc() : memref<8x32xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = arith.muli %arg0, %c32 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = index.castu %2 : index to i16
        %4 = arith.uitofp %3 : i16 to f16
        %5 = arith.divf %4, %cst_0 : f16
        memref.store %5, %alloc[%arg0, %arg1] : memref<8x32xf16>
      }
    }
    // intialize matrix B;
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = arith.muli %arg0, %c32 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = index.castu %2 : index to i16
        %4 = arith.uitofp %3 : i16 to f16
        %5 = arith.divf %4, %cst_0 : f16
        memref.store %5, %alloc_1[%arg0, %arg1] : memref<32x32xf16>
      }
    }
    // intialize matrix c_ref;
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        memref.store %cst, %alloc_2[%arg0, %arg1] : memref<8x32xf32>
        scf.for %arg2 = %c0 to %c32 step %c1 {
          %1 = memref.load %alloc_2[%arg0, %arg1] : memref<8x32xf32>
          %2 = memref.load %alloc[%arg0, %arg2] : memref<8x32xf16>
          %3 = memref.load %alloc_1[%arg2, %arg1] : memref<32x32xf16>
          %4 = arith.extf %2 : f16 to f32
          %5 = arith.extf %3 : f16 to f32
          %6 = arith.mulf %4, %5 : f32
          %7 = arith.addf %1, %6 : f32
          memref.store %7, %alloc_2[%arg0, %arg1] : memref<8x32xf32>
        }
      }
    }
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test_convert_layout_gemm(%alloc, %alloc_1) : (memref<8x32xf16>, memref<32x32xf16>) -> memref<8x32xf32>
    %cast = memref.cast %0 : memref<8x32xf32> to memref<*xf32>
    %cast_3 = memref.cast %alloc_2 : memref<8x32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%cast_3) : (memref<*xf32>) -> ()
    call @printAllcloseF32(%cast, %cast_3) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x32xf16>
    memref.dealloc %alloc_1 : memref<32x32xf16>
    memref.dealloc %alloc_2 : memref<8x32xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
