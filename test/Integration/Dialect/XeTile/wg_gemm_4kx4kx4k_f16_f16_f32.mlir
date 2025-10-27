// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc  () : memref<4096x4096xf16>
    gpu.memcpy  %memref, %arg0 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %memref_0 = gpu.alloc  () : memref<4096x4096xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %memref_1 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c4, %c4, %c1)  args(%memref : memref<4096x4096xf16>, %memref_0 : memref<4096x4096xf16>, %memref_1 : memref<4096x4096xf32>)
    gpu.dealloc  %memref : memref<4096x4096xf16>
    gpu.dealloc  %memref_0 : memref<4096x4096xf16>
    %alloc = memref.alloc() : memref<4096x4096xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc  %memref_1 : memref<4096x4096xf32>
    return %alloc : memref<4096x4096xf32>
  }
  gpu.module @test_kernel  {
        // intialize C tile and load it
        // initalize A and B tiles
        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
          // load A and B tiles
          // perform dpas and accumulate
          // update the offsets for A and B tiles
          // partial C tile result
        // store the final accumulated C tile result back to memory
    gpu.func @test_kernel(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 : index
      %1 = arith.muli %block_id_y, %c256 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<4096x4096xf32> -> !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 64]>>>
      %3 = xetile.load_tile %2 : !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 64]>>> -> vector<256x256xf32>
      %4 = xetile.init_tile %arg0[%0, %c0] : memref<4096x4096xf16> -> !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 256]>>>
      %5 = xetile.init_tile %arg1[%c0, %1] : memref<4096x4096xf16> -> !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [256, 64]>>>
      %6:3 = scf.for %arg3 = %c0 to %c4096 step %c256 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3) -> (!xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 256]>>>, !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [256, 64]>>>, vector<256x256xf32>) {
        %7 = xetile.load_tile %arg4 : !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 256]>>> -> vector<256x256xf16>
        %8 = xetile.load_tile %arg5 : !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [256, 64]>>> -> vector<256x256xf16>
        %9 = xetile.tile_mma %7, %8, %arg6 {wg_map_a = #xetile.wg_map<sg_layout = [4, 4], sg_data = [64, 256]>, wg_map_b = #xetile.wg_map<sg_layout = [4, 4], sg_data = [256, 64]>, wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [64, 64]>} : vector<256x256xf16>, vector<256x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        %10 = xetile.update_tile_offset %arg4, [%c0, %c256] : !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 256]>>>
        %11 = xetile.update_tile_offset %arg5, [%c256, %c0] : !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [256, 64]>>>
        scf.yield %10, %11, %9 : !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 256]>>>, !xetile.tile<256x256xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [256, 64]>>>, vector<256x256xf32>
      }
      xetile.store_tile %6#2,  %2 : vector<256x256xf32>, !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [64, 64]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    // intialize matrix A ; A[i, j] = j
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %alloc = memref.alloc() : memref<4096x4096xf16>
    %alloc_2 = memref.alloc() : memref<4096x4096xf16>
    %alloc_3 = memref.alloc() : memref<4096x4096xf32>
    %alloc_4 = memref.alloc() : memref<4096x4096xf32>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<4096x4096xf16>
      }
    }
    // make matrix B an identity matrix
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<4096x4096xf16>
        } else {
          memref.store %cst_0, %alloc_2[%arg0, %arg1] : memref<4096x4096xf16>
        }
      }
    }
    // intialize matrix C and C_ref ; C[i, j] = 0
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<4096x4096xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<4096x4096xf32>
      }
    }
    // compute C for reference
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<4096x4096xf32>
        %2 = scf.for %arg2 = %c0 to %c4096 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<4096x4096xf16>
          %4 = memref.load %alloc_2[%arg2, %arg1] : memref<4096x4096xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<4096x4096xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<4096x4096xf16>, memref<4096x4096xf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>
    %cast = memref.cast %0 : memref<4096x4096xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_4 : memref<4096x4096xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<4096x4096xf16>
    memref.dealloc %alloc_2 : memref<4096x4096xf16>
    memref.dealloc %alloc_3 : memref<4096x4096xf32>
    memref.dealloc %alloc_4 : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

