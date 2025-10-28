// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<4096x4096xbf16>, %arg1: memref<4096x4096xbf16>, %arg2: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc  () : memref<4096x4096xbf16>
    gpu.memcpy  %memref, %arg0 : memref<4096x4096xbf16>, memref<4096x4096xbf16>
    %memref_0 = gpu.alloc  () : memref<4096x4096xbf16>
    gpu.memcpy  %memref_0, %arg1 : memref<4096x4096xbf16>, memref<4096x4096xbf16>
    %memref_1 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c8, %c4, %c1)  args(%memref : memref<4096x4096xbf16>, %memref_0 : memref<4096x4096xbf16>, %memref_1 : memref<4096x4096xf32>)
    gpu.dealloc  %memref : memref<4096x4096xbf16>
    gpu.dealloc  %memref_0 : memref<4096x4096xbf16>
    %alloc = memref.alloc() : memref<4096x4096xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.dealloc  %memref_1 : memref<4096x4096xf32>
    return %alloc : memref<4096x4096xf32>
  }
  gpu.module @test_kernel  {
        // intialize C tile and load it
        // %prefetch_c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32>
          // -> !xetile.tile<256x256xf32, #tile_attr_c>
        // initalize A and B tiles
        // prefetch first 32 slice
        // prefetch second 32 slice
        // prefetch third 32 slice
        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
          // all SGs must arrive here first
          // %every_8th_iter = arith.remui %k, %c256 : index
          // %every_8th_iter_i32 = arith.index_cast %every_8th_iter : index to i32
          // %every_8th_iter_cond = arith.cmpi eq, %every_8th_iter_i32, %c0_i32 : i32
          // scf.if %every_8th_iter_cond  {
          // }
          // load A and B tiles
          // prefetch next A and B tiles
          // update prefetch tile offsets
          // update the offsets for A and B tiles
          // perform dpas and accumulate
          //  barrier wait
          // scf.if %every_8th_iter_cond {
          // }
          // partial C tile result
        // store the final accumulated C tile result back to memory
    gpu.func @test_kernel(%arg0: memref<4096x4096xbf16>, %arg1: memref<4096x4096xbf16>, %arg2: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 : index
      %1 = arith.muli %block_id_y, %c256 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<4096x4096xf32> -> !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      %3 = xetile.load_tile %2 : !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>> -> vector<256x256xf32>
      %4 = xetile.init_tile %arg0[%0, %c0] : memref<4096x4096xbf16> -> !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
      %5 = xetile.init_tile %arg1[%c0, %1] : memref<4096x4096xbf16> -> !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      %6 = xetile.init_tile %arg0[%0, %c0] : memref<4096x4096xbf16> -> !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
      %7 = xetile.init_tile %arg1[%c0, %1] : memref<4096x4096xbf16> -> !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      xetile.prefetch_tile %6 : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
      xetile.prefetch_tile %7 : !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      %8 = xetile.init_tile %arg0[%0, %c32] : memref<4096x4096xbf16> -> !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
      %9 = xetile.init_tile %arg1[%c32, %1] : memref<4096x4096xbf16> -> !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      xetile.prefetch_tile %8 : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
      xetile.prefetch_tile %9 : !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      %10 = xetile.init_tile %arg0[%0, %c64] : memref<4096x4096xbf16> -> !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
      %11 = xetile.init_tile %arg1[%c64, %1] : memref<4096x4096xbf16> -> !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      xegpu.alloc_nbarrier 1
      %c0_i8 = arith.constant 0 : i8
      %c32_i8 = arith.constant 32 : i8
      %12 = xegpu.init_nbarrier %c0_i8, %c32_i8 : i8, i8 -> !xegpu.nbarrier
      %c0_i32 = arith.constant 0 : i32
      %13:5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3, %arg7 = %10, %arg8 = %11) -> (!xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>, !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>, vector<256x256xf32>, !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>, !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>) {
        xegpu.nbarrier_arrive %12 : !xegpu.nbarrier
        %15 = xetile.load_tile %arg4 : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>> -> vector<256x32xbf16>
        %16 = xetile.load_tile %arg5 : !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>> -> vector<32x256xbf16>
        xegpu.compile_hint
        xetile.prefetch_tile %arg7 : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
        xetile.prefetch_tile %arg8 : !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
        xegpu.compile_hint
        %17 = xetile.update_tile_offset %arg7, [%c0, %c32] : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
        %18 = xetile.update_tile_offset %arg8, [%c32, %c0] : !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
        %19 = xetile.update_tile_offset %arg4, [%c0, %c32] : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
        %20 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
        xegpu.compile_hint
        %21 = xetile.tile_mma %15, %16, %arg6 {wg_map_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>, wg_map_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x32xbf16>, vector<32x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>
        xegpu.compile_hint
        xegpu.nbarrier_wait %12 : !xegpu.nbarrier
        scf.yield %19, %20, %21, %17, %18 : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>, !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>, vector<256x256xf32>, !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>, !xetile.tile<32x256xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      }
      %14 = xetile.init_tile %arg2[%0, %1] : memref<4096x4096xf32> -> !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      xetile.store_tile %13#2,  %14 : vector<256x256xf32>, !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>>>
      xegpu.compile_hint
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    // convert the memref to 1D and fill with random values in (0.0, 1.0)
    // convert the memref to 1D and fill with random values in (0.0, 1.0)
    // intialize matrix C and C_ref ; C[i, j] = 0
    %false = arith.constant false
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() : memref<4096x4096xbf16>
    %alloc_1 = memref.alloc() : memref<4096x4096xbf16>
    %alloc_2 = memref.alloc() : memref<4096x4096xf32>
    %alloc_3 = memref.alloc() : memref<4096x4096xf32>
    %cast = memref.cast %alloc : memref<4096x4096xbf16> to memref<*xbf16>
    call @fillResource1DRandomBF16(%cast, %cst, %cst_0, %false) : (memref<*xbf16>, f32, f32, i1) -> ()
    %cast_4 = memref.cast %alloc_1 : memref<4096x4096xbf16> to memref<*xbf16>
    call @fillResource1DRandomBF16(%cast_4, %cst, %cst_0, %false) : (memref<*xbf16>, f32, f32, i1) -> ()
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        memref.store %cst, %alloc_2[%arg0, %arg1] : memref<4096x4096xf32>
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<4096x4096xf32>
      }
    }
    // Run GPU.
    // Run CPU
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_1, %alloc_2) : (memref<4096x4096xbf16>, memref<4096x4096xbf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>
    %cast_5 = memref.cast %0 : memref<4096x4096xf32> to memref<*xf32>
    %cast_6 = memref.cast %alloc : memref<4096x4096xbf16> to memref<*xbf16>
    %cast_7 = memref.cast %alloc_1 : memref<4096x4096xbf16> to memref<*xbf16>
    %cast_8 = memref.cast %alloc_3 : memref<4096x4096xf32> to memref<*xf32>
    call @gemmBF16BF16F32(%cast_6, %cast_7, %cast_8) : (memref<*xbf16>, memref<*xbf16>, memref<*xf32>) -> ()
    call @printAllcloseF32(%cast_5, %cast_8) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<4096x4096xbf16>
    memref.dealloc %alloc_1 : memref<4096x4096xbf16>
    memref.dealloc %alloc_2 : memref<4096x4096xf32>
    memref.dealloc %alloc_3 : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseBF16(memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomBF16(memref<*xbf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmBF16BF16F32(memref<*xbf16>, memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
