// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf16>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %memref = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_0 = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_1 = gpu.alloc  () : memref<1024x1024xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    %memref_2 = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref_2, %arg3 : memref<1024x1024xf16>, memref<1024x1024xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c8, %c8, %c1) threads in (%c4, %c4, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>, %memref_2 : memref<1024x1024xf16>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    %alloc = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc  %memref_1 : memref<1024x1024xf32>
    return %alloc : memref<1024x1024xf32>
  }
  gpu.module @test_kernel  {
        // intialize C tile and load it
        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
          // load A and B tiles
          // perform dpas and accumulate
          // update the offsets for A and B tiles
          // partial C tile result
        // store the final accumulated C tile result back to memory
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c128 : index
      %1 = arith.muli %block_id_y, %c128 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<128x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 32]>>>
      %3 = xetile.load_tile %2 : !xetile.tile<128x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 32]>>> -> vector<128x128xf32>
      %4 = xetile.init_tile %arg0[%0, %c0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>
      %5 = xetile.init_tile %arg1[%1, %c0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>
      %6 = xetile.init_tile %arg3[%c0, %1] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [128, 32]>>>
      %7:4 = scf.for %arg4 = %c0 to %c1024 step %c128 iter_args(%arg5 = %4, %arg6 = %5, %arg7 = %6, %arg8 = %3) -> (!xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>, !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>, !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [128, 32]>>>, vector<128x128xf32>) {
        %8 = xetile.load_tile %arg5 : !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>> -> vector<128x128xf16>
        %9 = xetile.load_tile %arg6 : !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>> -> vector<128x128xf16>
        %10 = xetile.load_tile %arg7 : !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [128, 32]>>> -> vector<128x128xf16>
        %11 = vector.transpose %9, [1, 0] {map = #xetile.wg_map<sg_layout = [4, 4], sg_data = [128, 32]>} : vector<128x128xf16> to vector<128x128xf16>
        %12 = arith.addf %11, %10 {map = #xetile.wg_map<sg_layout = [4, 4], sg_data = [128, 32]>} : vector<128x128xf16>
        %13 = xetile.tile_mma %8, %12, %arg8 {wg_map_a = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 128]>, wg_map_b = #xetile.wg_map<sg_layout = [4, 4], sg_data = [128, 32]>, wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 32]>} : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>
        %14 = xetile.update_tile_offset %arg5, [%c0, %c128] : !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>
        %15 = xetile.update_tile_offset %arg6, [%c0, %c128] : !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>
        %16 = xetile.update_tile_offset %arg7, [%c128, %c0] : !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [128, 32]>>>
        scf.yield %14, %15, %16, %13 : !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>, !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 128]>>>, !xetile.tile<128x128xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [128, 32]>>>, vector<128x128xf32>
      }
      xetile.store_tile %7#3,  %2 : vector<128x128xf32>, !xetile.tile<128x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [32, 32]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    // intialize matrix A ; A[i, j] = j
    %cst_0 = arith.constant 5.000000e+00 : f16
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_1 = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf32>
    %alloc_3 = memref.alloc() : memref<1024x1024xf16>
    %alloc_4 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
  // Initialize matrix B with values such that B is not symmetric
      // Compute a value that ensures B[i,j] != B[j,i] when i != j
      // Store the value in B[i,j]
    // intialize matrix C and C_ref ; C[i, j] = 0
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.subi %1, %2 : i32
        %4 = arith.addi %3, %c1_i32 : i32
        %5 = arith.sitofp %4 : i32 to f16
        memref.store %5, %alloc_1[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    // Pre-op: Compute D = B + 5
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_2[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // compute C for reference
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_1[%arg0, %arg1] : memref<1024x1024xf16>
        %2 = arith.addf %1, %cst_0 : f16
        memref.store %2, %alloc_3[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    // CHECK: [ALLCLOSE: TRUE]
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<1024x1024xf16>
          %4 = memref.load %alloc_3[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %0 = call @test(%alloc, %alloc_1, %alloc_2, %alloc_3) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>, memref<1024x1024xf16>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_5 = memref.cast %alloc_4 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_5) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_1 : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf32>
    memref.dealloc %alloc_4 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

