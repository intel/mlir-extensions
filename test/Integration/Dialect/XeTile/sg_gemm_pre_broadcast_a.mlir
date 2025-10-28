// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>, %arg3: memref<1x1024xf16>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_0 = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_1 = gpu.alloc  () : memref<1024x1024xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    %memref_2 = gpu.alloc  () : memref<1x1024xf16>
    gpu.memcpy  %memref_2, %arg3 : memref<1x1024xf16>, memref<1x1024xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>, %memref_1 : memref<1024x1024xf32>, %memref_2 : memref<1x1024xf16>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    %alloc = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc  %memref_1 : memref<1024x1024xf32>
    return %alloc : memref<1024x1024xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>, %arg3: memref<1x1024xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      // intialize C tile and load it
      // initalize A and B tiles
      // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
        // load A and B tiles
        // broadcast and add to a
        // perform dpas and accumulate
        // update the offsets for A, B and bcast
        // partial C tile result
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<16x32xf32>
      %3 = xetile.load_tile %2 : !xetile.tile<16x32xf32> -> vector<16x32xf32>
      %4 = xetile.init_tile %arg0[%0, %c0] : memref<1024x1024xf16> -> !xetile.tile<16x32xf16>
      %5 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
      %6 = xetile.init_tile %arg3[%c0, %c0] : memref<1x1024xf16> -> !xetile.tile<1x32xf16>
      %7:4 = scf.for %arg4 = %c0 to %c1024 step %c32 iter_args(%arg5 = %4, %arg6 = %5, %arg7 = %6, %arg8 = %3) -> (!xetile.tile<16x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<1x32xf16>, vector<16x32xf32>) {
        %8 = xetile.load_tile %arg5 : !xetile.tile<16x32xf16> -> vector<16x32xf16>
        %9 = xetile.load_tile %arg6 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %10 = xetile.load_tile %arg7 : !xetile.tile<1x32xf16> -> vector<1x32xf16>
        %11 = xetile.broadcast %10 [0] : vector<1x32xf16> -> vector<16x32xf16>
        %12 = arith.addf %8, %11 : vector<16x32xf16>
        %13 = xetile.tile_mma %12, %9, %arg8 : vector<16x32xf16>, vector<32x32xf16>, vector<16x32xf32> -> vector<16x32xf32>
        %14 = xetile.update_tile_offset %arg5, [%c0, %c32] : !xetile.tile<16x32xf16>
        %15 = xetile.update_tile_offset %arg6, [%c32, %c0] : !xetile.tile<32x32xf16>
        %16 = xetile.update_tile_offset %arg7, [%c0, %c32] : !xetile.tile<1x32xf16>
        scf.yield %14, %15, %16, %13 : !xetile.tile<16x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<1x32xf16>, vector<16x32xf32>
      }
      // store the final accumulated C tile result back to memory
      xetile.store_tile %7#3,  %2 : vector<16x32xf32>, !xetile.tile<16x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %cst_0 = arith.constant 3.000000e+00 : f32
    %cst_1 = arith.constant -3.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // random init
    // intialize matrix C and C_ref ; C[i, j] = 0
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_2 = memref.alloc() : memref<1024x1024xf16>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    %alloc_4 = memref.alloc() : memref<1x1024xf16>
    %alloc_5 = memref.alloc() : memref<1024x1024xf32>
    %cast = memref.cast %alloc : memref<1024x1024xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast, %cst_1, %cst_0, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    %cast_6 = memref.cast %alloc_2 : memref<1024x1024xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast_6, %cst_1, %cst_0, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    %cast_7 = memref.cast %alloc_4 : memref<1x1024xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%cast_7, %cst_1, %cst_0, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_5[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // compute C for reference
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_5[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<1024x1024xf16>
          %4 = memref.load %alloc_2[%arg2, %arg1] : memref<1024x1024xf16>
          %5 = memref.load %alloc_4[%c0, %arg2] : memref<1x1024xf16>
          %6 = arith.addf %3, %5 : f16
          %7 = arith.mulf %6, %4 : f16
          %8 = arith.extf %7 : f16 to f32
          %9 = arith.addf %8, %arg3 : f32
          scf.yield %9 : f32
        }
        memref.store %2, %alloc_5[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // %cast = memref.cast %B : memref<1024x1024xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // Debugging prints (Do not remove)
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // %C_row_0 = memref.subview %2[0, 0][1, 1024][1, 1] : memref<1024x1024xf32> to memref<1x1024xf32>
    // %C_row_0_cast = memref.cast %C_row_0 : memref<1x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()
    // %C_ref_row_0 = memref.subview %C_ref[0, 0][1, 1024][1, 1] : memref<1024x1024xf32> to memref<1x1024xf32>
    // %C_ref_row_0_cast = memref.cast %C_ref_row_0 : memref<1x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%C_ref_row_0_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_2, %alloc_3, %alloc_4) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>, memref<1x1024xf16>) -> memref<1024x1024xf32>
    %cast_8 = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_9 = memref.cast %alloc_5 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_8, %cast_9) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_2 : memref<1024x1024xf16>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    memref.dealloc %alloc_5 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
}
