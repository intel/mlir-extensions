// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) -> memref<256x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<256x256xf16>
    gpu.memcpy  %memref, %arg0 : memref<256x256xf16>, memref<256x256xf16>
    %memref_0 = gpu.alloc  () : memref<256x256xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<256x256xf16>, memref<256x256xf16>
    %memref_1 = gpu.alloc  () : memref<256x256xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<256x256xf32>, memref<256x256xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<256x256xf16>, %memref_0 : memref<256x256xf16>, %memref_1 : memref<256x256xf32>)
    gpu.dealloc  %memref : memref<256x256xf16>
    gpu.dealloc  %memref_0 : memref<256x256xf16>
    %alloc = memref.alloc() : memref<256x256xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc  %memref_1 : memref<256x256xf32>
    return %alloc : memref<256x256xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<256x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<256x256xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      // intialize C tile and load it
      // initalize A and B tiles
      // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
        // load A and B tiles
        // %b_trans = vector.transpose %b_value, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
        // perform dpas and accumulate
        // update the offsets for A and B tiles
        // partial C tile result
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<256x256xf32> -> !xetile.tile<16x32xf32>
      %3 = xetile.load_tile %2 : !xetile.tile<16x32xf32> -> vector<16x32xf32>
      %4 = xetile.init_tile %arg0[%0, %c0] : memref<256x256xf16> -> !xetile.tile<16x32xf16>
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256, 256], strides: [1, 256] : memref<256x256xf16> to memref<256x256xf16, strided<[1, 256]>>
      %5 = xetile.init_tile %reinterpret_cast[%c0, %1] : memref<256x256xf16, strided<[1, 256]>> -> !xetile.tile<32x32xf16, #xetile.tile_attr<order = [0, 1]>>
      %6:3 = scf.for %arg3 = %c0 to %c256 step %c32 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3) -> (!xetile.tile<16x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<order = [0, 1]>>, vector<16x32xf32>) {
        %7 = xetile.load_tile %arg4 : !xetile.tile<16x32xf16> -> vector<16x32xf16>
        %8 = xetile.load_tile %arg5 : !xetile.tile<32x32xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<32x32xf16>
        %9 = xetile.tile_mma %7, %8, %arg6 : vector<16x32xf16>, vector<32x32xf16>, vector<16x32xf32> -> vector<16x32xf32>
        %10 = xetile.update_tile_offset %arg4, [%c0, %c32] : !xetile.tile<16x32xf16>
        %11 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<32x32xf16, #xetile.tile_attr<order = [0, 1]>>
        scf.yield %10, %11, %9 : !xetile.tile<16x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<order = [0, 1]>>, vector<16x32xf32>
      }
      // store the final accumulated C tile result back to memory
      xetile.store_tile %6#2,  %2 : vector<16x32xf32>, !xetile.tile<16x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %true = arith.constant true
    %cst = arith.constant 3.000000e+00 : f32
    %cst_0 = arith.constant -3.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    // fill A, B with random values
    // fill C, C_ref with zeros
    // compute C for reference
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<256x256xf16>
    %alloc_2 = memref.alloc() : memref<256x256xf16>
    %alloc_3 = memref.alloc() : memref<256x256xf32>
    %alloc_4 = memref.alloc() : memref<256x256xf32>
    %cast = memref.cast %alloc : memref<256x256xf16> to memref<*xf16>
    %cast_5 = memref.cast %alloc_2 : memref<256x256xf16> to memref<*xf16>
    %cast_6 = memref.cast %alloc_3 : memref<256x256xf32> to memref<*xf32>
    %cast_7 = memref.cast %alloc_4 : memref<256x256xf32> to memref<*xf32>
    call @fillResource1DRandomF16(%cast, %cst_0, %cst, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%cast_5, %cst_0, %cst, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DF32(%cast_6, %cst_1) : (memref<*xf32>, f32) -> ()
    call @fillResource1DF32(%cast_7, %cst_1) : (memref<*xf32>, f32) -> ()
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<256x256xf32>
        %2 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<256x256xf16>
          %4 = memref.load %alloc_2[%arg1, %arg2] : memref<256x256xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<256x256xf32>
      }
    }
    // %cast = memref.cast %B : memref<256x256xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // %C_row_0 = memref.subview %2[1, 0][1, 256][1, 1] : memref<256x256xf32> to memref<1x256xf32, strided<[256, 1], offset: 256>>
    // %C_row_0_cast = memref.cast %C_row_0 : memref<1x256xf32, strided<[256, 1], offset: 256>>to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf32>) -> memref<256x256xf32>
    %cast_8 = memref.cast %0 : memref<256x256xf32> to memref<*xf32>
    %cast_9 = memref.cast %alloc_4 : memref<256x256xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_8, %cast_9) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<256x256xf16>
    memref.dealloc %alloc_2 : memref<256x256xf16>
    memref.dealloc %alloc_3 : memref<256x256xf32>
    memref.dealloc %alloc_4 : memref<256x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
}

