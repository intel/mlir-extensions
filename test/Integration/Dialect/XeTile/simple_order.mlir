// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<32x64xf16>, %arg1: memref<64x64xf16>, %arg2: memref<32x64xf32>) -> memref<32x64xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<32x64xf16>
    gpu.memcpy  %memref, %arg0 : memref<32x64xf16>, memref<32x64xf16>
    %memref_0 = gpu.alloc  () : memref<64x64xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<64x64xf16>, memref<64x64xf16>
    %memref_1 = gpu.alloc  () : memref<32x64xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<32x64xf32>, memref<32x64xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<32x64xf16>, %memref_0 : memref<64x64xf16>, %memref_1 : memref<32x64xf32>)
    gpu.dealloc  %memref : memref<32x64xf16>
    gpu.dealloc  %memref_0 : memref<64x64xf16>
    %alloc = memref.alloc() : memref<32x64xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<32x64xf32>, memref<32x64xf32>
    gpu.dealloc  %memref_1 : memref<32x64xf32>
    return %alloc : memref<32x64xf32>
  }
  gpu.module @test_kernel  {
      /// canonicalize
    gpu.func @test_kernel(%arg0: memref<32x64xf16>, %arg1: memref<64x64xf16>, %arg2: memref<32x64xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      // intialize C tile and load it
      // k iter 0 : do a partial C tile 32x32x64
      // k iter 1 : update offsets and do a partial C tile 32x32x64
      // store the C tile result back to memory
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = xetile.init_tile %arg2[%c0, %c0] : memref<32x64xf32> -> !xetile.tile<32x64xf32>
      %1 = xetile.load_tile %0 : !xetile.tile<32x64xf32> -> vector<32x64xf32>
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 64], strides: [1, 64] : memref<64x64xf16> to memref<64x64xf16, strided<[1, 64]>>
      %2 = xetile.init_tile %arg0[%c0, %c0] : memref<32x64xf16> -> !xetile.tile<32x32xf16>
      %3 = xetile.init_tile %reinterpret_cast[%c0, %c0] : memref<64x64xf16, strided<[1, 64]>> -> !xetile.tile<32x64xf16, #xetile.tile_attr<order = [0, 1]>>
      %4 = xetile.load_tile %2 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %5 = xetile.load_tile %3 : !xetile.tile<32x64xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<32x64xf16>
      %6 = xetile.tile_mma %4, %5, %1 : vector<32x32xf16>, vector<32x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
      %7 = xetile.update_tile_offset %2, [%c0, %c32] : !xetile.tile<32x32xf16>
      %8 = xetile.update_tile_offset %3, [%c32, %c0] : !xetile.tile<32x64xf16, #xetile.tile_attr<order = [0, 1]>>
      %9 = xetile.load_tile %7 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %10 = xetile.load_tile %8 : !xetile.tile<32x64xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<32x64xf16>
      %11 = xetile.tile_mma %9, %10, %6 : vector<32x32xf16>, vector<32x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
      xetile.store_tile %11,  %0 : vector<32x64xf32>, !xetile.tile<32x64xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %true = arith.constant true
    %cst = arith.constant 3.000000e+00 : f32
    %cst_0 = arith.constant -3.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    // fill A, B with random values
    // fill C, C_ref with zeros
    // compute C for reference
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<32x64xf16>
    %alloc_2 = memref.alloc() : memref<64x64xf16>
    %alloc_3 = memref.alloc() : memref<32x64xf32>
    %alloc_4 = memref.alloc() : memref<32x64xf32>
    %cast = memref.cast %alloc : memref<32x64xf16> to memref<*xf16>
    %cast_5 = memref.cast %alloc_2 : memref<64x64xf16> to memref<*xf16>
    %cast_6 = memref.cast %alloc_3 : memref<32x64xf32> to memref<*xf32>
    %cast_7 = memref.cast %alloc_4 : memref<32x64xf32> to memref<*xf32>
    call @fillResource1DRandomF16(%cast, %cst_0, %cst, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%cast_5, %cst_0, %cst, %true) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DF32(%cast_6, %cst_1) : (memref<*xf32>, f32) -> ()
    call @fillResource1DF32(%cast_7, %cst_1) : (memref<*xf32>, f32) -> ()
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<32x64xf32>
        %2 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<32x64xf16>
          %4 = memref.load %alloc_2[%arg1, %arg2] : memref<64x64xf16>
          %5 = arith.mulf %3, %4 : f16
          %6 = arith.extf %5 : f16 to f32
          %7 = arith.addf %6, %arg3 : f32
          scf.yield %7 : f32
        }
        memref.store %2, %alloc_4[%arg0, %arg1] : memref<32x64xf32>
      }
    }
    // %cast = memref.cast %B : memref<1024x1024xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_2, %alloc_3) : (memref<32x64xf16>, memref<64x64xf16>, memref<32x64xf32>) -> memref<32x64xf32>
    %cast_8 = memref.cast %0 : memref<32x64xf32> to memref<*xf32>
    %cast_9 = memref.cast %alloc_4 : memref<32x64xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_8, %cast_9) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<32x64xf16>
    memref.dealloc %alloc_2 : memref<64x64xf16>
    memref.dealloc %alloc_3 : memref<32x64xf32>
    memref.dealloc %alloc_4 : memref<32x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
}
