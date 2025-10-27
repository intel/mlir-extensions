// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<1024x1024xf32>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xf32>, memref<1024x1024xf32>
    %memref_0 = gpu.alloc  () : memref<1024x1024xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xf32>, memref<1024x1024xf32>
    %memref_1 = gpu.alloc  () : memref<1024x1024xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf32>, %memref_0 : memref<1024x1024xf32>, %memref_1 : memref<1024x1024xf32>)
    gpu.dealloc  %memref : memref<1024x1024xf32>
    gpu.dealloc  %memref_0 : memref<1024x1024xf32>
    %alloc = memref.alloc() : memref<1024x1024xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.dealloc  %memref_1 : memref<1024x1024xf32>
    return %alloc : memref<1024x1024xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
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
        // perform dpas and accumulate
        // update the offsets for A and B tiles
        // partial C tile result
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<16x32xf32>
      %3 = xetile.load_tile %2 : !xetile.tile<16x32xf32> -> vector<16x32xf32>
      %4 = xetile.init_tile %arg0[%0, %c0] : memref<1024x1024xf32> -> !xetile.tile<16x32xf32>
      %5 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
      %6:3 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3) -> (!xetile.tile<16x32xf32>, !xetile.tile<32x32xf32>, vector<16x32xf32>) {
        %7 = xetile.load_tile %arg4 : !xetile.tile<16x32xf32> -> vector<16x32xf32>
        %8 = xetile.load_tile %arg5 : !xetile.tile<32x32xf32> -> vector<32x32xf32>
        %9 = arith.truncf %7 : vector<16x32xf32> to vector<16x32xf16>
        %10 = arith.truncf %8 : vector<32x32xf32> to vector<32x32xf16>
        %11 = xetile.tile_mma %9, %10, %arg6 : vector<16x32xf16>, vector<32x32xf16>, vector<16x32xf32> -> vector<16x32xf32>
        %12 = xetile.update_tile_offset %arg4, [%c0, %c32] : !xetile.tile<16x32xf32>
        %13 = xetile.update_tile_offset %arg5, [%c32, %c0] : !xetile.tile<32x32xf32>
        scf.yield %12, %13, %11 : !xetile.tile<16x32xf32>, !xetile.tile<32x32xf32>, vector<16x32xf32>
      }
      // store the final accumulated C tile result back to memory
      xetile.store_tile %6#2,  %2 : vector<16x32xf32>, !xetile.tile<16x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // intialize matrix A ; A[i, j] = j
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() : memref<1024x1024xf32>
    %alloc_1 = memref.alloc() : memref<1024x1024xf32>
    %alloc_2 = memref.alloc() : memref<1024x1024xf32>
    %alloc_3 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i32
        %2 = arith.uitofp %1 : i32 to f32
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // make matrix B an identity matrix
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %cst_0, %alloc_1[%arg0, %arg1] : memref<1024x1024xf32>
        } else {
          memref.store %cst, %alloc_1[%arg0, %arg1] : memref<1024x1024xf32>
        }
      }
    }
    // intialize matrix C and C_ref ; C[i, j] = 0
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_2[%arg0, %arg1] : memref<1024x1024xf32>
        memref.store %cst, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // compute C for reference
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (f32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<1024x1024xf32>
          %4 = memref.load %alloc_1[%arg2, %arg1] : memref<1024x1024xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = arith.addf %5, %arg3 : f32
          scf.yield %6 : f32
        }
        memref.store %2, %alloc_3[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    // %cast = memref.cast %B : memref<1024x1024xf32> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // %C_row_0 = memref.subview %2[0, 0][1, 1024][1, 1] : memref<1024x1024xf32> to memref<1x1024xf32>
    // %C_row_0_cast = memref.cast %C_row_0 : memref<1x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @test(%alloc, %alloc_1, %alloc_2) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast = memref.cast %0 : memref<1024x1024xf32> to memref<*xf32>
    %cast_4 = memref.cast %alloc_3 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_4) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf32>
    memref.dealloc %alloc_1 : memref<1024x1024xf32>
    memref.dealloc %alloc_2 : memref<1024x1024xf32>
    memref.dealloc %alloc_3 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

