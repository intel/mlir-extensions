// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// TODO: Add imex-runner commands
// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<1024x1024xi8>, %arg1: memref<1024x1024xi8>, %arg2: memref<1024x1024xi32>) -> memref<1024x1024xi32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<1024x1024xi8>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xi8>, memref<1024x1024xi8>
    %memref_0 = gpu.alloc  () : memref<1024x1024xi8>
    gpu.memcpy  %memref_0, %arg1 : memref<1024x1024xi8>, memref<1024x1024xi8>
    %memref_1 = gpu.alloc  () : memref<1024x1024xi32>
    gpu.memcpy  %memref_1, %arg2 : memref<1024x1024xi32>, memref<1024x1024xi32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xi8>, %memref_0 : memref<1024x1024xi8>, %memref_1 : memref<1024x1024xi32>)
    gpu.dealloc  %memref : memref<1024x1024xi8>
    gpu.dealloc  %memref_0 : memref<1024x1024xi8>
    %alloc = memref.alloc() : memref<1024x1024xi32>
    gpu.memcpy  %alloc, %memref_1 : memref<1024x1024xi32>, memref<1024x1024xi32>
    gpu.dealloc  %memref_1 : memref<1024x1024xi32>
    return %alloc : memref<1024x1024xi32>
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
    gpu.func @test_kernel(%arg0: memref<1024x1024xi8>, %arg1: memref<1024x1024xi8>, %arg2: memref<1024x1024xi32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
      %3 = xetile.load_tile %2 : !xetile.tile<16x32xi32> -> vector<16x32xi32>
      %4 = xetile.init_tile %arg0[%0, %c0] : memref<1024x1024xi8> -> !xetile.tile<16x64xi8>
      %5 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xi8> -> !xetile.tile<64x32xi8>
      %6:3 = scf.for %arg3 = %c0 to %c1024 step %c64 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3) -> (!xetile.tile<16x64xi8>, !xetile.tile<64x32xi8>, vector<16x32xi32>) {
        %7 = xetile.load_tile %arg4 : !xetile.tile<16x64xi8> -> vector<16x64xi8>
        %8 = xetile.load_tile %arg5 : !xetile.tile<64x32xi8> -> vector<64x32xi8>
        %9 = xetile.tile_mma %7, %8, %arg6 : vector<16x64xi8>, vector<64x32xi8>, vector<16x32xi32> -> vector<16x32xi32>
        %10 = xetile.update_tile_offset %arg4, [%c0, %c64] : !xetile.tile<16x64xi8>
        %11 = xetile.update_tile_offset %arg5, [%c64, %c0] : !xetile.tile<64x32xi8>
        scf.yield %10, %11, %9 : !xetile.tile<16x64xi8>, !xetile.tile<64x32xi8>, vector<16x32xi32>
      }
      xetile.store_tile %6#2,  %2 : vector<16x32xi32>, !xetile.tile<16x32xi32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // intialize matrix A ; A[i, j] = j
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %alloc = memref.alloc() : memref<1024x1024xi8>
    %alloc_0 = memref.alloc() : memref<1024x1024xi8>
    %alloc_1 = memref.alloc() : memref<1024x1024xi32>
    %alloc_2 = memref.alloc() : memref<1024x1024xi32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i8
        memref.store %1, %alloc[%arg0, %arg1] : memref<1024x1024xi8>
      }
    }
    // make matrix B an identity matrix
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg0 : index to i32
        %2 = index.castu %arg1 : index to i32
        %3 = arith.cmpi eq, %1, %2 : i32
        scf.if %3 {
          memref.store %c1_i8, %alloc_0[%arg0, %arg1] : memref<1024x1024xi8>
        } else {
          memref.store %c0_i8, %alloc_0[%arg0, %arg1] : memref<1024x1024xi8>
        }
      }
    }
    // intialize matrix C and C_ref ; C[i, j] = 0
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %c0_i32, %alloc_1[%arg0, %arg1] : memref<1024x1024xi32>
        memref.store %c0_i32, %alloc_2[%arg0, %arg1] : memref<1024x1024xi32>
      }
    }
    // compute C for reference
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_2[%arg0, %arg1] : memref<1024x1024xi32>
        %2 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %1) -> (i32) {
          %3 = memref.load %alloc[%arg0, %arg2] : memref<1024x1024xi8>
          %4 = memref.load %alloc_0[%arg2, %arg1] : memref<1024x1024xi8>
          %5 = arith.extui %3 : i8 to i32
          %6 = arith.extui %4 : i8 to i32
          %7 = arith.muli %5, %6 : i32
          %8 = arith.addi %7, %arg3 : i32
          scf.yield %8 : i32
        }
        memref.store %2, %alloc_2[%arg0, %arg1] : memref<1024x1024xi32>
      }
    }
    %0 = call @test(%alloc, %alloc_0, %alloc_1) : (memref<1024x1024xi8>, memref<1024x1024xi8>, memref<1024x1024xi32>) -> memref<1024x1024xi32>
    %cast = memref.cast %0 : memref<1024x1024xi32> to memref<*xi32>
    %cast_3 = memref.cast %alloc_2 : memref<1024x1024xi32> to memref<*xi32>
    call @printAllcloseI32(%cast, %cast_3) : (memref<*xi32>, memref<*xi32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xi8>
    memref.dealloc %alloc_0 : memref<1024x1024xi8>
    memref.dealloc %alloc_1 : memref<1024x1024xi32>
    memref.dealloc %alloc_2 : memref<1024x1024xi32>
    return
  }
  func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefI8(memref<*xi8>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseI32(memref<*xi32>, memref<*xi32>) attributes {llvm.emit_c_interface}
}

