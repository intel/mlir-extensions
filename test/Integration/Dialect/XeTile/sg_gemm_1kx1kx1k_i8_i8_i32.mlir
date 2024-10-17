// TODO: Add imex-runner commands
// RUN:

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<1024x1024xi8>, %B: memref<1024x1024xi8>, %C: memref<1024x1024xi32>) -> memref<1024x1024xi32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc  host_shared () : memref<1024x1024xi8>
    memref.copy %A, %A_gpu : memref<1024x1024xi8> to memref<1024x1024xi8>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xi8>
    memref.copy %B, %B_gpu : memref<1024x1024xi8> to memref<1024x1024xi8>
    %C_gpu = gpu.alloc  host_shared () : memref<1024x1024xi32>
    memref.copy %C, %C_gpu : memref<1024x1024xi32> to memref<1024x1024xi32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<1024x1024xi8>, %B_gpu : memref<1024x1024xi8>, %C_gpu : memref<1024x1024xi32>)
    gpu.dealloc  %A_gpu : memref<1024x1024xi8>
    gpu.dealloc  %B_gpu : memref<1024x1024xi8>
    return %C_gpu : memref<1024x1024xi32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<1024x1024xi8>, %B: memref<1024x1024xi8>, %C: memref<1024x1024xi32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c1024 = arith.constant 1024 : index
        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y
        %m = arith.muli %block_id_x, %c16: index
        %n = arith.muli %block_id_y, %c32: index
        // intialize C tile and load it
        %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
        %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<16x32xi32> -> vector<16x32xi32>
        // initalize A and B tiles
        %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xi8> -> !xetile.tile<16x64xi8>
        %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xi8> -> !xetile.tile<64x32xi8>
        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
        %out:3 = scf.for %k = %c0 to %c1024 step %c64
          iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
          -> (!xetile.tile<16x64xi8>, !xetile.tile<64x32xi8>, vector<16x32xi32>) {

          // load A and B tiles
          %a_value = xetile.load_tile %a_tile  : !xetile.tile<16x64xi8> -> vector<16x64xi8>
          %b_value = xetile.load_tile %b_tile : !xetile.tile<64x32xi8> -> vector<64x32xi8>
          // perform dpas and accumulate
          %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
            : vector<16x64xi8>, vector<64x32xi8>, vector<16x32xi32> -> vector<16x32xi32>
          // update the offsets for A and B tiles
          %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64] : !xetile.tile<16x64xi8>
          %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0] : !xetile.tile<64x32xi8>
          // partial C tile result
          scf.yield %a_next_tile, %b_next_tile, %c_new_value
            : !xetile.tile<16x64xi8>, !xetile.tile<64x32xi8>, vector<16x32xi32>
        }
        // store the final accumulated C tile result back to memory
        xetile.store_tile %out#2, %c_init_tile : vector<16x32xi32>, !xetile.tile<16x32xi32>
        gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %ci_0 = arith.constant 0 : i8
    %ci_1 = arith.constant 1 : i8
    %A = memref.alloc() : memref<1024x1024xi8>
    %B = memref.alloc() : memref<1024x1024xi8>
    %C = memref.alloc() : memref<1024x1024xi32>
    %C_ref = memref.alloc() : memref<1024x1024xi32>
    // intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %val = index.castu %j : index to i8
        memref.store %val, %A[%i, %j] : memref<1024x1024xi8>
      }
    }
    // make matrix B an identity matrix
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %i_i32 = index.castu %i : index to i32
        %j_i32 = index.castu %j : index to i32
        %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

        scf.if %i_j_same {
          memref.store %ci_1, %B[%i, %j] : memref<1024x1024xi8>
        } else {
          memref.store %ci_0, %B[%i, %j] : memref<1024x1024xi8>
        }
      }
    }
    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_i32 = arith.constant 0: i32
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        memref.store %c0_i32, %C[%i, %j] : memref<1024x1024xi32>
        memref.store %c0_i32, %C_ref[%i, %j] : memref<1024x1024xi32>
      }
    }
    // compute C for reference
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %c_curr = memref.load %C_ref[%i, %j] : memref<1024x1024xi32>
        %c_val = scf.for %k = %c0 to %c1024 step %c1 iter_args(%c_partial = %c_curr) -> i32 {
          %a_val = memref.load %A[%i, %k] : memref<1024x1024xi8>
          %b_val = memref.load %B[%k, %j] : memref<1024x1024xi8>
          %a_val_i32 = arith.extui %a_val : i8 to i32
          %b_val_i32 = arith.extui %b_val : i8 to i32
          %t = arith.muli %a_val_i32, %b_val_i32 : i32
          %c_sum = arith.addi %t, %c_partial : i32
          scf.yield %c_sum : i32
        }
        memref.store %c_val , %C_ref[%i, %j] : memref<1024x1024xi32>
      }
    }
    %2 = call @test(%A, %B, %C) : (memref<1024x1024xi8>, memref<1024x1024xi8>, memref<1024x1024xi32>) -> memref<1024x1024xi32>
    %cast_C = memref.cast %2 : memref<1024x1024xi32> to memref<*xi32>
    %cast_C_ref = memref.cast %C_ref : memref<1024x1024xi32> to memref<*xi32>

    call @printAllcloseI32(%cast_C, %cast_C_ref) : (memref<*xi32>, memref<*xi32>) -> ()
    memref.dealloc %A : memref<1024x1024xi8>
    memref.dealloc %B : memref<1024x1024xi8>
    memref.dealloc %C : memref<1024x1024xi32>
    memref.dealloc %C_ref : memref<1024x1024xi32>
    return
  }
  func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefI8(memref<*xi8>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseI32(memref<*xi32>, memref<*xi32>) attributes {llvm.emit_c_interface}
}
