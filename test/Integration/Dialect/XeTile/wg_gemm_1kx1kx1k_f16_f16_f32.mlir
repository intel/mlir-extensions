// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#wg_map_a = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 128]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a>

#wg_map_b = #xetile.wg_map<sg_layout = [4, 4], sg_data = [128, 32]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b>

#wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 32]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c>

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %A, %A_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %B, %B_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %C, %C_gpu : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c8, %c8, %c1) threads in (%c4, %c4, %c1) args(%A_gpu : memref<1024x1024xf16>, %B_gpu : memref<1024x1024xf16>, %C_gpu : memref<1024x1024xf32>)
    gpu.dealloc  %A_gpu : memref<1024x1024xf16>
    gpu.dealloc  %B_gpu : memref<1024x1024xf16>
    return %C_gpu : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index

        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y
        %m = arith.muli %block_id_x, %c128 : index
        %n = arith.muli %block_id_y, %c128 : index

        // intialize C tile and load it
        %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32>
          -> !xetile.tile<128x128xf32, #tile_attr_c>
        %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x128xf32, #tile_attr_c>
          -> vector<128x128xf32>

        %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_a>

        %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16>
          -> !xetile.tile<128x128xf16, #tile_attr_b>

        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
        %out:3 = scf.for %k = %c0 to %c1024 step %c128
          iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
          -> (!xetile.tile<128x128xf16, #tile_attr_a>,
              !xetile.tile<128x128xf16, #tile_attr_b>,
              vector<128x128xf32>) {

          // load A and B tiles
          %a_value = xetile.load_tile %a_tile  : !xetile.tile<128x128xf16, #tile_attr_a>
            -> vector<128x128xf16>

          %b_value = xetile.load_tile %b_tile : !xetile.tile<128x128xf16, #tile_attr_b>
            -> vector<128x128xf16>

          // perform dpas and accumulate
          %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c}
            : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>

          // update the offsets for A and B tiles
          %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c128] : !xetile.tile<128x128xf16, #tile_attr_a>
          %b_next_tile = xetile.update_tile_offset %b_tile, [%c128, %c0] : !xetile.tile<128x128xf16, #tile_attr_b>
          // partial C tile result
          scf.yield %a_next_tile, %b_next_tile, %c_new_value
            : !xetile.tile<128x128xf16, #tile_attr_a>,
            !xetile.tile<128x128xf16, #tile_attr_b>, vector<128x128xf32>
        }
        // store the final accumulated C tile result back to memory
        xetile.store_tile %out#2, %c_init_tile : vector<128x128xf32>,
          !xetile.tile<128x128xf32, #tile_attr_c>
        gpu.return
    }
    }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<1024x1024xf16>
    %B = memref.alloc() : memref<1024x1024xf16>
    %C = memref.alloc() : memref<1024x1024xf32>
    %C_ref = memref.alloc() : memref<1024x1024xf32>
    // intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %t = index.castu %j : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %A[%i, %j] : memref<1024x1024xf16>
      }
    }
    // make matrix B an identity matrix
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %i_i32 = index.castu %i : index to i32
        %j_i32 = index.castu %j : index to i32
        %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

        scf.if %i_j_same {
          memref.store %cf_1, %B[%i, %j] : memref<1024x1024xf16>
        } else {
          memref.store %cf_0, %B[%i, %j] : memref<1024x1024xf16>
        }
      }
    }
    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<1024x1024xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }
    // compute C for reference
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %c_curr = memref.load %C_ref[%i, %j] : memref<1024x1024xf32>
        %c_val = scf.for %k = %c0 to %c1024 step %c1 iter_args(%c_partial = %c_curr) -> f32 {
          %a_val = memref.load %A[%i, %k] : memref<1024x1024xf16>
          %b_val = memref.load %B[%k, %j] : memref<1024x1024xf16>
          %t = arith.mulf %a_val, %b_val : f16
          %t_cast = arith.extf %t : f16 to f32
          %c_sum = arith.addf %t_cast, %c_partial : f32
          scf.yield %c_sum : f32
        }
        memref.store %c_val , %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }
    %2 = call @test(%A, %B, %C) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast_C = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<1024x1024xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<1024x1024xf16>
    memref.dealloc %B : memref<1024x1024xf16>
    memref.dealloc %C : memref<1024x1024xf32>
    memref.dealloc %C_ref : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
