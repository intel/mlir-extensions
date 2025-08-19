// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#wg_map_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a>

#wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b>

#wg_map_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c>

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<4096x4096xbf16>, %B: memref<4096x4096xbf16>, %C: memref<4096x4096xf32>) -> memref<4096x4096xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc  host_shared () : memref<4096x4096xbf16>
    memref.copy %A, %A_gpu : memref<4096x4096xbf16> to memref<4096x4096xbf16>
    %B_gpu = gpu.alloc  host_shared () : memref<4096x4096xbf16>
    memref.copy %B, %B_gpu : memref<4096x4096xbf16> to memref<4096x4096xbf16>
    %C_gpu = gpu.alloc  host_shared () : memref<4096x4096xf32>
    memref.copy %C, %C_gpu : memref<4096x4096xf32> to memref<4096x4096xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c16, %c16, %c1) threads in (%c8, %c4, %c1) args(%A_gpu : memref<4096x4096xbf16>, %B_gpu : memref<4096x4096xbf16>, %C_gpu : memref<4096x4096xf32>)
    gpu.dealloc  %A_gpu : memref<4096x4096xbf16>
    gpu.dealloc  %B_gpu : memref<4096x4096xbf16>
    return %C_gpu : memref<4096x4096xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<4096x4096xbf16>, %B: memref<4096x4096xbf16>, %C: memref<4096x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c256 = arith.constant 256 : index
        %c4096 = arith.constant 4096 : index
        %block_id_x = gpu.block_id x
        %block_id_y = gpu.block_id y
        %m = arith.muli %block_id_x, %c256 : index
        %n = arith.muli %block_id_y, %c256 : index
        // intialize C tile and load it
        // %prefetch_c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32>
          // -> !xetile.tile<256x256xf32, #tile_attr_c>
        %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32>
          -> !xetile.tile<256x256xf32, #tile_attr_c>
        %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<256x256xf32, #tile_attr_c>
          -> vector<256x256xf32>

        // initalize A and B tiles
        %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<4096x4096xbf16>
          -> !xetile.tile<256x32xbf16, #tile_attr_a>
        %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xbf16>
          -> !xetile.tile<32x256xbf16, #tile_attr_b>

        // prefetch first 32 slice
        %prefetch_a_init_tile_1 = xetile.init_tile %A[%m, %c0] : memref<4096x4096xbf16>
          -> !xetile.tile<256x32xbf16, #tile_attr_a>
        %prefetch_b_init_tile_1 = xetile.init_tile %B[%c0, %n] : memref<4096x4096xbf16>
          -> !xetile.tile<32x256xbf16, #tile_attr_b>
        xetile.prefetch_tile %prefetch_a_init_tile_1 : !xetile.tile<256x32xbf16, #tile_attr_a>
        xetile.prefetch_tile %prefetch_b_init_tile_1 : !xetile.tile<32x256xbf16, #tile_attr_b>

        // prefetch second 32 slice
        %prefetch_a_init_tile_2 = xetile.init_tile %A[%m, %c32] : memref<4096x4096xbf16>
          -> !xetile.tile<256x32xbf16, #tile_attr_a>
        %prefetch_b_init_tile_2 = xetile.init_tile %B[%c32, %n] : memref<4096x4096xbf16>
          -> !xetile.tile<32x256xbf16, #tile_attr_b>
        xetile.prefetch_tile %prefetch_a_init_tile_2 : !xetile.tile<256x32xbf16, #tile_attr_a>
        xetile.prefetch_tile %prefetch_b_init_tile_2 : !xetile.tile<32x256xbf16, #tile_attr_b>


        // prefetch third 32 slice
        %prefetch_a_init_tile_3 = xetile.init_tile %A[%m, %c64] : memref<4096x4096xbf16>
          -> !xetile.tile<256x32xbf16, #tile_attr_a>
        %prefetch_b_init_tile_3 = xetile.init_tile %B[%c64, %n] : memref<4096x4096xbf16>
          -> !xetile.tile<32x256xbf16, #tile_attr_b>

        xegpu.alloc_nbarrier 1
        %nbarrier_id = arith.constant 0 : i8
        %num_threads = arith.constant 32 : i8
        %nbarrier = xegpu.init_nbarrier %nbarrier_id, %num_threads : i8, i8 -> !xegpu.nbarrier
        %c0_i32 = arith.constant 0 : i32

        // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
        %out:5 = scf.for %k = %c0 to %c4096 step %c32
          iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value,
          %prefetch_a_tile = %prefetch_a_init_tile_3,
          %prefetch_b_tile = %prefetch_b_init_tile_3
          )
          -> (!xetile.tile<256x32xbf16, #tile_attr_a>,
              !xetile.tile<32x256xbf16, #tile_attr_b>,
              vector<256x256xf32>,
              !xetile.tile<256x32xbf16, #tile_attr_a>,
              !xetile.tile<32x256xbf16, #tile_attr_b>
              ) {

          // all SGs must arrive here first
          // %every_8th_iter = arith.remui %k, %c256 : index
          // %every_8th_iter_i32 = arith.index_cast %every_8th_iter : index to i32
          // %every_8th_iter_cond = arith.cmpi eq, %every_8th_iter_i32, %c0_i32 : i32
          // scf.if %every_8th_iter_cond  {
            xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
          // }


          // load A and B tiles
          %a_value = xetile.load_tile %a_tile  : !xetile.tile<256x32xbf16, #tile_attr_a>
            -> vector<256x32xbf16>
          %b_value = xetile.load_tile %b_tile : !xetile.tile<32x256xbf16, #tile_attr_b>
            -> vector<32x256xbf16>

          xegpu.compile_hint

          // prefetch next A and B tiles
          xetile.prefetch_tile %prefetch_a_tile : !xetile.tile<256x32xbf16, #tile_attr_a>
          xetile.prefetch_tile %prefetch_b_tile : !xetile.tile<32x256xbf16, #tile_attr_b>

          xegpu.compile_hint

          // update prefetch tile offsets
          %15 = xetile.update_tile_offset %prefetch_a_tile, [%c0,  %c32] : !xetile.tile<256x32xbf16, #tile_attr_a>
          %16 = xetile.update_tile_offset %prefetch_b_tile, [%c32,  %c0] : !xetile.tile<32x256xbf16, #tile_attr_b>
          // update the offsets for A and B tiles
          %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]
            : !xetile.tile<256x32xbf16, #tile_attr_a>
          %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]
            : !xetile.tile<32x256xbf16, #tile_attr_b>

          xegpu.compile_hint

          // perform dpas and accumulate
          %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c}
            : vector<256x32xbf16>, vector<32x256xbf16>, vector<256x256xf32> -> vector<256x256xf32>

          xegpu.compile_hint
          //  barrier wait
          // scf.if %every_8th_iter_cond {
            xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
          // }
          // partial C tile result
          scf.yield %a_next_tile, %b_next_tile, %c_new_value, %15, %16
            : !xetile.tile<256x32xbf16, #tile_attr_a>,
            !xetile.tile<32x256xbf16, #tile_attr_b>, vector<256x256xf32>,
            !xetile.tile<256x32xbf16, #tile_attr_a>,
            !xetile.tile<32x256xbf16, #tile_attr_b>
        }
        // store the final accumulated C tile result back to memory
        %c_init_tile_1 = xetile.init_tile %C[%m, %n] : memref<4096x4096xf32>
          -> !xetile.tile<256x256xf32, #tile_attr_c>
        xetile.store_tile %out#2, %c_init_tile_1 : vector<256x256xf32>,
          !xetile.tile<256x256xf32, #tile_attr_c>
        xegpu.compile_hint
        gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_f16 = arith.constant 1.0 : bf16
    %c2_f16 = arith.constant 2.0 : bf16
    %c4096 = arith.constant 4096 : index
    %cf_0 = arith.constant 0.0 : bf16
    %cf_1 = arith.constant 1.0 : bf16
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant 0.0 : f32
    %cf_upper = arith.constant 1.0 : f32

    %A = memref.alloc() : memref<4096x4096xbf16>
    %B = memref.alloc() : memref<4096x4096xbf16>
    %C = memref.alloc() : memref<4096x4096xf32>
    %C_ref = memref.alloc() : memref<4096x4096xf32>

    // convert the memref to 1D and fill with random values in (0.0, 1.0)
    %A_random = memref.cast %A : memref<4096x4096xbf16> to memref<*xbf16>
    call @fillResource1DRandomBF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xbf16>, f32, f32, i1) -> ()

    // convert the memref to 1D and fill with random values in (0.0, 1.0)
    %B_random = memref.cast %B : memref<4096x4096xbf16>  to memref<*xbf16>
    call @fillResource1DRandomBF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xbf16>, f32, f32, i1) -> ()

    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f16 = arith.constant 0.0 : bf16
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<4096x4096xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<4096x4096xf32>
      }
    }

    // Run GPU.
    %2 = call @test(%A, %B, %C) : (memref<4096x4096xbf16>, memref<4096x4096xbf16>, memref<4096x4096xf32>) -> memref<4096x4096xf32>
    %cast_C = memref.cast %2 : memref<4096x4096xf32> to memref<*xf32>

    // Run CPU
    %A_cast = memref.cast %A : memref<4096x4096xbf16> to memref<*xbf16>
    %B_cast = memref.cast %B : memref<4096x4096xbf16> to memref<*xbf16>
    %cast_C_ref = memref.cast %C_ref : memref<4096x4096xf32> to memref<*xf32>
    call @gemmBF16BF16F32(%A_cast, %B_cast, %cast_C_ref) : (memref<*xbf16>, memref<*xbf16>, memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<4096x4096xbf16>
    memref.dealloc %B : memref<4096x4096xbf16>
    memref.dealloc %C : memref<4096x4096xf32>
    memref.dealloc %C_ref : memref<4096x4096xf32>
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseBF16(memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomBF16(memref<*xbf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @gemmBF16BF16F32(memref<*xbf16>, memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
