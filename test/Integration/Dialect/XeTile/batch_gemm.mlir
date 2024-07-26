// TODO: replace "--no-mlir-runner" with "--runner imex-cpu-runner" once xesim simulator is available in CI.
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --no-mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --no-mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

// TODO: Remove this check and enable the check "[ALLCLOSE: TRUE]".
// CHECK: llvm.mlir.global internal constant @b2x3_m128_n256_k96_spirv_binary
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<2x3x128x96xf16>, %B: memref<2x3x256x96xf16>) -> memref<2x3x128x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %A_gpu = gpu.alloc  host_shared () : memref<2x3x128x96xf16>
    memref.copy %A, %A_gpu : memref<2x3x128x96xf16> to memref<2x3x128x96xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<2x3x256x96xf16>
    memref.copy %B, %B_gpu : memref<2x3x256x96xf16> to memref<2x3x256x96xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<2x3x128x256xf32>
    gpu.launch_func  @b2x3_m128_n256_k96::@b2x3_m128_n256_k96
                       blocks in (%c1, %c1, %c1)
                       threads in (%c4, %c8, %c1)
                       args(%A_gpu : memref<2x3x128x96xf16>,
                            %B_gpu : memref<2x3x256x96xf16>,
                            %C_gpu : memref<2x3x128x256xf32>)
    gpu.dealloc  %A_gpu : memref<2x3x128x96xf16>
    gpu.dealloc  %B_gpu : memref<2x3x256x96xf16>
    return %C_gpu : memref<2x3x128x256xf32>
  }

  gpu.module @b2x3_m128_n256_k96 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @b2x3_m128_n256_k96(%arg0: memref<2x3x128x96xf16>, %arg1: memref<2x3x256x96xf16>, %arg2: memref<2x3x128x256xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 4, 8, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c96 = arith.constant 96 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
      %thread_id_x_0 = gpu.thread_id  x
      %thread_id_y_1 = gpu.thread_id  y
      %0 = arith.muli %thread_id_x_0, %c8 : index
      %1 = arith.addi %0, %thread_id_y_1 : index
      %2 = arith.muli %1, %c4 : index
      %3 = arith.muli %1, %c8 : index
      %block_dim_y_2 = gpu.block_dim  y
      %4 = arith.muli %thread_id_x_0, %block_dim_y_2 : index
      %5 = arith.addi %4, %thread_id_y_1 : index
      %6 = arith.divsi %5, %c8 : index
      %7 = arith.remsi %5, %c8 : index
      %8 = arith.muli %6, %c32 : index
      %9 = arith.remsi %8, %c128 : index
      %10 = arith.muli %7, %c32 : index
      %11 = arith.remsi %10, %c256 : index
      %12 = arith.divsi %9, %c128 : index
      %13 = arith.muli %12, %c128 : index
      %14 = arith.divsi %11, %c256 : index
      %15 = arith.muli %14, %c256 : index
      %16 = arith.addi %13, %2 : index
      %17 = arith.addi %15, %3 : index
      %18 = arith.addi %13, %2 : index
      %19 = arith.addi %15, %3 : index
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c3 step %c1 {
          %20 = xetile.init_tile %arg0[%arg3, %arg4, %9, %c0] : memref<2x3x128x96xf16> -> !xetile.tile<32x32xf16>
          %21 = xetile.init_tile %arg1[%arg3, %arg4, %11, %c0] : memref<2x3x256x96xf16> -> !xetile.tile<32x32xf16>
          %22 = xetile.init_tile %arg0[%arg3, %arg4, %16, %c0] : memref<2x3x128x96xf16> -> !xetile.tile<4x32xf16>
          xetile.prefetch_tile %22 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
          %23 = xetile.update_tile_offset %22, [%c0,  %c32] : !xetile.tile<4x32xf16>, index, index -> !xetile.tile<4x32xf16>
          xetile.prefetch_tile %23 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
          %24 = xetile.update_tile_offset %23, [%c0,  %c32] : !xetile.tile<4x32xf16>, index, index -> !xetile.tile<4x32xf16>
          %25 = xetile.init_tile %arg1[%arg3, %arg4, %17, %c0] : memref<2x3x256x96xf16> -> !xetile.tile<8x32xf16>
          xetile.prefetch_tile %25 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
          %26 = xetile.update_tile_offset %25, [%c0,  %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
          xetile.prefetch_tile %26 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
          %27 = xetile.update_tile_offset %26, [%c0,  %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
          %28 = xetile.init_tile %arg0[%arg3, %arg4, %18, %c0] : memref<2x3x128x96xf16> -> !xetile.tile<4x32xf16>
          xetile.prefetch_tile %28 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
          %29 = xetile.update_tile_offset %28, [%c0,  %c32] : !xetile.tile<4x32xf16>, index, index -> !xetile.tile<4x32xf16>
          xetile.prefetch_tile %29 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
          %30 = xetile.update_tile_offset %29, [%c0,  %c32] : !xetile.tile<4x32xf16>, index, index -> !xetile.tile<4x32xf16>
          %31 = xetile.init_tile %arg1[%arg3, %arg4, %19, %c0] : memref<2x3x256x96xf16> -> !xetile.tile<8x32xf16>
          xetile.prefetch_tile %31 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
          %32 = xetile.update_tile_offset %31, [%c0,  %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
          xetile.prefetch_tile %32 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
          %33 = xetile.update_tile_offset %32, [%c0,  %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
          %34:8 = scf.for %arg5 = %c0 to %c96 step %c32 iter_args(%arg6 = %cst, %arg7 = %20, %arg8 = %21, %arg9 = %24, %arg10 = %27, %arg11 = %30, %arg12 = %33, %arg13 = %c0) -> (vector<32x32xf32>, !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<4x32xf16>, !xetile.tile<8x32xf16>, !xetile.tile<4x32xf16>, !xetile.tile<8x32xf16>, index) {
            %36 = xetile.load_tile %arg7 { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
            %37 = xetile.load_tile %arg8 { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
            %38 = arith.cmpi eq, %arg13, %c10 : index
            %39 = arith.select %38, %c0, %arg13 : index
            scf.if %38 {
              gpu.barrier
            }
            %40 = arith.addi %39, %c1 : index
            xegpu.compile_hint
            xetile.prefetch_tile %arg9 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
            xetile.prefetch_tile %arg10 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
            xetile.prefetch_tile %arg11 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
            xetile.prefetch_tile %arg12 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<8x32xf16>
            xegpu.compile_hint
            %41 = xetile.update_tile_offset %arg9, [%c0,  %c32] : !xetile.tile<4x32xf16>, index, index -> !xetile.tile<4x32xf16>
            %42 = xetile.update_tile_offset %arg10, [%c0,  %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
            %43 = xetile.update_tile_offset %arg11, [%c0,  %c32] : !xetile.tile<4x32xf16>, index, index -> !xetile.tile<4x32xf16>
            %44 = xetile.update_tile_offset %arg12, [%c0,  %c32] : !xetile.tile<8x32xf16>, index, index -> !xetile.tile<8x32xf16>
            %45 = vector.transpose %37, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
            xegpu.compile_hint
            %46 = xetile.update_tile_offset %arg7, [%c0,  %c32] : !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>
            %47 = xetile.update_tile_offset %arg8, [%c0,  %c32] : !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>
            xegpu.compile_hint
            %48 = xetile.tile_mma %36, %45, %arg6 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
            xegpu.compile_hint
            scf.yield %48, %46, %47, %41, %42, %43, %44, %40 : vector<32x32xf32>, !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<4x32xf16>, !xetile.tile<8x32xf16>, !xetile.tile<4x32xf16>, !xetile.tile<8x32xf16>, index
          } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 8>, step = 32 : index, upperBoundMap = affine_map<() -> (96)>}
          %35 = xetile.init_tile %arg2[%arg3, %arg4, %9, %11] : memref<2x3x128x256xf32> -> !xetile.tile<32x32xf32>
          xetile.store_tile %34#0,  %35 : vector<32x32xf32>, !xetile.tile<32x32xf32>
        } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, syn.parall_level = 2 : i64, upperBoundMap = affine_map<() -> (3)>}
      } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, syn.parall_level = 2 : i64, upperBoundMap = affine_map<() -> (2)>}
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c96 = arith.constant 96 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %cf_1 = arith.constant 1.0 : f16
    %cf_3 = arith.constant 3.0 : f16
    %cf_96 = arith.constant 96.0 : f32
    %A = memref.alloc() : memref<2x3x128x96xf16>
    %B = memref.alloc() : memref<2x3x256x96xf16>
    %C_ref = memref.alloc() : memref<2x3x128x256xf32>

    // The batch contains 6 gemms, fill the first A/B matrices with ones,
    // the second A/B matrices with twos, and so. The output should be:
    // first matrix filled with 1*1*96, the second one with 2*2*96, and so on.
    scf.for %i0 = %c0 to %c2 step %c1 {
      scf.for %i1 = %c0 to %c3 step %c1 {
        %i0_i16 = index.castu %i0 : index to i16
        %i0_f16 = arith.uitofp %i0_i16 : i16 to f16
        %i1_i16 = index.castu %i1 : index to i16
        %i1_f16 = arith.uitofp %i1_i16 : i16 to f16
        %v0 = arith.mulf %i0_f16, %cf_3 : f16
        %v1 = arith.addf %v0, %i1_f16 : f16
        %v = arith.addf %v1, %cf_1 : f16

        scf.for %i2 = %c0 to %c128 step %c1 {
          scf.for %i3 = %c0 to %c96 step %c1 {
            memref.store %v, %A[%i0, %i1, %i2, %i3] : memref<2x3x128x96xf16>
          }
        }

        scf.for %i2 = %c0 to %c256 step %c1 {
          scf.for %i3 = %c0 to %c96 step %c1 {
            memref.store %v, %B[%i0, %i1, %i2, %i3] : memref<2x3x256x96xf16>
          }
        }

        %r0 = arith.extf %v : f16 to f32
        %r1 = arith.mulf %r0, %r0 : f32
        %r = arith.mulf %r1, %cf_96 : f32

        scf.for %i2 = %c0 to %c128 step %c1 {
          scf.for %i3 = %c0 to %c256 step %c1 {
            memref.store %r, %C_ref[%i0, %i1, %i2, %i3] : memref<2x3x128x256xf32>
          }
        }
      }
    }

    %C = call @test(%A, %B) : (memref<2x3x128x96xf16>, memref<2x3x256x96xf16>) -> memref<2x3x128x256xf32>
    %cast_C = memref.cast %C : memref<2x3x128x256xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<2x3x128x256xf32> to memref<*xf32>
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // DISABLED-CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<2x3x128x96xf16>
    memref.dealloc %B : memref<2x3x256x96xf16>
    memref.dealloc %C_ref : memref<2x3x128x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
