// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (384)>
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<128x384xf16>, %B: memref<1x384xf16>) -> memref<128x256xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %A_gpu = gpu.alloc  host_shared () : memref<128x384xf16>
    memref.copy %A, %A_gpu : memref<128x384xf16> to memref<128x384xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1x384xf16>
    memref.copy %B, %B_gpu : memref<1x384xf16> to memref<1x384xf16>
    %D_gpu = gpu.alloc  host_shared () : memref<128x256xf32>
    gpu.launch_func  @m128_n256_k384::@m128_n256_k384 blocks in (%c1, %c1, %c1) threads in (%c4, %c8, %c1) args(%A_gpu : memref<128x384xf16>, %B_gpu : memref<1x384xf16>, %D_gpu : memref<128x256xf32>)
    gpu.dealloc  %A_gpu : memref<128x384xf16>
    gpu.dealloc  %B_gpu : memref<1x384xf16>
    return %D_gpu : memref<128x256xf32>
  }

  gpu.module @m128_n256_k384 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @m128_n256_k384(%arg0: memref<128x384xf16>, %arg1: memref<1x384xf16>, %arg2: memref<128x256xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 4, 8, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c384 = arith.constant 384 : index
      %c51 = arith.constant 51 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %0 = arith.muli %thread_id_x, %c8 : index
      %1 = arith.addi %0, %thread_id_y : index
      %2 = arith.muli %1, %c4 : index
      %block_dim_y = gpu.block_dim  y
      %3 = arith.muli %thread_id_x, %block_dim_y : index
      %4 = arith.addi %3, %thread_id_y : index
      %5 = arith.divsi %4, %c8 : index
      %6 = arith.remsi %4, %c8 : index
      %7 = arith.muli %5, %c32 : index
      %8 = arith.remsi %7, %c128 : index
      %9 = arith.muli %6, %c32 : index
      %10 = arith.remsi %9, %c256 : index
      %11 = arith.divsi %8, %c128 : index
      %12 = arith.muli %11, %c128 : index
      %13 = xetile.init_tile %arg0[%8, %c0] : memref<128x384xf16> -> !xetile.tile<32x32xf16>
      %14 = xetile.init_tile %arg1[%c0, %c0] : memref<1x384xf16> -> !xetile.tile<1x32xf16>
      %15 = arith.addi %12, %2 : index
      %16 = xetile.init_tile %arg0[%15, %c0] : memref<128x384xf16> -> !xetile.tile<4x32xf16>
      xetile.prefetch_tile %16 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
      %17 = xetile.update_tile_offset %16, [%c0,  %c32] : !xetile.tile<4x32xf16>
      %18 = xetile.init_tile %arg1[%1, %c0] : memref<1x384xf16> -> !xetile.tile<1x32xf16>
      xetile.prefetch_tile %18 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x32xf16>
      %19 = xetile.update_tile_offset %18, [%c0,  %c32] : !xetile.tile<1x32xf16>
      %20:6 = scf.for %arg3 = %c0 to %c384 step %c32 iter_args(%arg4 = %cst, %arg5 = %13, %arg6 = %14, %arg7 = %17, %arg8 = %19, %arg9 = %c0) -> (vector<32x32xf32>, !xetile.tile<32x32xf16>, !xetile.tile<1x32xf16>, !xetile.tile<4x32xf16>, !xetile.tile<1x32xf16>, index) {
        %22 = xetile.load_tile %arg5 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %23 = xetile.load_tile %arg6 {padding = 0.000000e+00 : f32}  : !xetile.tile<1x32xf16> -> vector<1x32xf16>
        %24 = arith.cmpi eq, %arg9, %c51 : index
        %25 = arith.select %24, %c0, %arg9 : index
        scf.if %24 {
          gpu.barrier
        }
        %26 = arith.addi %25, %c1 : index
        xegpu.compile_hint
        xetile.prefetch_tile %arg7 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x32xf16>
        xetile.prefetch_tile %arg8 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x32xf16>
        xegpu.compile_hint
        %27 = xetile.update_tile_offset %arg7, [%c0,  %c32] : !xetile.tile<4x32xf16>
        %28 = xetile.update_tile_offset %arg8, [%c0,  %c32] : !xetile.tile<1x32xf16>
        %29 = vector.transpose %23, [1, 0] : vector<1x32xf16> to vector<32x1xf16>
        xegpu.compile_hint
        %30 = xetile.broadcast %29 [1] : vector<32x1xf16> -> vector<32x32xf16>
        %31 = xetile.update_tile_offset %arg5, [%c0,  %c32] :  !xetile.tile<32x32xf16>
        %32 = xetile.update_tile_offset %arg6, [%c0,  %c32] : !xetile.tile<1x32xf16>
        xegpu.compile_hint
        // Use broadcast result directly
        %33 = xetile.tile_mma %22, %30, %arg4 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        xegpu.compile_hint
        scf.yield %33, %31, %32, %27, %28, %26 : vector<32x32xf32>, !xetile.tile<32x32xf16>, !xetile.tile<1x32xf16>, !xetile.tile<4x32xf16>, !xetile.tile<1x32xf16>, index
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 6>, step = 32 : index, upperBoundMap = #map1}
      %21 = xetile.init_tile %arg2[%8, %10] : memref<128x256xf32> -> !xetile.tile<32x32xf32>
      xetile.store_tile %20#0,  %21 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c384 = arith.constant 384 : index
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<128x384xf16>
    %B = memref.alloc() : memref<1x384xf16>
    %C_ref = memref.alloc() : memref<128x256xf32>
    // Make matrix A an identity matrix
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c384 step %c1 {
        memref.store %cf_1, %A[%i, %j] : memref<128x384xf16>
      }
    }
    // Make matrix B an identity matrix
    scf.for %j = %c0 to %c384 step %c1 {
      memref.store %cf_1, %B[%c0, %j] : memref<1x384xf16>
    }
    // intialize matrix C_ref
    %cf_384 = arith.constant 384.0 : f32
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c256 step %c1 {
        memref.store %cf_384, %C_ref[%i, %j] : memref<128x256xf32>
      }
    }
    %C = call @test(%A, %B) : (memref<128x384xf16>, memref<1x384xf16>) -> memref<128x256xf32>
    %cast_C = memref.cast %C : memref<128x256xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<128x256xf32> to memref<*xf32>
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<128x384xf16>
    memref.dealloc %B : memref<1x384xf16>
    memref.dealloc %C_ref : memref<128x256xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
