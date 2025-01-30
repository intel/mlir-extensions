gpu.module @gemm_m4096_n4096_k4096_AB attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @gemm_m4096_n4096_k4096_AB(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 4, 8, 1>, known_grid_size = array<i32: 2, 28, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c0_i8 = arith.constant 0 : i8
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %c511 = arith.constant 511 : index
    %c32_i8 = arith.constant 32 : i8
    %c2048 = arith.constant 2048 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c2240 = arith.constant 2240 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %cst = arith.constant dense<0.000000e+00> : vector<80x64xf32>
    %thread_id_x = gpu.thread_id  x
    %thread_id_y = gpu.thread_id  y
    %0 = arith.muli %thread_id_x, %c8 : index
    %1 = arith.addi %0, %thread_id_y : index
    %2 = arith.divui %1, %c16 : index
    %3 = arith.remsi %1, %c16 : index
    %4 = arith.muli %2, %c16 : index
    %5 = arith.muli %3, %c32 : index
    %6 = arith.divui %1, %c4 : index
    %7 = arith.remsi %1, %c4 : index
    %8 = arith.muli %6, %c4 : index
    %9 = arith.muli %7, %c128 : index
    %block_dim_y = gpu.block_dim  y
    %10 = arith.muli %thread_id_x, %block_dim_y : index
    %11 = arith.addi %10, %thread_id_y : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %12 = arith.remsi %block_id_x, %c2 : index
    %13 = arith.remsi %12, %c2 : index
    %14 = arith.divsi %block_id_y, %c4 : index
    %15 = arith.remsi %14, %c7 : index
    %16 = arith.remsi %block_id_y, %c4 : index
    %17 = arith.remsi %16, %c4 : index
    %18 = arith.divsi %11, %c8 : index
    %19 = arith.remsi %11, %c8 : index
    %20 = arith.muli %18, %c80 : index
    %21 = arith.remsi %20, %c320 : index
    %22 = arith.muli %19, %c64 : index
    %23 = arith.remsi %22, %c512 : index
    %24 = arith.muli %13, %c2048 : index
    %25 = arith.muli %17, %c512 : index
    %26 = arith.addi %24, %25 : index
    %27 = arith.addi %26, %23 : index
    %28 = arith.muli %15, %c320 : index
    %29 = arith.addi %28, %21 : index
    %30 = xetile.init_tile %arg2[%29, %27] : memref<4096x4096xf16> -> !xetile.tile<80x64xf16>
    %31 = xetile.init_tile %arg0[%29, %c0] {xetile.wg_data = dense<[320, 32]> : vector<2xi32>} : memref<4096x4096xf16> -> !xetile.tile<80x32xf16>
    %32 = arith.divsi %27, %c512 : index
    %33 = arith.muli %32, %c512 : index
    %34 = arith.addi %33, %5 : index
    %35 = arith.addi %33, %9 : index
    %36 = xetile.init_tile %arg1[%c0, %27] {xetile.wg_data = dense<[32, 512]> : vector<2xi32>} : memref<4096x4096xf16> -> !xetile.tile<32x64xf16>
    %37 = xetile.init_tile %arg1[%8, %35] : memref<4096x4096xf16> -> !xetile.tile<4x128xf16>
    %38 = xetile.init_tile %arg1[%4, %34] : memref<4096x4096xf16> -> !xetile.tile<16x32xf16>
    %39:2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %30, %arg5 = %31) -> (!xetile.tile<80x64xf16>, !xetile.tile<80x32xf16>) {
      %40 = xetile.update_tile_offset %arg5, [%c2240, %c0] : !xetile.tile<80x32xf16>
      %41 = xetile.update_tile_offset %arg4, [%c2240, %c0] : !xetile.tile<80x64xf16>
      xegpu.alloc_nbarrier 1
      %42 = xegpu.init_nbarrier %c0_i8, %c32_i8 : i8, i8 -> !xegpu.nbarrier
      xetile.prefetch_tile %37 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x128xf16>
      %43 = xetile.update_tile_offset %37, [%c32, %c0] : !xetile.tile<4x128xf16>
      xetile.prefetch_tile %38 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<16x32xf16>
      %44 = xetile.update_tile_offset %38, [%c32, %c0] : !xetile.tile<16x32xf16>
      %45:5 = scf.for %arg6 = %c0 to %c4096 step %c32 iter_args(%arg7 = %cst, %arg8 = %arg5, %arg9 = %36, %arg10 = %43, %arg11 = %44) -> (vector<80x64xf32>, !xetile.tile<80x32xf16>, !xetile.tile<32x64xf16>, !xetile.tile<4x128xf16>, !xetile.tile<16x32xf16>) {
        %47 = xetile.update_tile_offset %arg9, [%c32, %c0] : !xetile.tile<32x64xf16>
        %48 = xetile.update_tile_offset %arg8, [%c0, %c32] : !xetile.tile<80x32xf16>
        %49 = xetile.load_tile %arg8 {padding = 0.000000e+00 : f32} : !xetile.tile<80x32xf16> -> vector<80x32xf16>
        %50 = xetile.load_tile %arg9 : !xetile.tile<32x64xf16> -> vector<32x64xf16>
        xetile.prefetch_tile %arg10 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<4x128xf16>
        %51 = xetile.update_tile_offset %arg10, [%c32, %c0] : !xetile.tile<4x128xf16>
        xetile.prefetch_tile %arg11 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<16x32xf16>
        %52 = xetile.update_tile_offset %arg11, [%c32, %c0] : !xetile.tile<16x32xf16>
        %53 = arith.andi %arg6, %c511 : index
        %54 = arith.cmpi eq, %53, %c0 : index
        scf.if %54 {
          xegpu.nbarrier_arrive %42 : !xegpu.nbarrier
        }
        xegpu.compile_hint
        xegpu.compile_hint
        %55 = xetile.tile_mma %49, %50, %arg7 : vector<80x32xf16>, vector<32x64xf16>, vector<80x64xf32> -> vector<80x64xf32>
        xegpu.compile_hint
        scf.if %54 {
          xegpu.nbarrier_wait %42 : !xegpu.nbarrier
        }
        xegpu.compile_hint
        scf.yield %55, %48, %47, %51, %52 : vector<80x64xf32>, !xetile.tile<80x32xf16>, !xetile.tile<32x64xf16>, !xetile.tile<4x128xf16>, !xetile.tile<16x32xf16>
      } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 5>, step = 32 : index, upperBoundMap = affine_map<() -> (4096)>}
      %46 = arith.truncf %45#0 : vector<80x64xf32> to vector<80x64xf16>
      xetile.store_tile %46,  %arg4 : vector<80x64xf16>, !xetile.tile<80x64xf16>
      scf.yield %41, %40 : !xetile.tile<80x64xf16>, !xetile.tile<80x32xf16>
    } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 2>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}
    gpu.return
  }
}
