#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (64)>
module attributes {gpu.container_module} {
  func.func @bcast_add_exp_entry(%arg0: memref<1x64xf16>, %arg1: memref<512x64xf16>, %arg2: memref<1x64xf16>, %arg3: memref<512x256xf32>) attributes {gemm_tiles_b = 1 : i64, gemm_tiles_x = dense<[1, 1, 1, 16]> : vector<4xi64>, gemm_tiles_y = dense<[4, 1, 1, 2]> : vector<4xi64>, physical_nd_range = dense<[4, 1]> : vector<2xi64>, region_partition = 0 : i64, region_size = 1 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<1x64xf16>, tensor<512x64xf16>, tensor<1x64xf16>) -> tensor<512x256xf32>, synFusionGenOps = 6 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 9630.3999999999996 : f64} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    gpu.launch_func  @bcast_add_exp::@bcast_add_exp blocks in (%c4, %c1, %c1) threads in (%c16, %c2, %c1)  args(%arg0 : memref<1x64xf16>, %arg1 : memref<512x64xf16>, %arg2 : memref<1x64xf16>, %arg3 : memref<512x256xf32>)
    return
  }
  gpu.module @bcast_add_exp attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @bcast_add_exp(%arg0: memref<1x64xf16>, %arg1: memref<512x64xf16>, %arg2: memref<1x64xf16>, %arg3: memref<512x256xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 16, 2, 1>, known_grid_size = array<i32: 4, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c127 = arith.constant 127 : index
      %c1023 = arith.constant 1023 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %0 = arith.muli %thread_id_x, %c2 : index
      %1 = arith.addi %0, %thread_id_y : index
      %2 = arith.muli %1, %c16 : index
      %block_dim_y = gpu.block_dim  y
      %3 = arith.muli %thread_id_x, %block_dim_y : index
      %4 = arith.addi %3, %thread_id_y : index
      %block_id_x = gpu.block_id  x
      %5 = arith.remsi %block_id_x, %c4 : index
      %6 = arith.remsi %5, %c4 : index
      %7 = arith.divsi %4, %c2 : index
      %8 = arith.remsi %4, %c2 : index
      %9 = arith.muli %7, %c32 : index
      %10 = arith.remsi %9, %c512 : index
      %11 = arith.muli %8, %c32 : index
      %12 = arith.remsi %11, %c64 : index
      %13 = xetile.init_tile %arg0[%c0, %c0] {xetile.wg_data = dense<[16, 32]> : vector<2xi32>} : memref<1x64xf16> -> !xetile.tile<1x32xf16>
      %14 = arith.divsi %10, %c512 : index
      %15 = arith.muli %14, %c512 : index
      %16 = arith.addi %15, %2 : index
      %17 = xetile.init_tile %arg1[%10, %c0] {xetile.wg_data = dense<[512, 32]> : vector<2xi32>} : memref<512x64xf16> -> !xetile.tile<32x32xf16>
      %18 = xetile.init_tile %arg2[%c0, %c0] {xetile.wg_data = dense<[2, 32]> : vector<2xi32>} : memref<1x64xf16> -> !xetile.tile<1x32xf16>
      %19 = xetile.init_tile %arg0[%1, %c0] : memref<1x64xf16> -> !xetile.tile<1x128xf16>
      xetile.prefetch_tile %19 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x128xf16>
      %20 = xetile.update_tile_offset %19, [%c0, %c128] : !xetile.tile<1x128xf16>
      %21 = xetile.init_tile %arg1[%16, %c0] : memref<512x64xf16> -> !xetile.tile<16x128xf16>
      xetile.prefetch_tile %21 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<16x128xf16>
      %22 = xetile.update_tile_offset %21, [%c0, %c128] : !xetile.tile<16x128xf16>
      %23 = xetile.init_tile %arg2[%1, %c0] : memref<1x64xf16> -> !xetile.tile<1x128xf16>
      xetile.prefetch_tile %23 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x128xf16>
      %24 = xetile.update_tile_offset %23, [%c0, %c128] : !xetile.tile<1x128xf16>
      %25 = xetile.init_tile %arg0[%1, %c0] : memref<1x64xf16> -> !xetile.tile<1x32xf16>
      xetile.prefetch_tile %25 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x32xf16>
      %26 = xetile.update_tile_offset %25, [%c0, %c32] : !xetile.tile<1x32xf16>
      %27 = xetile.init_tile %arg1[%16, %c0] : memref<512x64xf16> -> !xetile.tile<16x32xf16>
      xetile.prefetch_tile %27 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<16x32xf16>
      %28 = xetile.update_tile_offset %27, [%c0, %c32] : !xetile.tile<16x32xf16>
      %29 = xetile.init_tile %arg2[%1, %c0] : memref<1x64xf16> -> !xetile.tile<1x32xf16>
      xetile.prefetch_tile %29 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x32xf16>
      %30 = xetile.update_tile_offset %29, [%c0, %c32] : !xetile.tile<1x32xf16>
      %31:10 = scf.for %arg4 = %c0 to %c64 step %c32 iter_args(%arg5 = %cst, %arg6 = %13, %arg7 = %17, %arg8 = %18, %arg9 = %20, %arg10 = %22, %arg11 = %24, %arg12 = %26, %arg13 = %28, %arg14 = %30) -> (vector<32x32xf32>, !xetile.tile<1x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<1x32xf16>, !xetile.tile<1x128xf16>, !xetile.tile<16x128xf16>, !xetile.tile<1x128xf16>, !xetile.tile<1x32xf16>, !xetile.tile<16x32xf16>, !xetile.tile<1x32xf16>) {
        %35 = xetile.update_tile_offset %arg8, [%c0, %c32] : !xetile.tile<1x32xf16>
        %36 = xetile.update_tile_offset %arg7, [%c0, %c32] : !xetile.tile<32x32xf16>
        %37 = xetile.update_tile_offset %arg6, [%c0, %c32] : !xetile.tile<1x32xf16>
        %38 = xetile.load_tile %arg6 : !xetile.tile<1x32xf16> -> vector<1x32xf16>
        %39 = xetile.load_tile %arg7 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %40 = xetile.load_tile %arg8 : !xetile.tile<1x32xf16> -> vector<1x32xf16>
        %41 = arith.andi %arg4, %c1023 : index
        %42 = arith.cmpi eq, %41, %c0 : index
        scf.if %42 {
          gpu.barrier
        }
        xegpu.compile_hint
        %43 = arith.andi %arg4, %c127 : index
        %44 = arith.cmpi eq, %43, %c0 : index
        scf.if %44 {
          xetile.prefetch_tile %arg9 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x128xf16>
        }
        %45 = xetile.update_tile_offset %arg9, [%c0, %c32] : !xetile.tile<1x128xf16>
        scf.if %44 {
          xetile.prefetch_tile %arg10 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<16x128xf16>
        }
        %46 = xetile.update_tile_offset %arg10, [%c0, %c32] : !xetile.tile<16x128xf16>
        scf.if %44 {
          xetile.prefetch_tile %arg11 {l1_hint = #xetile.cache_hint<uncached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x128xf16>
        }
        %47 = xetile.update_tile_offset %arg11, [%c0, %c32] : !xetile.tile<1x128xf16>
        xetile.prefetch_tile %arg12 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x32xf16>
        %48 = xetile.update_tile_offset %arg12, [%c0, %c32] : !xetile.tile<1x32xf16>
        xetile.prefetch_tile %arg13 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<16x32xf16>
        %49 = xetile.update_tile_offset %arg13, [%c0, %c32] : !xetile.tile<16x32xf16>
        xetile.prefetch_tile %arg14 {l1_hint = #xetile.cache_hint<cached>, l2_hint = #xetile.cache_hint<cached>} : !xetile.tile<1x32xf16>
        %50 = xetile.update_tile_offset %arg14, [%c0, %c32] : !xetile.tile<1x32xf16>
        %51 = vector.transpose %40, [1, 0] : vector<1x32xf16> to vector<32x1xf16>
        xegpu.compile_hint
        %52 = xetile.broadcast %38 [0] : vector<1x32xf16> -> vector<32x32xf16>
        %53 = arith.addf %52, %39 : vector<32x32xf16>
        %54 = math.exp %51 : vector<32x1xf16>
        %55 = xetile.broadcast %54 [1] : vector<32x1xf16> -> vector<32x32xf16>
        xegpu.compile_hint
        %56 = xetile.tile_mma %53, %55, %arg5 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        xegpu.compile_hint
        scf.yield %56, %37, %36, %35, %45, %46, %47, %48, %49, %50 : vector<32x32xf32>, !xetile.tile<1x32xf16>, !xetile.tile<32x32xf16>, !xetile.tile<1x32xf16>, !xetile.tile<1x128xf16>, !xetile.tile<16x128xf16>, !xetile.tile<1x128xf16>, !xetile.tile<1x32xf16>, !xetile.tile<16x32xf16>, !xetile.tile<1x32xf16>
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 10>, step = 32 : index, upperBoundMap = #map1}
      %32 = arith.muli %6, %c64 : index
      %33 = arith.addi %32, %12 : index
      %34 = xetile.init_tile %arg3[%10, %33] : memref<512x256xf32> -> !xetile.tile<32x32xf32>
      xetile.store_tile %31#0,  %34 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      gpu.return
    }
  }
}
