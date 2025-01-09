// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (12288)>
#map2 = affine_map<() -> (2)>
module attributes {gpu.container_module} {
  func.func @postop_reduce_m_entry(%arg0: memref<16384x12288xbf16>, %arg1: memref<2048x12288xbf16>, %arg2: memref<32x2048xf32>) attributes {gemm_tiles_b = 1 : i64, gemm_tiles_x = dense<[8, 2, 4, 8]> : vector<4xi64>, gemm_tiles_y = dense<[1, 2, 8, 4]> : vector<4xi64>, physical_nd_range = dense<[8, 32]> : vector<2xi64>, region_partition = 0 : i64, region_size = 32 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<16384x12288xbf16>, tensor<2048x12288xbf16>) -> tensor<32x2048xf32>, synFusionGenOps = 6 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1003595802.6 : f64} {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @postop_reduce_m::@postop_reduce_m blocks in (%c8, %c32, %c1) threads in (%c8, %c4, %c1)  args(%arg0 : memref<16384x12288xbf16>, %arg1 : memref<2048x12288xbf16>, %arg2 : memref<32x2048xf32>)
    return
  }
  gpu.module @postop_reduce_m attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @postop_reduce_m(%arg0: memref<16384x12288xbf16>, %arg1: memref<2048x12288xbf16>, %arg2: memref<32x2048xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 8, 4, 1>, known_grid_size = array<i32: 8, 32, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c12288 = arith.constant 12288 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c2048 = arith.constant 2048 : index
      %c128 = arith.constant 128 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} dense<0.000000e+00> : vector<256x128xf32>
      //CHECK: %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<1x32xf32>
      %cst_0 = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} dense<0.000000e+00> : vector<8x128xf32>
      //CHECK: %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
      %cst_1 = arith.constant {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} dense<0.000000e+00> : vector<128xf32>
      %cst_2 = arith.constant {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} dense<0.000000e+00> : vector<1x128xf32>
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.divsi %block_id_y, %c8 : index
      %1 = arith.remsi %block_id_y, %c8 : index
      %2 = arith.muli %block_id_x, %c4 : index
      %3 = arith.addi %2, %0 : index
      %4 = arith.muli %1, %c128 : index
      %5 = xetile.init_tile %arg2[%3, %4] : memref<32x2048xf32> -> !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>, memory_space = 0 : i32, scattered = false>>
      %6 = arith.muli %block_id_x, %c2048 : index
      %7 = arith.muli %0, %c256 : index
      %8 = arith.addi %6, %7 : index
      %9 = xetile.init_tile %arg0[%8, %c0] : memref<16384x12288xbf16> -> !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
      %10 = xetile.init_tile %arg1[%4, %c0] : memref<2048x12288xbf16> -> !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
      %11:2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %5, %arg5 = %10) -> (!xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>) {
        %12 = xetile.update_tile_offset %arg5, [%c1024, %c0] : !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %13 = xetile.update_tile_offset %arg4, [%c0, %c1024] : !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>, memory_space = 0 : i32, scattered = false>>
        %14:2 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %cst_2, %arg8 = %9) -> (vector<1x128xf32>, !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>) {
          %16 = xetile.update_tile_offset %arg8, [%c1024, %c0] : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
          %17:3 = scf.for %arg9 = %c0 to %c12288 step %c32 iter_args(%arg10 = %cst, %arg11 = %arg8, %arg12 = %arg5) -> (vector<256x128xf32>, !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>) {
            %27 = xetile.update_tile_offset %arg12, [%c0, %c32] : !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
            %28 = xetile.update_tile_offset %arg11, [%c0, %c32] : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
            %29 = xetile.load_tile %arg11 : !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<256x32xbf16>
            %30 = math.exp %29 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x32xbf16>
            %31 = xetile.load_tile %arg12 : !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<128x32xbf16>
            %32 = vector.transpose %31, [1, 0] {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<128x32xbf16> to vector<32x128xbf16>
            xegpu.compile_hint
            %33 = xetile.tile_mma %30, %32, %arg10 {wg_map_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>, wg_map_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x32xbf16>, vector<32x128xbf16>, vector<256x128xf32> -> vector<256x128xf32>
            xegpu.compile_hint
            scf.yield %33, %28, %27 : vector<256x128xf32>, !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
          } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 3>, step = 32 : index, upperBoundMap = #map1}
          //CHECK: %[[EXP:.*]] = math.exp {{%.*}} : vector<32x32xf32>
          //CHECK: %[[SHAPECAST_0:.*]] = vector.shape_cast %[[CST_0]] : vector<1x32xf32> to vector<32xf32>
          //CHECK: %[[REDUCTION_0:.*]] = vector.multi_reduction <add>, %[[EXP]], %[[SHAPECAST_0]] [0] : vector<32x32xf32> to vector<32xf32>
          //CHECK: %[[SHAPECAST_1:.*]] = vector.shape_cast %[[REDUCTION_0]] : vector<32xf32> to vector<1x32xf32>
          %18 = math.exp %17#0 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32>
          %19 = vector.shape_cast %18 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32> to vector<8x32x128xf32>
          %20 = vector.multi_reduction <add>, %19, %cst_0 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} [1] : vector<8x32x128xf32> to vector<8x128xf32>
          //CHECK: xetile.store_tile  {{%.*}}, {{%.*}} : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3 : i32>>
          //CHECK: gpu.barrier
          //CHECK: %[[LOADTILE_SLM:.*]] = xetile.load_tile {{%.*}} : !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3 : i32>> -> vector<8x4xf32>
          //CHECK: %[[REDUCTION_1:.*]] = vector.multi_reduction <add>, %[[LOADTILE_SLM]], %[[CST_1]] [0] : vector<8x4xf32> to vector<4xf32>
          //CHECK: %[[SHAPECAST_2:.*]] = vector.shape_cast %[[REDUCTION_1]] : vector<4xf32> to vector<1x4xf32>
          //CHECK: %[[ADDF:.*]] = arith.addf %[[SHAPECAST_2]], {{%.*}} :  vector<1x4xf32>
          %21 = xetile.convert_layout %20 {wg_map_result = #xetile.wg_map<sg_layout = [1, 32], sg_data = [8, 4]>} : vector<8x128xf32>
          %22 = vector.multi_reduction <add>, %21, %cst_1 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} [0] : vector<8x128xf32> to vector<128xf32>
          %23 = vector.shape_cast %22 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} : vector<128xf32> to vector<1x128xf32>
          %24 = arith.addf %23, %arg7 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} : vector<1x128xf32>
          scf.yield %24, %16 : vector<1x128xf32>, !xetile.tile<256x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 2>, step = 1 : index, upperBoundMap = #map2}
        xetile.store_tile %14#0,  %arg4 : vector<1x128xf32>, !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>, memory_space = 0 : i32, scattered = false>>
        scf.yield %13, %12 : !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<128x32xbf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 2>, step = 1 : index, upperBoundMap = #map2}
      gpu.return
    }
  }
}
