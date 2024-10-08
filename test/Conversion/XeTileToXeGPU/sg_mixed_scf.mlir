
// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse --canonicalize %s -verify-diagnostics -o -| FileCheck %s

//CHECK-LABEL: gpu.module @postop_reduce_m
gpu.module @postop_reduce_m attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @postop_reduce_m(%arg0: memref<16384x12288xbf16>, %arg1: memref<2048x12288xbf16>, %arg2: memref<32x2048xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 8, 4, 1>, gpu.known_grid_size = array<i32: 8, 32, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c8 = arith.constant 8 : index
    %cst = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c12288 = arith.constant 12288 : index
    %c128 = arith.constant 128 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %cst_1 = arith.constant dense<0.000000e+00> : vector<32xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<4xf32>
    %thread_id_x = gpu.thread_id  x
    %thread_id_y = gpu.thread_id  y
    %block_dim_y = gpu.block_dim  y
    %0 = arith.muli %thread_id_x, %block_dim_y : index
    %1 = arith.addi %0, %thread_id_y : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %2 = arith.divsi %block_id_y, %c8 : index
    %3 = arith.remsi %block_id_y, %c8 : index
    %4 = arith.divsi %1, %c4 : index
    %5 = arith.remsi %1, %c4 : index
    %6 = arith.muli %4, %c32 : index
    %7 = arith.remsi %6, %c256 : index
    %8 = arith.muli %5, %c12288 : index
    %9 = arith.remsi %8, %c12288 : index
    %10 = arith.muli %block_id_x, %c2048 : index
    %11 = arith.muli %2, %c256 : index
    %12 = arith.muli %5, %c32 : index
    %13 = arith.remsi %12, %c128 : index
    %14 = arith.muli %4, %c12288 : index
    %15 = arith.remsi %14, %c12288 : index
    %16 = arith.muli %3, %c128 : index
    %17 = arith.remsi %4, %c8 : index
    %18 = arith.divsi %1, %c32 : index
    %19 = arith.remsi %1, %c32 : index
    %20 = arith.muli %18, %c8 : index
    %21 = arith.remsi %20, %c8 : index
    %22 = arith.muli %19, %c4 : index
    %23 = arith.remsi %22, %c128 : index
    %24 = arith.muli %block_id_x, %c4 : index
    %25 = arith.addi %24, %2 : index

    scf.for %arg3 = %c0 to %c2 step %c1 {
      %26 = arith.muli %arg3, %c1024 : index
      %27 = arith.addi %26, %13 : index
      %28 = arith.addi %27, %16 : index
      //CHECK: %{{.*}} = xegpu.create_nd_tdesc {{.*}} : memref<2048x12288xbf16> -> !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
      %29 = xetile.init_tile %arg1[%28, %15] : memref<2048x12288xbf16> -> !xetile.tile<32x32xbf16>

      %30 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %cst) -> (vector<1x4xf32>) {
        %33 = arith.muli %arg4, %c1024 : index
        %34 = arith.addi %33, %7 : index
        %35 = arith.addi %34, %10 : index
        %36 = arith.addi %35, %11 : index

        //CHECK: %{{.*}} = xegpu.create_nd_tdesc %{{.*}} : memref<16384x12288xbf16> -> !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
        %37 = xetile.init_tile %arg0[%36, %9] : memref<16384x12288xbf16> -> !xetile.tile<32x32xbf16>
        %38:3 = scf.for %arg6 = %c0 to %c12288 step %c32 iter_args(%arg7 = %37, %arg8 = %29, %arg9 = %cst_0) -> (!xetile.tile<32x32xbf16>, !xetile.tile<32x32xbf16>, vector<32x32xf32>) {

          //CHECK: %{{.*}} = xegpu.load_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xbf16>
          //CHECK-COUNT-2: %{{.*}} = vector.extract %{{.*}} : vector<32x16xbf16> from vector<2x32x16xbf16>
          //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = {{.*}}, sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %48 = xetile.load_tile %arg7 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xbf16> -> vector<32x32xbf16>


          //CHECK: %{{.*}} = xegpu.load_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xbf16>
          //CHECK-COUNT-2: %{{.*}} = vector.extract %{{.*}} : vector<32x16xbf16> from vector<2x32x16xbf16>
          //CHECK-COUNT-4: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = {{.*}}, sizes = [16, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<16x16xbf16>
          %49 = xetile.load_tile %arg8 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xbf16> -> vector<32x32xbf16>

          //CHECK: %{{.*}} = xegpu.update_nd_offset %{{.*}} : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
          //CHECK: %{{.*}} = xegpu.update_nd_offset %{{.*}} : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
          %50 = xetile.update_tile_offset %arg7, [%c0,  %c32] : !xetile.tile<32x32xbf16>, index, index -> !xetile.tile<32x32xbf16>
          %51 = xetile.update_tile_offset %arg8, [%c0,  %c32] : !xetile.tile<32x32xbf16>, index, index -> !xetile.tile<32x32xbf16>

          //CHECK-COUNT-16: %{{.*}} = xegpu.dpas %{{.*}}, %{{.*}}, %{{.*}} : vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %52 = xetile.tile_mma %48, %49, %arg9 : vector<32x32xbf16>, vector<32x32xbf16>, vector<32x32xf32> -> vector<32x32xf32>
          scf.yield %50, %51, %52 : !xetile.tile<32x32xbf16>, !xetile.tile<32x32xbf16>, vector<32x32xf32>
        } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 3>, step = 32 : index, upperBoundMap = affine_map<() -> (12288)>}

        //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = {{.*}}, sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        //CHECK-COUNT-64: %{{.*}} = math.exp %{{.*}} : vector<1x16xf32>
        %39 = math.exp %38#2 : vector<32x32xf32>

        //CHECK-COUNT-62: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x16xf32>
        //CHECK-COUNT-2: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x16xf32> to vector<16xf32>
        //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %40 = vector.multi_reduction <add>, %39, %cst_1 [0] : vector<32x32xf32> to vector<32xf32>
        %41 = vector.shape_cast %40 : vector<32xf32> to vector<1x32xf32>
        %alloc = memref.alloc() : memref<8x128xf32, 3>

        //CHECK: %{{.*}} = xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x32xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        %42 = xetile.init_tile %alloc[%17, %13] : memref<8x128xf32, 3> -> !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3>>

        //CHECK: xegpu.store_nd %{{.*}}, %{{.*}} <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x32xf32>, !xegpu.tensor_desc<1x32xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        xetile.store_tile %41,  %42 : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<memory_space = 3>>

        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK: xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf32, 3> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
        //CHECK-COUNT-8: xegpu.load_nd {{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x4xf32>
        //CHECK-COUNT-8: arith.addf %{{.*}}, %{{.*}} : vector<1x4xf32>
        %43 = xetile.init_tile %alloc[%21, %23] : memref<8x128xf32, 3> -> !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3>>
        %44 = xetile.load_tile %43 {padding = 0.000000e+00 : f32}  : !xetile.tile<8x4xf32, #xetile.tile_attr<memory_space = 3>> -> vector<8x4xf32>
        %45 = vector.multi_reduction <add>, %44, %cst_2 [0] : vector<8x4xf32> to vector<4xf32>
        %46 = vector.shape_cast %45 : vector<4xf32> to vector<1x4xf32>
        %47 = arith.addf %arg5, %46 : vector<1x4xf32>
        scf.yield %47 : vector<1x4xf32>
      } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, syn.mm_dim = 0 : i64, syn.parall_level = 2 : i64, upperBoundMap = affine_map<() -> (2)>}

      //CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
      //CHECK: %{{.*}} = xegpu.create_nd_tdesc %{{.*}} : memref<32x2048xf32> -> !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      //CHECK: xegpu.store_nd %{{.*}}, %{{.*}} <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x4xf32>, !xegpu.tensor_desc<1x4xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %31 = arith.addi %26, %16 : index
      %32 = xetile.init_tile %arg2[%25, %31] : memref<32x2048xf32> -> !xetile.tile<1x4xf32>
      xetile.store_tile %30,  %32 : vector<1x4xf32>, !xetile.tile<1x4xf32>
    } {lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, syn.mm_dim = 1 : i64, syn.parall_level = 2 : i64, upperBoundMap = affine_map<() -> (2)>}
    gpu.return
  }
}
