// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

module attributes {gpu.container_module} {
  func.func @tiles_b_2_entry(%arg0: memref<4x3x2x128x96xf16>, %arg1: memref<4x3x2x64x96xf16>, %arg2: memref<4x3x2x128x64xf32>) attributes {gemm_tiles_b = 2 : i64, gemm_tiles_x = dense<[2, 1, 1, 2]> : vector<4xi64>, gemm_tiles_y = dense<[2, 1, 1, 1]> : vector<4xi64>, habana_runner.num_inputs = 2 : i64, habana_runner.tests = [{inputs = [dense<1.000000e+00> : tensor<4x3x2x128x96xf16>, dense<1.000000e+00> : tensor<4x3x2x64x96xf16>], outputs = [dense<9.600000e+01> : tensor<4x3x2x128x64xf32>]}], physical_nd_range = dense<[8, 1]> : vector<2xi64>, region_partition = 0 : i64, region_size = 1 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<4x3x2x128x96xf16>, tensor<4x3x2x64x96xf16>) -> tensor<4x3x2x128x64xf32>, synFusionGenOps = 9 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1000021695.96 : f64} {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    gpu.launch_func  @tiles_b_2::@tiles_b_2 blocks in (%c8, %c1, %c1) threads in (%c2, %c1, %c1)  args(%arg0 : memref<4x3x2x128x96xf16>, %arg1 : memref<4x3x2x64x96xf16>, %arg2 : memref<4x3x2x128x64xf32>)
    return
  }
  gpu.module @tiles_b_2 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @tiles_b_2(%arg0: memref<4x3x2x128x96xf16>, %arg1: memref<4x3x2x64x96xf16>, %arg2: memref<4x3x2x128x64xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 2, 1, 1>, known_grid_size = array<i32: 8, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c96 = arith.constant 96 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} dense<0.000000e+00> : vector<64x32xf32>
      %c3 = arith.constant 3 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %block_id_x = gpu.block_id  x
      %0 = arith.remsi %block_id_x, %c4 : index
      %1 = arith.divsi %0, %c2 : index
      %2 = arith.remsi %0, %c2 : index
      %3 = arith.muli %block_id_x, %c2 : index
      %4 = arith.divsi %3, %c8 : index
      %5 = arith.muli %4, %c12 : index
      %6 = arith.muli %1, %c64 : index
      %7 = arith.muli %2, %c32 : index
      scf.for %arg3 = %c0 to %c12 step %c1 {
        %8 = arith.addi %5, %arg3 : index
        %9 = arith.divsi %8, %c6 : index
        %10 = arith.remsi %9, %c4 : index
        %11 = arith.divsi %8, %c2 : index
        %12 = arith.remsi %11, %c3 : index
        %13 = arith.remsi %8, %c2 : index
        //CHECK: %[[INITTILE:.*]] = xetile.init_tile {{%.*}}[{{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}] : memref<4x3x2x128x96xf16> -> !xetile.tile<32x32xf16>
        //CHECK: %[[INITTILE:.*]] = xetile.init_tile {{%.*}}[{{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}] : memref<4x3x2x64x96xf16> -> !xetile.tile<32x32xf16>
        %14 = xetile.init_tile %arg0[%10, %12, %13, %6, %c0] : memref<4x3x2x128x96xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %15 = xetile.init_tile %arg1[%10, %12, %13, %7, %c0] : memref<4x3x2x64x96xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        %16:3 = scf.for %arg4 = %c0 to %c96 step %c32 iter_args(%arg5 = %cst, %arg6 = %14, %arg7 = %15) -> (vector<64x32xf32>, !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>) {
          %18 = xetile.update_tile_offset %arg7, [%c0, %c32] : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
          %19 = xetile.update_tile_offset %arg6, [%c0, %c32] : !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
          %20 = xetile.load_tile %arg6 : !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<64x32xf16>
          %21 = xetile.load_tile %arg7 : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>> -> vector<32x32xf16>
          %22 = vector.transpose %21, [1, 0] {map = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} : vector<32x32xf16> to vector<32x32xf16>
          xegpu.compile_hint
          %23 = xetile.tile_mma %20, %22, %cst {wg_map_a = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>, wg_map_c = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} : vector<64x32xf16>, vector<32x32xf16>, vector<64x32xf32> -> vector<64x32xf32>
          xegpu.compile_hint
          %24 = arith.addf %arg5, %23 {map = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} : vector<64x32xf32>
          scf.yield %24, %19, %18 : vector<64x32xf32>, !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        }
        %17 = xetile.init_tile %arg2[%10, %12, %13, %6, %7] : memref<4x3x2x128x64xf32> -> !xetile.tile<64x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
        xetile.store_tile %16#0,  %17 : vector<64x32xf32>, !xetile.tile<64x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, memory_space = 0 : i32, scattered = false>>
      }
      gpu.return
    }
  }
}
