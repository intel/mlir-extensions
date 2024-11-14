// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

module attributes {gpu.container_module} {
  func.func @tiles_b_4_oob_entry(%arg0: memref<2x3x3x128x96xf16>, %arg1: memref<2x3x3x64x96xf16>, %arg2: memref<2x3x3x128x64xf32>) attributes {gemm_tiles_b = 4 : i64, gemm_tiles_x = dense<[2, 1, 1, 2]> : vector<4xi64>, gemm_tiles_y = dense<[1, 1, 2, 1]> : vector<4xi64>, habana_runner.num_inputs = 2 : i64, habana_runner.tests = [{inputs = [dense<1.000000e+00> : tensor<2x3x3x128x96xf16>, dense<1.000000e+00> : tensor<2x3x3x64x96xf16>], outputs = [dense<9.600000e+01> : tensor<2x3x3x128x64xf32>]}], physical_nd_range = dense<[8, 2]> : vector<2xi64>, region_partition = 0 : i64, region_size = 2 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<2x3x3x128x96xf16>, tensor<2x3x3x64x96xf16>) -> tensor<2x3x3x128x64xf32>, synFusionGenOps = 9 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1000016310.36 : f64} {
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @tiles_b_4_oob::@tiles_b_4_oob blocks in (%c8, %c2, %c1) threads in (%c2, %c1, %c1)  args(%arg0 : memref<2x3x3x128x96xf16>, %arg1 : memref<2x3x3x64x96xf16>, %arg2 : memref<2x3x3x128x64xf32>)
    return
  }
  gpu.module @tiles_b_4_oob attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @tiles_b_4_oob(%arg0: memref<2x3x3x128x96xf16>, %arg1: memref<2x3x3x64x96xf16>, %arg2: memref<2x3x3x128x64xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 2, 1, 1>, known_grid_size = array<i32: 8, 2, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c96 = arith.constant 96 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} dense<0.000000e+00> : vector<64x32xf32>
      %c3 = arith.constant 3 : index
      %c9 = arith.constant 9 : index
      %c18 = arith.constant 18 : index
      %c5 = arith.constant 5 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.remsi %block_id_x, %c2 : index
      %1 = arith.remsi %block_id_y, %c2 : index
      %2 = arith.muli %block_id_x, %c4 : index
      %3 = arith.divsi %2, %c8 : index
      %4 = arith.muli %3, %c5 : index
      %5 = arith.subi %c18, %4 : index
      %6 = arith.cmpi sgt, %5, %c5 : index
      %7 = arith.select %6, %c5, %5 : index
      %8 = arith.muli %0, %c64 : index
      %9 = arith.muli %1, %c32 : index
      scf.for %arg3 = %c0 to %7 step %c1 {
        %10 = arith.addi %4, %arg3 : index
        %11 = arith.divsi %10, %c9 : index
        %12 = arith.remsi %11, %c2 : index
        %13 = arith.divsi %10, %c3 : index
        %14 = arith.remsi %13, %c3 : index
        %15 = arith.remsi %10, %c3 : index
        //CHECK: %[[INITTILE:.*]] = xetile.init_tile {{%.*}}[{{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}] : memref<2x3x3x128x96xf16> -> !xetile.tile<32x32xf16>
        //CHECK: %[[INITTILE:.*]] = xetile.init_tile {{%.*}}[{{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}] : memref<2x3x3x64x96xf16> -> !xetile.tile<32x32xf16>
        %16 = xetile.init_tile %arg0[%12, %14, %15, %8, %c0] : memref<2x3x3x128x96xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        %17 = xetile.init_tile %arg1[%12, %14, %15, %9, %c0] : memref<2x3x3x64x96xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        %18:3 = scf.for %arg4 = %c0 to %c96 step %c32 iter_args(%arg5 = %cst, %arg6 = %16, %arg7 = %17) -> (vector<64x32xf32>, !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>) {
          %20 = xetile.update_tile_offset %arg7, [%c0, %c32] : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
          %21 = xetile.update_tile_offset %arg6, [%c0, %c32] : !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
          %22 = xetile.load_tile %arg6 : !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>> -> vector<64x32xf16>
          %23 = xetile.load_tile %arg7 : !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>> -> vector<32x32xf16>
          %24 = vector.transpose %23, [1, 0] {map = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} : vector<32x32xf16> to vector<32x32xf16>
          xegpu.compile_hint
          %25 = xetile.tile_mma %22, %24, %cst {wg_map_a = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>, wg_map_b = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>, wg_map_c = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} : vector<64x32xf16>, vector<32x32xf16>, vector<64x32xf32> -> vector<64x32xf32>
          xegpu.compile_hint
          %26 = arith.addf %arg5, %25 {map = #xetile.wg_map<sg_layout = [2, 1], sg_data = [32, 32]>} : vector<64x32xf32>
          scf.yield %26, %21, %20 : vector<64x32xf32>, !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>, !xetile.tile<32x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [1, 2], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        }
        %19 = xetile.init_tile %arg2[%12, %14, %15, %8, %9] : memref<2x3x3x128x64xf32> -> !xetile.tile<64x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
        xetile.store_tile %18#0,  %19 : vector<64x32xf32>, !xetile.tile<64x32xf32, #xetile.tile_attr<wg_map = <sg_layout = [2, 1], sg_data = [32, 32]>, inner_blocks = [], memory_space = 0 : i32, scattered = false>>
      }
      gpu.return
    }
  }
}
