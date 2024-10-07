// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s

module @test_module attributes {gpu.container_module} {
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<24x32xf16>, %B: memref<24x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %a_tile = xetile.init_tile %A[%c0, %c0] : memref<24x32xf16> -> !xetile.tile<24x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      %b_tile = xetile.init_tile %B[%c0, %c0] : memref<24x32xf16> -> !xetile.tile<24x32xf16, #xetile.tile_attr<inner_blocks = [6, 16]>>
      %a_value = xetile.load_tile %a_tile {padding = 0.000000e+00 : f32}  : !xetile.tile<24x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>> -> vector<3x1x8x32xf16>
      %b_value = xetile.load_tile %b_tile {padding = 0.000000e+00 : f32}  : !xetile.tile<24x32xf16, #xetile.tile_attr<inner_blocks = [6, 16]>> -> vector<4x2x6x16xf16>

      %a_valuee = xetile.tile_unpack %a_value {inner_blocks = array<i64: 8, 32>}: vector<3x1x8x32xf16> -> vector<24x32xf16>
      %b_valuee = xetile.tile_unpack %b_value {inner_blocks = array<i64: 6, 16>}  : vector<4x2x6x16xf16> -> vector<24x32xf16>

      %c_value = arith.addf %a_valuee, %b_valuee : vector<24x32xf16>

      // not a power-of-two (size is 6) inner block stacking reduction
      // CHECK: %[[STACK1:.*]] = vector.shuffle {{.*}}, {{.*}} [0, 1] : vector<1x32xf16>, vector<1x32xf16>
      // CHECK: %[[STACK2:.*]] = vector.shuffle {{.*}}, {{.*}} [0, 1] : vector<1x32xf16>, vector<1x32xf16>
      // CHECK: %[[STACK3:.*]] = vector.shuffle {{.*}}, {{.*}} [0, 1] : vector<1x32xf16>, vector<1x32xf16>
      // CHECK: %[[STACK4:.*]] = vector.shuffle %[[STACK1]], %[[STACK2]] [0, 1, 2, 3] : vector<2x32xf16>, vector<2x32xf16>
      // CHECK: %[[STACK5:.*]] = vector.shuffle %[[STACK4]], %[[STACK3]] [0, 1, 2, 3, 4, 5] : vector<4x32xf16>, vector<2x32xf16>

      %c_valuee = xetile.tile_pack %c_value {inner_blocks = array<i64: 6, 16>}  : vector<24x32xf16> -> vector<4x2x6x16xf16>
      xetile.store_tile %c_valuee, %b_tile  : vector<4x2x6x16xf16>, !xetile.tile<24x32xf16, #xetile.tile_attr<inner_blocks = [6, 16]>>

      gpu.return
    }
  }
}
