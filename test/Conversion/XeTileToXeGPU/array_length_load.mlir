// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {

  //CHECK: gpu.func @sg_load_tile(%[[arg0:.*]]: memref<32x32xf16>, %[[arg1:.*]]: memref<1x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_load_tile(%a: memref<32x32xf16>, %b: memref<1x32xf16>, %c: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index

    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%{{.*}}, %{{.*}}] : memref<32x32xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    //CHECK: %[[r2:.*]] = vector.extract %[[r1]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r3:.*]] = vector.extract %[[r1]][1] : vector<32x16xf16> from vector<2x32x16xf16>

  	%a_tile = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    %a_loaded = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

    // Do not let XeGPU do one load with multiple blocks (array_length > 1), where each block is finer than one GRF.
  	%b_tile = xetile.init_tile %b[%c0, %c0] : memref<1x32xf16> -> !xetile.tile<1x32xf16>
    %b_loaded = xetile.load_tile %b_tile : !xetile.tile<1x32xf16> -> vector<1x32xf16>

    %b_loaded_T = xetile.transpose %b_loaded, [1, 0] : vector<1x32xf16> -> vector<32x1xf16>
    xegpu.compile_hint
    %b_loaded_T_bc = xetile.broadcast %b_loaded_T [1] : vector<32x1xf16> -> vector<32x32xf16>

    %ab = xetile.tile_mma %a_loaded, %b_loaded_T_bc : vector<32x32xf16>, vector<32x32xf16> -> vector<32x32xf16>
    %ab_f32 = arith.extf %ab : vector<32x32xf16> to vector<32x32xf32>

  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
    xetile.store_tile %ab_f32,  %c_tile : vector<32x32xf32>, !xetile.tile<32x32xf32>

  	gpu.return
  }
}
