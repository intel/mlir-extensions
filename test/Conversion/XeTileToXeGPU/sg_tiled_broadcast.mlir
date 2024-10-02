// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  // CHECK-LABEL: @sglevel_broadcast_test_1
  gpu.func @sglevel_broadcast_test_1(%arg0: memref<1024x1024xf16>) {
    // CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<1x16xf16>
    %cst = arith.constant dense<0.000000e+00> : vector<1x4x1x16xf16>
    %0 = xetile.tile_unpack %cst {inner_blocks = array<i64: 1, 16>}: vector<1x4x1x16xf16> -> vector<1x64xf16>
    %1 = xetile.tile_pack %0 {inner_blocks = array<i64: 1, 16>}: vector<1x64xf16> -> vector<1x4x1x16xf16>
    %2 = xetile.broadcast %1 [0, 2] : vector<1x4x1x16xf16> -> vector<32x4x1x16xf16>
    %3 = xetile.tile_unpack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x4x1x16xf16> -> vector<32x64xf16>
    %4 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
    %5 = xetile.tile_pack %3 {inner_blocks = array<i64: 1, 16>}: vector<32x64xf16> -> vector<32x4x1x16xf16>
    // CHECK-COUNT-128: xegpu.store_nd %[[cst]], %{{.*}} <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x16xf16>, !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    xetile.store_tile %5,  %4 : vector<32x4x1x16xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
    gpu.return
  }

  // CHECK-LABEL: @sglevel_broadcast_test_2
  gpu.func @sglevel_broadcast_test_2(%arg0: memref<1024x1024xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<32x1xf16>
    %0 = xetile.tile_pack %cst {inner_blocks = array<i64: 1, 1>}  : vector<32x1xf16> -> vector<32x1x1x1xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    //CHECK: vector.extract %{{.*}}[0, 0] : f16 from vector<1x1xf16>
    //CHECK: vector.splat %{{.*}} : vector<1x16xf16>
    %1 = xetile.broadcast %0 [1, 3] : vector<32x1x1x1xf16> -> vector<32x4x1x16xf16>
    %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 1, 16>}: vector<32x4x1x16xf16> -> vector<32x64xf16>
    %3 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
    %4 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x64xf16> -> vector<32x4x1x16xf16>
    // CHECK-COUNT-128: xegpu.store_nd %{{.*}}, %{{.*}} <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x16xf16>, !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    xetile.store_tile %4,  %3 : vector<32x4x1x16xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
    gpu.return
  }

}
