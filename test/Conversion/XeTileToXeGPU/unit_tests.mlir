// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK-LABEL: gpu.func @sg_store_tile
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x32xf32>) {
  gpu.func @sg_store_tile(%arg0: memref<32x32xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[cst]], %[[r0]] {{.*}}: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    xetile.store_tile %cst,  %0 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }

  //-----

  // CHECK: gpu.func @sg_tile_mma(%[[arg0:.*]]: memref<8x16xf16>, %[[arg1:.*]]: memref<16x16xf16>, %[[arg2:.*]]: memref<8x16xf32>)
  gpu.func @sg_tile_mma(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] {{.*}}: !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<8x16xf16>
    //CHECK: %[[r2:.*]] = xegpu.create_nd_tdesc %[[arg1]][0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r2]] {{.*}}: !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf16>
    //CHECK: %[[r4:.*]] = xegpu.dpas %[[r1]], %[[r3]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r5:.*]] = xegpu.create_nd_tdesc %[[arg2]][0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r4]], %[[r5]] {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %0 = xetile.init_tile %arg0[0, 0] : memref<8x16xf16> -> !xetile.tile<8x16xf16>
    %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<8x16xf16> -> vector<8x16xf16>
    %2 = xetile.init_tile %arg1[0, 0] : memref<16x16xf16> -> !xetile.tile<16x16xf16>
    %3 = xetile.load_tile %2 {padding = 0.000000e+00 : f32}  : !xetile.tile<16x16xf16> -> vector<16x16xf16>
    %4 = xetile.tile_mma %1, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %5 = xetile.init_tile %arg2[0, 0] : memref<8x16xf32> -> !xetile.tile<8x16xf32>
    xetile.store_tile %4,  %5 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }

  //-----
  //CHECK: gpu.func @sg_prefetch_tile(%[[arg0:.*]]: memref<2x64xf16>)
  gpu.func @sg_prefetch_tile(%a: memref<2x64xf16>) {
    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<2x64xf16> -> !xegpu.tensor_desc<2x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.prefetch_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<2x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.prefetch_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}> : !xegpu.tensor_desc<2x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
    %0 = xetile.init_tile %a[0, 0] : memref<2x64xf16> -> !xetile.tile<2x16xf16>
    xetile.prefetch_tile %0 : !xetile.tile<2x16xf16>
    xetile.prefetch_tile %0 {l1_hint = #xetile.cache_hint<uncached>, l3_hint = #xetile.cache_hint<streaming>} : !xetile.tile<2x16xf16>
    gpu.return
  }

  //-----

  //CHECK: gpu.func @sg_scattered_ops(%[[arg0:.*]]: memref<1024xf16>)
  gpu.func @sg_scattered_ops(%arg0: memref<1024xf16>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<1x16xi1>
    //CHECK: %[[cst_0:.*]] = arith.constant dense<{{.*}}> : vector<1x16xindex>
    //CHECK: %[[cst_1:.*]] = arith.constant dense<16> : vector<1x16xindex>
    //CHECK: %[[r0:.*]] = vector.shape_cast %[[cst_0]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r1:.*]] = xegpu.create_tdesc %[[arg0]], %[[r0]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>
    //CHECK: %[[r2:.*]] = vector.shape_cast %[[cst]] : vector<1x16xi1> to vector<16xi1>
    //CHECK: %[[r3:.*]] = xegpu.load %[[r1]], %[[r2]] {{.*}} : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r4:.*]] = vector.shape_cast %[[r3]] : vector<16xf16> to vector<1x16xf16>
    //CHECK: %[[r5:.*]] = vector.shape_cast %[[cst_1]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r6:.*]] = xegpu.update_offset %[[r1]], %[[r5]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xindex>
    //CHECK: %[[r7:.*]] = vector.shape_cast %[[cst]] : vector<1x16xi1> to vector<16xi1>
    //CHECK: %[[r8:.*]] = vector.shape_cast %[[r4]] : vector<1x16xf16> to vector<16xf16>
    //CHECK: xegpu.store %[[r8]], %[[r6]], %[[r7]] {{.*}} : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1>
    %mask = arith.constant dense<true> : vector<1x16xi1>
    %idx = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : vector<1x16xindex>
    %offsets = arith.constant dense<16> : vector<1x16xindex>
    %0 = xetile.init_tile %arg0, %idx : memref<1024xf16>, vector<1x16xindex> -> !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>
    %1 = xetile.load %0, %mask : !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf16>
    %2 = xetile.update_tile_offset %0, %offsets : !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>, vector<1x16xindex>
    xetile.store %1, %2, %mask : vector<1x16xf16>, !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>, vector<1x16xi1>
    gpu.return
  }
}
