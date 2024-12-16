// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking="enable-2d-transform=true" \
// RUN: --cse --convert-xetile-to-xegpu --cse --canonicalize --cse %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test {
  //CHECK-LABEL: @test_init_tile_for_scattered
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>
  gpu.func @test_init_tile_for_scattered(%arg0: memref<1024xf16>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<32xi1>
    //CHECK: %[[cst_0:.*]] = arith.constant dense<1> : vector<32xindex>
    //CHECK: %[[r0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst_0]] : memref<1024xf16>, vector<32xindex> -> !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>
    //CHECK: %[[r1:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r2:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r3:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r4:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r5:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: %[[r6:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: %[[r7:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: %[[r8:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: xegpu.store %[[r1]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    //CHECK: xegpu.store %[[r2]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    //CHECK: xegpu.store %[[r3]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    //CHECK: xegpu.store %[[r4]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    %cst = arith.constant dense<true> : vector<4x32xi1>
    %cst_0 = arith.constant dense<1> : vector<4x32xindex>
    %0 = xetile.init_tile %arg0, %cst_0 : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
    %1 = xetile.load %0, %cst : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1> -> vector<4x32xf16>
    %2 = xetile.update_tile_offset %0, %cst_0 : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xindex>
    xetile.store %1, %0, %cst : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
    gpu.return
  }

  //-----

  //CHECK-LABEL: @test_init_tile_for_scattered_cache_attr
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>
  gpu.func @test_init_tile_for_scattered_cache_attr(%arg0: memref<1024xf16>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<32xi1>
    //CHECK: %[[cst_0:.*]] = arith.constant dense<1> : vector<32xindex>
    //CHECK: %[[r0:.*]] = xegpu.create_tdesc %[[arg0]], %[[cst_0]] : memref<1024xf16>, vector<32xindex> -> !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>
    //CHECK: %[[r1:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r2:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r3:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r4:.*]] = xegpu.load %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}> : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1> -> vector<32xf16>
    //CHECK: %[[r5:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: %[[r6:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: %[[r7:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: %[[r8:.*]] = xegpu.update_offset %[[r0]], %[[cst_0]] : !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xindex>
    //CHECK: xegpu.store %[[r1]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    //CHECK: xegpu.store %[[r2]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    //CHECK: xegpu.store %[[r3]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    //CHECK: xegpu.store %[[r4]], %[[r0]], %[[cst]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<uncached>}> : vector<32xf16>, !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<32xi1>
    %cst = arith.constant dense<true> : vector<4x32xi1>
    %cst_0 = arith.constant dense<1> : vector<4x32xindex>
    %0 = xetile.init_tile %arg0, %cst_0 : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
    %1 = xetile.load %0, %cst {l1_hint = #xetile.cache_hint<uncached>, l3_hint = #xetile.cache_hint<streaming>} : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1> -> vector<4x32xf16>
    %2 = xetile.update_tile_offset %0, %cst_0 : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xindex>
    xetile.store %1, %0, %cst {l1_hint = #xetile.cache_hint<uncached>, l3_hint = #xetile.cache_hint<uncached>} : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
    gpu.return
  }

  //-----

  //CHECK-LABEL: @add_kernel
  //CHECK-SAME: %[[arg0:.*]]: memref<*xf32>, %[[arg1:.*]]: memref<*xf32>, %[[arg2:.*]]: memref<*xf32>
  gpu.func @add_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[cast:.*]] = memref.cast %[[arg0]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_0:.*]] = memref.cast %[[arg1]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_1:.*]] = memref.cast %[[arg2]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
    //CHECK: %[[r0:.*]] = arith.muli %[[block_id_x]], %[[c1024]] : index
    //CHECK: %[[r1:.*]] = vector.splat %[[r0]] : vector<1x16xindex>
    //CHECK: %[[r2:.*]] = vector.shape_cast %[[r1]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r3:.*]] = xegpu.create_tdesc %[[cast]], %[[r2]] : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>
    //CHECK: %[[r4:.*]] = xegpu.load %[[r3]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    //CHECK: %[[r5:.*]] = vector.shape_cast %[[r4]] : vector<16xf32> to vector<1x16xf32>
    //CHECK: %[[r6:.*]] = xegpu.load %[[r3]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    //CHECK: %[[r7:.*]] = vector.shape_cast %[[r6]] : vector<16xf32> to vector<1x16xf32>
    //CHECK: %[[r8:.*]] = xegpu.create_tdesc %[[cast_0]], %[[r2]] : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>
    //CHECK: %[[r9:.*]] = xegpu.load %[[r8]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    //CHECK: %[[r10:.*]] = vector.shape_cast %[[r9]] : vector<16xf32> to vector<1x16xf32>
    //CHECK: %[[r11:.*]] = xegpu.load %[[r8]], %[[cst]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1> -> vector<16xf32>
    //CHECK: %[[r12:.*]] = vector.shape_cast %[[r11]] : vector<16xf32> to vector<1x16xf32>
    //CHECK: %[[r13:.*]] = arith.addf %[[r5]], %[[r10]] : vector<1x16xf32>
    //CHECK: %[[r14:.*]] = arith.addf %[[r7]], %[[r12]] : vector<1x16xf32>
    //CHECK: %[[r15:.*]] = xegpu.create_tdesc %[[cast_1]], %[[r2]] : memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>
    //CHECK: %[[r16:.*]] = vector.shape_cast %[[r13]] : vector<1x16xf32> to vector<16xf32>
    //CHECK: xegpu.store %[[r16]], %[[r15]], %[[cst]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1>
    //CHECK: %[[r17:.*]] = vector.shape_cast %[[r14]] : vector<1x16xf32> to vector<16xf32>
    //CHECK: xegpu.store %[[r17]], %[[r15]], %[[cst]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1 : i64>>, vector<16xi1>
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant dense<true> : vector<1x32xi1>
    %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %block_id_x = gpu.block_id  x
    %0 = arith.muli %block_id_x, %c1024 : index
    %1 = vector.splat %0 : vector<1x32xindex>
    %2 = xetile.init_tile %cast, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    %3 = xetile.load %2, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
    %4 = xetile.init_tile %cast_0, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    %5 = xetile.load %4, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
    %6 = arith.addf %3, %5 : vector<1x32xf32>
    %7 = xetile.init_tile %cast_1, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    xetile.store %6, %7, %cst : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    gpu.return
  }
}
