// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK-LABEL: gpu.func @sg_store_tile
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x32xf32>) {
  gpu.func @sg_store_tile(%arg0: memref<32x32xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    //CHECK: xegpu.store_nd %[[cst]], %[[r0]] {{.*}}: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    %0 = xetile.init_tile %arg0[0, 0] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    xetile.store_tile %cst,  %0 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }

  //-----

  // CHECK: gpu.func @sg_tile_mma(%[[arg0:.*]]: memref<8x16xf16>, %[[arg1:.*]]: memref<16x16xf16>, %[[arg2:.*]]: memref<8x16xf32>)
  gpu.func @sg_tile_mma(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] {{.*}}: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    //CHECK: %[[r2:.*]] = xegpu.create_nd_tdesc %[[arg1]][0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
    //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r2]] {{.*}}: !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    //CHECK: %[[r4:.*]] = xegpu.dpas %[[r1]], %[[r3]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r5:.*]] = xegpu.create_nd_tdesc %[[arg2]][0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
    //CHECK: xegpu.store_nd %[[r4]], %[[r5]] {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
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
    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, 0] : memref<2x64xf16> -> !xegpu.tensor_desc<2x16xf16>
    //CHECK: xegpu.prefetch_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<2x16xf16>
    //CHECK: xegpu.prefetch_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}> : !xegpu.tensor_desc<2x16xf16>
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
    //CHECK: %[[r1:.*]] = xegpu.create_tdesc %[[arg0]], %[[r0]] : memref<1024xf16>, vector<16xindex> -> !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<>>
    //CHECK: %[[r2:.*]] = vector.shape_cast %[[cst]] : vector<1x16xi1> to vector<16xi1>
    //CHECK: %[[r3:.*]] = xegpu.load %[[r1]], %[[r2]] {{.*}} : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf16>
    //CHECK: %[[r4:.*]] = vector.shape_cast %[[r3]] : vector<16xf16> to vector<1x16xf16>
    //CHECK: %[[r5:.*]] = vector.shape_cast %[[cst_1]] : vector<1x16xindex> to vector<16xindex>
    //CHECK: %[[r6:.*]] = xegpu.update_offset %[[r1]], %[[r5]] : !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<>>, vector<16xindex>
    //CHECK: %[[r7:.*]] = vector.shape_cast %[[cst]] : vector<1x16xi1> to vector<16xi1>
    //CHECK: %[[r8:.*]] = vector.shape_cast %[[r4]] : vector<1x16xf16> to vector<16xf16>
    //CHECK: xegpu.store %[[r8]], %[[r6]], %[[r7]] {{.*}} : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
    %mask = arith.constant dense<true> : vector<1x16xi1>
    %idx = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : vector<1x16xindex>
    %offsets = arith.constant dense<16> : vector<1x16xindex>
    %0 = xetile.init_tile %arg0, %idx : memref<1024xf16>, vector<1x16xindex> -> !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>
    %1 = xetile.load %0, %mask : !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf16>
    %2 = xetile.update_tile_offset %0, %offsets : !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>, vector<1x16xindex>
    xetile.store %1, %2, %mask : vector<1x16xf16>, !xetile.tile<1x16xf16, #xetile.tile_attr<scattered = true>>, vector<1x16xi1>
    gpu.return
  }

  //-----
  //CHECK-LABEL: gpu.func @copy_via_slm
  //CHECK-SAME: (%[[arg0:.*]]: memref<64x64xf16>, %[[arg1:.*]]: memref<64x64xf16>)
  gpu.func @copy_via_slm(%A: memref<64x64xf16>, %B: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    //CHECK: %[[thread_id_x:.*]] = gpu.thread_id  x
    //CHECK: %[[thread_id_y:.*]] = gpu.thread_id  y
    %tid_x = gpu.thread_id x
    %tid_y = gpu.thread_id y

    //CHECK: %[[r0:.*]] = arith.muli %[[thread_id_x]], %[[c8]] : index
    //CHECK: %[[r1:.*]] = arith.muli %[[thread_id_y]], %[[c16]] : index
    %m = arith.muli %tid_x, %c8 : index
    %n = arith.muli %tid_y, %c16 : index

    //CHECK: %[[r2:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r0]], %[[r1]]] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r2]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %a_tile = xetile.init_tile %A[%m, %n] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
    %a = xetile.load_tile %a_tile : !xetile.tile<8x16xf16> -> vector<8x16xf16>

    //CHECK: %[[alloc:.*]] = memref.alloc() {alignment = 32 : i64} : memref<2048xf32, 3>
    %slm = memref.alloc() : memref<8192xi8, 3>
    %view = memref.view %slm[%c0][] : memref<8192xi8, 3> to memref<64x64xf16, 3>

    //CHECK: %[[r13:.*]] = xegpu.create_nd_tdesc %[[alloc]][%{{.*}}] : memref<2048xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
    %st_tile = xetile.init_tile %view[%m, %n] : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space=3>>

    //CHECK: %[[r14:.*]] = vector.shape_cast %[[r3]] : vector<8x16xf16> to vector<128xf16>
    //CHECK: %[[r15:.*]] = vector.bitcast %[[r14]] : vector<128xf16> to vector<64xf32>
    //CHECK: xegpu.store_nd %[[r15]], %[[r13]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
    xetile.store_tile %a, %st_tile : vector<8x16xf16>, !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space=3>>

    //CHECK: %[[r25:.*]] = xegpu.create_nd_tdesc %[[alloc]][%{{.*}}] : memref<2048xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
    %ld_tile = xetile.init_tile %view[%m, %n] : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space=3>>

    //CHECK: %[[r26:.*]] = xegpu.load_nd %[[r25]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>> -> vector<64xf32>
    //CHECK: %[[r27:.*]] = vector.bitcast %[[r26]] : vector<64xf32> to vector<128xf16>
    //CHECK: %[[r28:.*]] = vector.shape_cast %[[r27]] : vector<128xf16> to vector<8x16xf16>
    %d = xetile.load_tile %ld_tile : !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space=3>> -> vector<8x16xf16>

    %b_tile = xetile.init_tile %B[%m, %n] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
    xetile.store_tile %d, %b_tile: vector<8x16xf16>, !xetile.tile<8x16xf16>
    gpu.return
  }

  //-----
  //CHECK: gpu.func @transpose_via_slm(%[[arg0:.*]]: memref<64x64xf16>, %[[arg1:.*]]: memref<64x64xf16>)
  gpu.func @transpose_via_slm(%A: memref<64x64xf16>, %B: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    %tid_x = gpu.thread_id x
    %tid_y = gpu.thread_id y
    %m = arith.muli %tid_x, %c8 : index
    %n = arith.muli %tid_y, %c16 : index

    %a_tile = xetile.init_tile %A[%m, %n] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
    //CHECK: %[[r3:.*]] = xegpu.load_nd {{.*}}: !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
    %a = xetile.load_tile %a_tile : !xetile.tile<8x16xf16> -> vector<8x16xf16>

    //CHECK: %[[alloc:.*]] = memref.alloc() {alignment = 32 : i64} : memref<2048xf32, 3>
    %slm = memref.alloc() : memref<8192xi8, 3>
    %view = memref.view %slm[%c0][] : memref<8192xi8, 3> to memref<64x64xf16, 3>
    %trans_view = memref.transpose %view (i, j) -> (j, i) : memref<64x64xf16, 3> to memref<64x64xf16, strided<[1, 64], offset:0>, 3>

    //CHECK: %[[r15:.*]] = xegpu.create_tdesc %[[alloc]], %{{.*}} : memref<2048xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 4 : i64>>
    %st_tile = xetile.init_tile %trans_view[%m, %n] : memref<64x64xf16, strided<[1, 64], offset:0>, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<order=[0, 1], memory_space=3>>
    //CHECK: %[[r19:.*]] = vector.shape_cast %[[r3]] : vector<4x16x2xf16> to vector<4x32xf16>
    //CHECK: %[[r20:.*]] = vector.bitcast %[[r19]] : vector<4x32xf16> to vector<4x16xf32>
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
    //CHECK: %[[r21:.*]] = vector.transpose %[[r20]], [1, 0] : vector<4x16xf32> to vector<16x4xf32>
    //CHECK: xegpu.store %[[r21]], %[[r15]], %[[cst]] {{.*}} : vector<16x4xf32>, !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 4 : i64>>, vector<16xi1>
    xetile.store_tile %a, %st_tile : vector<8x16xf16>, !xetile.tile<8x16xf16, #xetile.tile_attr<order=[0, 1], memory_space=3>>

    //CHECK: %[[r30:.*]] = xegpu.create_nd_tdesc %[[alloc]][%{{.*}}] : memref<2048xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
    %ld_tile = xetile.init_tile %view[%m, %n] : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space=3>>

    //CHECK: %[[r31:.*]] = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>> -> vector<64xf32>
    //CHECK: %[[r32:.*]] = vector.bitcast %[[r31]] : vector<64xf32> to vector<128xf16>
    //CHECK: %[[r33:.*]] = vector.shape_cast %[[r32]] : vector<128xf16> to vector<8x16xf16>
    %d = xetile.load_tile %ld_tile : !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space=3>> -> vector<8x16xf16>

    %b_tile = xetile.init_tile %B[%m, %n] : memref<64x64xf16> -> !xetile.tile<8x16xf16>
    xetile.store_tile %d, %b_tile: vector<8x16xf16>, !xetile.tile<8x16xf16>
    gpu.return
  }

}
