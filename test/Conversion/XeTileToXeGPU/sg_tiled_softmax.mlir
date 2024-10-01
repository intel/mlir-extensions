// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
    //CHECK-LABEL: @sglevel_softmax_dim_0
    //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
    gpu.func @sglevel_softmax_dim_0(%arg0: memref<1024x1024xf16>) {
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[c32:.*]] = arith.constant 32 : index
      //CHECK: %[[r1:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %0 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>

      //CHECK: %[[r2:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
      //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
      %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>

      //CHECK-COUNT-32: {{.*}} = vector.extract_strided_slice %[[r2]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
      //CHECK-COUNT-32: {{.*}} = vector.extract_strided_slice %[[r3]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
      //CHECK-COUNT-128: {{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x64xf16> -> vector<32x4x1x16xf16>
      //CHECK-COUNT-128: {{.*}} = math.exp %{{.*}} : vector<1x16xf16>
      %4 = math.exp %3 : vector<32x4x1x16xf16>
      //CHECK-COUNT-124: arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      %5 = xetile.reduction <add>, %4 [0, 2] : vector<32x4x1x16xf16> -> vector<1x4x1x16xf16>
      %6 = xetile.broadcast %5 [0, 2] : vector<1x4x1x16xf16> -> vector<32x4x1x16xf16>
      //CHECK-COUNT-128: arith.divf {{.*}}, {{.*}} : vector<1x16xf16>
      %7 = arith.divf %4, %6 : vector<32x4x1x16xf16>
      %8 = xetile.tile_unpack %7 {inner_blocks = array<i64: 1, 16>}: vector<32x4x1x16xf16> -> vector<32x64xf16>
      %9 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      %10 = xetile.tile_pack %8 {inner_blocks = array<i64: 8, 32>}: vector<32x64xf16> -> vector<4x2x8x32xf16>
      xetile.store_tile %10,  %9 : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      gpu.return
    }
    //CHECK-LABEL: @sglevel_softmax_dim_1
    //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
    gpu.func @sglevel_softmax_dim_1(%arg0: memref<1024x1024xf16>) {
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[c32:.*]] = arith.constant 32 : index
      //CHECK: %[[r1:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %0 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      //CHECK: %[[r2:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
      //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
      %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
      //CHECK-COUNT-32: {{.*}} = vector.extract_strided_slice %[[r2]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
      //CHECK-COUNT-32: {{.*}} = vector.extract_strided_slice %[[r3]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
      //CHECK-COUNT-128: {{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
      %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x64xf16> -> vector<32x4x1x16xf16>
      //CHECK-COUNT-128: {{.*}} = math.exp %{{.*}} : vector<1x16xf16>
      %4 = math.exp %3 : vector<32x4x1x16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>
      //CHECK-COUNT-3: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf16> to vector<16xf16>

      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<16xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = vector.extractelement {{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x1xf16>
      %5 = xetile.reduction <add>, %4 [1, 3] : vector<32x4x1x16xf16> -> vector<32x1x1x1xf16>

      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      //CHECK: {{.*}} = vector.extract {{.*}}[0, 0] : f16 from vector<1x1xf16>
      //CHECK: {{.*}} = vector.splat {{.*}} : vector<1x16xf16>
      %6 = xetile.broadcast %5 [1, 3] : vector<32x1x1x1xf16> -> vector<32x4x1x16xf16>
      // CHECK-COUNT-128: {{.*}} = arith.divf {{.*}}, {{.*}} : vector<1x16xf16>
      %7 = arith.divf %4, %6 : vector<32x4x1x16xf16>
      %8 = xetile.tile_unpack %7 {inner_blocks = array<i64: 1, 16>}: vector<32x4x1x16xf16> -> vector<32x64xf16>
      %9 = xetile.init_tile %arg0[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      %10 = xetile.tile_pack %8 {inner_blocks = array<i64: 8, 32>}: vector<32x64xf16> -> vector<4x2x8x32xf16>
      xetile.store_tile %10,  %9 : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      gpu.return
    }
}
