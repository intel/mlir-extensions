// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
    //CHECK-LABEL: @sglevel_softmax_dim_0
    //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
    gpu.func @sglevel_softmax_dim_0(%a: memref<1024x1024xf16>) {
      //CHECK: %[[c24:.*]] = arith.constant 24 : index
      //CHECK: %[[c16:.*]] = arith.constant 16 : index
      //CHECK: %[[c8:.*]] = arith.constant 8 : index
      //CHECK: %[[c32:.*]] = arith.constant 32 : index
      //CHECK: %[[c0:.*]] = arith.constant 0 : index

      //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>
      //CHECK: %[[r1:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>
      %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

      //CHECK: %[[r2:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>> -> vector<32x32xf16>
      //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>> -> vector<32x32xf16>
      %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>

      //CHECK-COUNT-8: {{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
      //CHECK-COUNT-8: {{.*}} = math.exp %{{.*}} : vector<8x32xf16>
      %3 = math.exp %2: vector<32x64xf16>
      //CHECK-COUNT-62: arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      %4 = xetile.reduction <add>, %3 [0]: vector<32x64xf16> -> vector<1x64xf16>
      %5 = xetile.broadcast %4 [0]: vector<1x64xf16> -> vector<32x64xf16>
      //CHECK-COUNT-8: arith.divf {{.*}}, {{.*}} : vector<8x32xf16>
      %6 = arith.divf %3, %5: vector<32x64xf16>
      %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
      xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
      gpu.return
    }
    //CHECK-LABEL: @sglevel_softmax_dim_1
    //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
    gpu.func @sglevel_softmax_dim_1(%a: memref<1024x1024xf16>) {
      //CHECK: %[[c32:.*]] = arith.constant 32 : index
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>
      //CHECK: %[[r1:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>
      %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
      //CHECK: %[[r2:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>> -> vector<32x32xf16>
      //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>> -> vector<32x32xf16>
      %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
      //CHECK-COUNT-8: {{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
      //CHECK-COUNT-8: {{.*}} = math.exp %{{.*}} : vector<8x32xf16>
      %3 = math.exp %2: vector<32x64xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle {{.*}}, {{.*}} [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<32xf16>
      //CHECK-COUNT-32: {{.*}} = vector.extractelement {{.*}}[{{.*}} : index] : vector<32xf16>
      //CHECK-COUNT-32: {{.*}} = vector.splat {{.*}} : vector<1x32xf16>
      %4 = xetile.reduction <add>, %3 [1]: vector<32x64xf16> -> vector<32x1xf16>

      //CHECK-COUNT-64: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x32xf16> into vector<32x64xf16>
      //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x64xf16> to vector<8x32xf16>

      %5 = xetile.broadcast %4 [1]: vector<32x1xf16> -> vector<32x64xf16>
      // CHECK-COUNT-8: {{.*}} = arith.divf {{.*}}, {{.*}} : vector<8x32xf16>
      %6 = arith.divf %3, %5: vector<32x64xf16>
      %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
      xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
      gpu.return
    }
}
