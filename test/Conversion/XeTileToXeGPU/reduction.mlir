// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
module {
  gpu.module @test_kernel {

    //CHECK: gpu.func @inner_reduction(%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>) {
    gpu.func @inner_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
      //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<16x32xf16>
      //CHECK-COUNT-16: {{.*}} = vector.extract_strided_slice %[[r1]] {offsets = {{.*}}, sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
      //CHECK-COUNT-16: {{.*}} = math.exp %{{.*}} : vector<1x32xf16>
      %e = math.exp %v: vector<16x32xf16>
      //CHECK-COUNT-16: {{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] : vector<32xf16>, vector<32xf16>
      //CHECK: {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<16xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<16xf16>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf16>
      //CHECK-COUNT-8: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1] : vector<1x1xf16>, vector<1x1xf16>
      //CHECK-COUNT-4: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
      //CHECK-COUNT-2: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
      %r = xetile.reduction <add>, %e[1] : vector<16x32xf16> -> vector<16x1xf16>
      //CHECK: %[[r177:.*]] = vector.shape_cast {{.*}} : vector<16x1xf16> to vector<2x8xf16>
      %c = vector.shape_cast %r: vector<16x1xf16> to vector<2x8xf16>
      //CHECK: %[[r178:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xegpu.tensor_desc<2x8xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<2x8xf16>
      //CHECK: xegpu.store_nd %[[r177]], %[[r178]] <{{.*}}> : vector<2x8xf16>, !xegpu.tensor_desc<2x8xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      xetile.store_tile %c, %s : vector<2x8xf16>, !xetile.tile<2x8xf16>
      gpu.return
    }

    gpu.func @inner_reduction_1(%a: memref<8x32xf32>, %b: memref<8x1xf32>) {
      %c0 = arith.constant 0 : index
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[c16:.*]] = arith.constant 16 : index
      //CHECK: %[[r1:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c16]]] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %a_tile = xetile.init_tile %a[%c0, %c0] : memref<8x32xf32> -> !xetile.tile<8x32xf32>
      //CHECK: %[[r2:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[c0]]] : memref<8x1xf32> -> !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %b_tile = xetile.init_tile %b[%c0, %c0] : memref<8x1xf32> -> !xetile.tile<8x1xf32>
      //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16xf32>
      //CHECK: %[[r4:.*]] = xegpu.load_nd %[[r1]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16xf32>
      //CHECK-COUNT-8: {{.*}} = vector.extract_strided_slice %[[r3]] {offsets = {{.*}}, sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
      //CHECK-COUNT-8: {{.*}} = vector.extract_strided_slice %[[r4]] {offsets = {{.*}}, sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
      %a_loaded = xetile.load_tile %a_tile: !xetile.tile<8x32xf32> -> vector<8x32xf32>

      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<1x16xf32>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<1x16xf32>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<1x16xf32>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<1x16xf32>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<1x16xf32>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<1x16xf32>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<1x16xf32>
      //CHECK: {{.*}} = vector.shape_cast {{.*}} : vector<1x16xf32> to vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xf32>, vector<16xf32>
      //CHECK: {{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<8xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK: {{.*}} = arith.constant {{.*}} : i32
      //CHECK: {{.*}} = vector.extractelement %{{.*}}[{{.*}} : i32] : vector<8xf32>
      //CHECK: {{.*}} = vector.splat %{{.*}} : vector<1x1xf32>
      //CHECK-COUNT-4: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1] : vector<1x1xf32>, vector<1x1xf32>
      //CHECK-COUNT-2: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3] : vector<2x1xf32>, vector<2x1xf32>
      //CHECK: {{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf32>, vector<4x1xf32>
      %3 = xetile.reduction <maximumf>, %a_loaded[1] : vector<8x32xf32> -> vector<8x1xf32> // fastmath<nnan> is implicit here
      //CHECK: xegpu.store_nd {{.*}} : vector<8x1xf32>, !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      xetile.store_tile %3, %b_tile : vector<8x1xf32>, !xetile.tile<8x1xf32>
      gpu.return
    }

    //CHECK: gpu.func @outter_reduction(%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>) {
    gpu.func @outter_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
      //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] <{{.*}}> : !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<16x32xf16>
      //CHECK-COUNT-16: {{.*}} = vector.extract_strided_slice %[[r1]] {offsets = {{.*}}, sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
      //CHECK-COUNT-16: {{.*}} = math.exp {{.*}} : vector<1x32xf16>
      %e = math.exp %v: vector<16x32xf16>
      //CHECK: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<1x32xf16>
      %r = xetile.reduction <add>, %e[0] : vector<16x32xf16> -> vector<1x32xf16>
      //CHECK: %[[r49:.*]] = vector.shape_cast {{.*}} : vector<1x32xf16> to vector<4x8xf16>
      %c = vector.shape_cast %r: vector<1x32xf16> to vector<4x8xf16>
      //CHECK: %[[r50:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xegpu.tensor_desc<4x8xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<4x8xf16>
      //CHECK: xegpu.store_nd %[[r49]], %[[r50]] <{{.*}}> : vector<4x8xf16>, !xegpu.tensor_desc<4x8xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      xetile.store_tile %c, %s : vector<4x8xf16>, !xetile.tile<4x8xf16>
      gpu.return
    }
  }
}
