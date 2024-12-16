// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking="enable-2d-transform=true" \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_load_tile(%[[arg0:.*]]: memref<1024x1024xf16>, %[[arg1:.*]]: memref<1024x1024xf16>, %[[arg2:.*]]: memref<1024x1024xf32>) {
  gpu.func @sg_load_tile(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    %c64 = arith.constant 64 : index
    //CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c64]]]
    //CHECK-SAME: memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<{{.*}}array_length = 1 : i64, boundary_check = true>>
  	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
    //CHECK-SAME: !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<{{.*}}array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
    %2 = xetile.load_tile %1 {padding = 0 : i32} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_cache_attr(%[[arg0:.*]]: memref<1024x1024xf16>, %[[arg1:.*]]: memref<1024x1024xf16>, %[[arg2:.*]]: memref<1024x1024xf32>) {
  gpu.func @sg_load_tile_cache_attr(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    %c64 = arith.constant 64 : index
    //CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c64]]]
    //CHECK-SAME: memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<{{.*}}array_length = 1 : i64, boundary_check = true>>
    %1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
    //CHECK-SAME: !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<{{.*}}array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
    %2 = xetile.load_tile %1 {l1_hint = #xetile.cache_hint<uncached>, l3_hint = #xetile.cache_hint<streaming>} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    gpu.return
  }
}
