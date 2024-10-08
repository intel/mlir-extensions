// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu -cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  //CHECK: sg_load_tile(%[[ARG:.*]]: memref<1024x1024xf16>)
  gpu.func @sg_load_tile(%arg0: memref<1024x1024xf16>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    // CHECK: %[[C64:.*]] = arith.constant 64 : index
    %c64 = arith.constant 64 : index

    // CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %arg0[%[[C0]], %[[C64]]] : memref<1024x1024xf16>
    // CHECK-SAME: !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %0 = xetile.init_tile %arg0[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>

    // CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
    // CHECK-SAME: !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x32xf16>
    %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
    gpu.return
  }
}
