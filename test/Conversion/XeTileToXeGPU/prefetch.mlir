// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_prefetch
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  xegpu.prefetch_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<streaming>}>
//       CHECK:  gpu.return
gpu.module @test_kernel {
gpu.func @test_prefetch(%a: memref<2x64xf16>) {
  %c0 = arith.constant 0 : index
  %0 = xetile.init_tile %a[%c0, %c0] : memref<2x64xf16> -> !xetile.tile<2x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
  xetile.prefetch_tile %0 : !xetile.tile<2x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
  xetile.prefetch_tile %0 {l1_hint = #xetile.cache_hint<uncached>, l3_hint = #xetile.cache_hint<streaming>} : !xetile.tile<2x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
  gpu.return
}
}
