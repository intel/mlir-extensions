// RUN: imex-opt --split-input-file --xetile-blocking %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {

// CHECK-LABEL: test_blocking_elementwise
//  CHECK-SAME: (%[[A_ORIG:.*]]: vector<64x64xf16>, %[[B_ORIG:.*]]: vector<64x64xf16>)
//       CHECK: %[[A1:.*]] = xetile.tile_pack %[[A_ORIG]] {inner_blocks = array<i64: 1, 32>} : vector<64x64xf16> -> vector<64x2x1x32xf16>
//       CHECK: %[[B1:.*]] = xetile.tile_pack %[[B_ORIG]] {inner_blocks = array<i64: 1, 32>} : vector<64x64xf16> -> vector<64x2x1x32xf16>
//       CHECK: %[[RES1:.*]] = arith.addf %[[A1]], %[[B1]] : vector<64x2x1x32xf16>
//       CHECK: %[[RES_UNP1:.*]] = xetile.tile_unpack %[[RES1]] {inner_blocks = array<i64: 1, 32>} : vector<64x2x1x32xf16> -> vector<64x64xf16>
//       CHECK: %[[A2:.*]] = xetile.tile_pack %[[A_ORIG]] {inner_blocks = array<i64: 1, 32>} : vector<64x64xf16> -> vector<64x2x1x32xf16>
//       CHECK: %[[RES2:.*]] = arith.negf %[[A2]] : vector<64x2x1x32xf16>
//       CHECK: %[[RES_UNP2:.*]] = xetile.tile_unpack %[[RES2]] {inner_blocks = array<i64: 1, 32>} : vector<64x2x1x32xf16> -> vector<64x64xf16>
//       CHECK: %[[A3:.*]] = xetile.tile_pack %[[A_ORIG]] {inner_blocks = array<i64: 1, 32>} : vector<64x64xf16> -> vector<64x2x1x32xf16>
//       CHECK: %[[RES3:.*]] = math.exp %[[A3]] : vector<64x2x1x32xf16>
//       CHECK: %[[RES_UNP3:.*]] = xetile.tile_unpack %[[RES3]] {inner_blocks = array<i64: 1, 32>} : vector<64x2x1x32xf16> -> vector<64x64xf16>
//       CHECK: return %[[RES_UNP1]], %[[RES_UNP2]], %[[RES_UNP3]] : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>
func.func @test_blocking_elementwise(%a: vector<64x64xf16>, %b: vector<64x64xf16>) -> (vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>) {
// Elementwise arith ops are handled in unified way, check some
  %0 = arith.addf %a, %b: vector<64x64xf16>
  %1 = arith.negf %a: vector<64x64xf16>
  %2 = math.exp %a: vector<64x64xf16>
  return %0, %1, %2 : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>
}

}

// -----

gpu.module @test_kernel {

// CHECK-LABEL: test_blocking_transpose
//  CHECK-SAME: (%[[SRC:.*]]: vector<64x32xf16>)
//       CHECK: %[[PACK:.*]] = xetile.tile_pack %[[SRC]] {inner_blocks = array<i64: 8, 16>} : vector<64x32xf16> -> vector<8x2x8x16xf16>
//       CHECK: %[[T:.*]] = vector.transpose %[[PACK]], [1, 0, 3, 2] : vector<8x2x8x16xf16> to vector<2x8x16x8xf16>
//       CHECK: %[[UNPACK:.*]] = xetile.tile_unpack %[[T]] {inner_blocks = array<i64: 16, 8>} : vector<2x8x16x8xf16> -> vector<32x64xf16>
//       CHECK: return %[[UNPACK]] : vector<32x64xf16>
func.func @test_blocking_transpose(%a: vector<64x32xf16>) -> vector<32x64xf16> {
  %0 = vector.transpose %a, [1, 0]: vector<64x32xf16> to vector<32x64xf16>
  return %0 : vector<32x64xf16>
}

}

// -----

gpu.module @test_kernel {

// CHECK-LABEL: test_blocking_transpose_small
//  CHECK-SAME: (%[[SRC:.*]]: vector<16x8xf16>)
//       CHECK: %[[PACK:.*]] = xetile.tile_pack %[[SRC]] {inner_blocks = array<i64: 8, 8>} : vector<16x8xf16> -> vector<2x1x8x8xf16>
//       CHECK: %[[T:.*]] = vector.transpose %[[PACK]], [1, 0, 3, 2] : vector<2x1x8x8xf16> to vector<1x2x8x8xf16>
//       CHECK: %[[UNPACK:.*]] = xetile.tile_unpack %[[T]] {inner_blocks = array<i64: 8, 8>} : vector<1x2x8x8xf16> -> vector<8x16xf16>
//       CHECK: return %[[UNPACK]] : vector<8x16xf16>
func.func @test_blocking_transpose_small(%a: vector<16x8xf16>) -> vector<8x16xf16> {
  %0 = vector.transpose %a, [1, 0]: vector<16x8xf16> to vector<8x16xf16>
  return %0 : vector<8x16xf16>
}

}
