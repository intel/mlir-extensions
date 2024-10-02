// RUN: imex-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s


// CHECK-LABEL: func @test_pack_unpack_chain
//  CHECK-SAME:  (%[[SRC:.*]]: vector<32x64xf16>)
//       CHECK:  return %[[SRC]] : vector<32x64xf16>
func.func @test_pack_unpack_chain(%source : vector<32x64xf16>) -> vector<32x64xf16> {
  %1 = xetile.tile_pack %source {inner_blocks = array<i64: 16, 16>} : vector<32x64xf16> -> vector<2x4x16x16xf16>
  %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 16, 16>} : vector<2x4x16x16xf16> ->  vector<32x64xf16>
  return %2 : vector<32x64xf16>
}

// CHECK-LABEL: func @test_unpack_pack_chain
//  CHECK-SAME:  (%[[SRC:.*]]: vector<2x4x16x16xf16>)
//       CHECK:  return %[[SRC]] : vector<2x4x16x16xf16>
func.func @test_unpack_pack_chain(%source : vector<2x4x16x16xf16>) -> vector<2x4x16x16xf16> {
  %1 = xetile.tile_unpack %source {inner_blocks = array<i64: 16, 16>} : vector<2x4x16x16xf16> ->  vector<32x64xf16>
  %2 = xetile.tile_pack %1 {inner_blocks = array<i64: 16, 16>} : vector<32x64xf16> -> vector<2x4x16x16xf16>
  return %2 : vector<2x4x16x16xf16>
}
