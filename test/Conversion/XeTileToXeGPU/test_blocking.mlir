// RUN: imex-opt --split-input-file --xetile-blocking %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {

// CHECK-LABEL: test_blocking_elementwise
//  CHECK-SAME: (%[[A_ORIG:.*]]: vector<64x64xf16>, %[[B_ORIG:.*]]: vector<64x64xf16>)
//       CHECK: %[[A1:.*]] = xetile.tile_pack %[[A_ORIG]] { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
//       CHECK: %[[B1:.*]] = xetile.tile_pack %[[B_ORIG]] { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
//       CHECK: %[[RES1:.*]] = arith.addf %[[A1]], %[[B1]] : vector<64x4x1x16xf16>
//       CHECK: %[[RES_UNP1:.*]] = xetile.tile_unpack %[[RES1]] { inner_blocks = [1, 16] } : vector<64x4x1x16xf16> -> vector<64x64xf16>
//       CHECK: %[[A2:.*]] = xetile.tile_pack %[[A_ORIG]] { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
//       CHECK: %[[RES2:.*]] = arith.negf %[[A2]] : vector<64x4x1x16xf16>
//       CHECK: %[[RES_UNP2:.*]] = xetile.tile_unpack %[[RES2]] { inner_blocks = [1, 16] } : vector<64x4x1x16xf16> -> vector<64x64xf16>
//       CHECK: %[[A3:.*]] = xetile.tile_pack %[[A_ORIG]] { inner_blocks = [1, 16] } : vector<64x64xf16> -> vector<64x4x1x16xf16>
//       CHECK: %[[RES3:.*]] = math.exp %[[A3]] : vector<64x4x1x16xf16>
//       CHECK: %[[RES_UNP3:.*]] = xetile.tile_unpack %[[RES3]] { inner_blocks = [1, 16] } : vector<64x4x1x16xf16> -> vector<64x64xf16>
//       CHECK: return %[[RES_UNP1]], %[[RES_UNP2]], %[[RES_UNP3]] : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>
func.func @test_blocking_elementwise(%a: vector<64x64xf16>, %b: vector<64x64xf16>) -> (vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>) {
// Elementwise arith ops are handled in unified way, check some
  %0 = arith.addf %a, %b: vector<64x64xf16>
  %1 = arith.negf %a: vector<64x64xf16>
  %2 = math.exp %a: vector<64x64xf16>
  return %0, %1, %2 : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>
}

}
