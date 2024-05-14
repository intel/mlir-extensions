// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_transpose
// Compare original args order with transposed
//       CHECK:  %[[RES1:.*]] = builtin.unrealized_conversion_cast %[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]], %[[ARG4:.*]], %[[ARG5:.*]], %[[ARG6:.*]], %[[ARG7:.*]], %[[ARG8:.*]] :
//   CHECK-DAG:  %[[TARG1:.*]] = vector.transpose %[[ARG1]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//   CHECK-DAG:  %[[TARG2:.*]] = vector.transpose %[[ARG2]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//   CHECK-DAG:  %[[TARG3:.*]] = vector.transpose %[[ARG3]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//   CHECK-DAG:  %[[TARG4:.*]] = vector.transpose %[[ARG4]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//   CHECK-DAG:  %[[TARG5:.*]] = vector.transpose %[[ARG5]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//   CHECK-DAG:  %[[TARG6:.*]] = vector.transpose %[[ARG6]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//   CHECK-DAG:  %[[TARG7:.*]] = vector.transpose %[[ARG7]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//   CHECK-DAG:  %[[TARG8:.*]] = vector.transpose %[[ARG8]], [1, 0] : vector<1x16xf16> to vector<16x1xf16>
//       CHECK:  %[[RES2:.*]] = builtin.unrealized_conversion_cast %[[TARG1]], %[[TARG5]], %[[TARG2]], %[[TARG6]], %[[TARG3]], %[[TARG7]], %[[TARG4]], %[[TARG8]]
//       CHECK:  gpu.return %[[RES1]], %[[RES2]]
gpu.module @test_kernel {
gpu.func @test_transpose(%a: memref<2x64xf16>) -> (vector<2x4x1x16xf16>, vector<4x2x16x1xf16>) {
  %c0 = arith.constant 0 : index
  %0 = xetile.init_tile %a[%c0, %c0] : memref<2x64xf16> -> !xetile.tile<2x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>>
  %1 = xetile.load_tile %0 : !xetile.tile<2x64xf16, #xetile.tile_attr<inner_blocks = [1, 16]>> -> vector<2x4x1x16xf16>
  %2 = vector.transpose %1, [1, 0, 3, 2] : vector<2x4x1x16xf16> to vector<4x2x16x1xf16>
  gpu.return %1, %2 : vector<2x4x1x16xf16>, vector<4x2x16x1xf16>
}
}
