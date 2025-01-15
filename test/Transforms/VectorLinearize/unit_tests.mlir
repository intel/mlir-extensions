// RUN: imex-opt %s -split-input-file -imex-vector-linearize | FileCheck %s

// -----
func.func @test() -> vector<4x2xf16> {
  %cst = arith.constant dense<0.0> : vector<8x1xf32>
  %1 = vector.extract_strided_slice %cst {offsets = [0, 0], sizes = [4, 1], strides = [1, 1]} : vector<8x1xf32> to vector<4x1xf32>
  //CHECK: %[[r1:.*]] = vector.bitcast %{{.*}} : vector<4xf32> to vector<8xf16>
  //CHECK: %[[r2:.*]] = vector.shape_cast %[[r1]] : vector<8xf16> to vector<4x2xf16>
  %2 = vector.bitcast %1 : vector<4x1xf32> to vector<4x2xf16>
  return %2 : vector<4x2xf16>
}
