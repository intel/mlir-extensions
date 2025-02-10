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

// -----
// test the insert_strided_slice
func.func @test() -> vector<2x4xf32> {
  %src = arith.constant dense<1.0> : vector<1x2xf32>
  %dst = arith.constant dense<1.0> : vector<2x4xf32>
  //CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0], strides = [1]} : vector<2xf32> into vector<8xf32>
  %1 = vector.insert_strided_slice %src, %dst {offsets = [0, 0], strides = [1, 1]} : vector<1x2xf32> into vector<2x4xf32>
  //CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [6], strides = [1]} : vector<2xf32> into vector<8xf32>
  %2 = vector.insert_strided_slice %src, %1 {offsets = [1, 2], strides = [1, 1]} : vector<1x2xf32> into vector<2x4xf32>
  return %2 : vector<2x4xf32>
}


// -----
// test the loop
func.func @test() -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %1 = arith.constant dense<1.0> : vector<2x4xf32>
  //CHECK: scf.for %{{.*}} iter_args(%{{.*}}) -> (vector<8xf32>)
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg1 = %1) -> (vector<2x4xf32>) {
    %2 = arith.addf %1, %arg1 : vector<2x4xf32>
    scf.yield %2 : vector<2x4xf32>
  }
  return %r : vector<2x4xf32>
}
