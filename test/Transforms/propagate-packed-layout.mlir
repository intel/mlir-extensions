// RUN: imex-opt %s -split-input-file -imex-propagate-packed-layout | FileCheck %s

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: vector<8x16xf16>, %[[ARG2:.*]]: vector<16x16xf16>)
//       CHECK:  %[[A1:.*]] = vector.shape_cast %[[ARG1]] : vector<8x16xf16> to vector<128xf16>
//       CHECK:  %[[A2:.*]] = vector.shuffle %[[A1]], %[[A1]] [{{.*}}] : vector<128xf16>, vector<128xf16>
//       CHECK:  %[[A3:.*]] = vector.shape_cast %[[A2]] : vector<128xf16> to vector<8x8x2xf16>
//       CHECK:  %[[B1:.*]] = vector.shape_cast %[[ARG2]] : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B2:.*]] = vector.shuffle %[[B1]], %[[B1]] [{{.*}}] : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B2]] : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A3]], %[[B3]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : vector<8x16xf16>, %arg2 : vector<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.dpas %arg1, %arg2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[RES1:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  %[[RES2:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES1]], %[[RES2]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<8x16xf32>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %3 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %3 : vector<8x16xf32>, vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  %[[B1:.*]] = vector.shape_cast %[[B]] : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B2:.*]] = vector.shuffle %[[B1]], %[[B1]] [{{.*}}] : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B2]] : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B3]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %1 : vector<8x16xf32>, vector<16x16xf16>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B1:.*]] = arith.extf %[[B]] : vector<8x16x2xf16> to vector<8x16x2xf32>
//       CHECK:  %[[B2:.*]] = arith.truncf %[[B1]] : vector<8x16x2xf32> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B2]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.extf %1: vector<16x16xf16> to vector<16x16xf32>
  %3 = arith.truncf %2: vector<16x16xf32> to vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B3:.*]] = arith.addf %[[B1]], %[[B2]] : vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B3]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B2:.*]] = arith.constant dense<1.000000e+00> : vector<16x16xf16>
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B2]] : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B4:.*]] = vector.shuffle %[[B3]], %[[B3]] [{{.*}}] : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B5:.*]] = vector.shape_cast %[[B4]] : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[B6:.*]] = arith.addf %[[B1]], %[[B5]] : vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B6]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.constant dense<1.0> : vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  %[[B3:.*]] = arith.addf %[[B1]], %[[B2]] : vector<16x16xf16>
//       CHECK:  %[[B4:.*]] = vector.shape_cast %[[B3]] : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B5:.*]] = vector.shuffle %[[B4]], %[[B4]] [{{.*}}] : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B6:.*]] = vector.shape_cast %[[B5]] : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B6]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]], %[[B2]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4, %2 : vector<8x16xf32>, vector<16x16xf16>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %{{.*}}: index)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  %[[RES1:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER1:.*]] = %[[B]], %[[ITER2:.*]] = %[[RES]]) -> (vector<8x16x2xf16>, vector<8x16xf32>) {
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B2:.*]] = arith.addf %[[ITER1]], %[[B1]] : vector<8x16x2xf16>
//       CHECK:  %[[RES2:.*]] = xegpu.dpas %[[A]], %[[B2]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  scf.yield %[[B2]], %[[RES2]] : vector<8x16x2xf16>, vector<8x16xf32>
//       CHECK:  }
//       CHECK:  return %[[RES1]]#1 : vector<8x16xf32>

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : index) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %3:2 = scf.for %i = %c0 to %arg3 step %c1 iter_args(%iter = %1, %res = %2) -> (vector<16x16xf16>, vector<8x16xf32>) {
    %4 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %5 = arith.addf %iter, %4 : vector<16x16xf16>
    %6 = xegpu.dpas %0, %5 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    scf.yield %5, %6: vector<16x16xf16>, vector<8x16xf32>
  }
  return %3#1 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %{{.*}}: i1)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = scf.if %{{.*}} -> (vector<8x16x2xf16>)
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  scf.yield %[[B1]]
//       CHECK:  else
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  scf.yield %[[B2]]
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : i1) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg3 -> (vector<16x16xf16>) {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  } else {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %{{.*}}: i1)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = scf.if %{{.*}} -> (vector<16x16xf16>)
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  scf.yield %[[B1]]
//       CHECK:  else
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  scf.yield %[[B2]]
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B]] : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B4:.*]] = vector.shuffle %[[B3]], %[[B3]] [{{.*}}] : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B5:.*]] = vector.shape_cast %[[B4]] : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B5]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]], %[[B]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : i1) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg3 -> (vector<16x16xf16>) {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  } else {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %1 : vector<8x16xf32>, vector<16x16xf16>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %[[ARG3:.*]]: vector<16x16xf16>, %{{.*}}: i1)
//       CHECK:  %[[B2:.*]] = vector.shape_cast %[[ARG3]] : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B3:.*]] = vector.shuffle %[[B2]], %[[B2]] [{{.*}}] : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B4:.*]] = vector.shape_cast %[[B3]] : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] <{vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
//       CHECK:  %[[B:.*]] = scf.if %{{.*}} -> (vector<8x16x2xf16>)
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  scf.yield %[[B1]]
//       CHECK:  else
//       CHECK:  scf.yield %[[B4]]
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : vector<16x16xf16>, %arg4 : i1) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg4 -> (vector<16x16xf16>) {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  } else {
    scf.yield %arg3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}
