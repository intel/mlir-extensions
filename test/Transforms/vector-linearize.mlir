// RUN: imex-opt %s -split-input-file -imex-vector-linearize | FileCheck %s

// CHECK-LABEL: @test_linearize
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x2xf32>) -> vector<2x2xf32> {
//       CHECK: %[[T0:.*]] = vector.shape_cast %[[ARG0]] : vector<2x2xf32> to vector<4xf32>
//       CHECK: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
//       CHECK: %[[T1:.*]] = math.sin %[[T0]] : vector<4xf32>
//       CHECK: %[[T2:.*]] = arith.addf %[[T0]], %[[CST]] : vector<4xf32>
//       CHECK: %[[T3:.*]] = arith.addf %[[T2]], %[[T1]] : vector<4xf32>
//       CHECK: %[[T4:.*]] = vector.shape_cast %[[T3]] : vector<4xf32> to vector<2x2xf32>
//       CHECK: return %[[T4]] : vector<2x2xf32>
func.func @test_linearize(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>
// Arith and math ops are handled in generic way, check some of them
  %1 = math.sin %arg0 : vector<2x2xf32>
  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>
  %3 = arith.addf %2, %1 :  vector<2x2xf32>
  return %3 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: test_const_novector
//       CHECK:  %[[R:.*]] = arith.constant 42 : i32
//       CHECK:  return %[[R]] : i32
func.func @test_const_novector() -> i32 {
  %0 = arith.constant 42 : i32
  return %0 : i32
}

// -----
// CHECK-LABEL: test_create_mask
//       CHECK: vector.create_mask {{.*}} : vector<16xi1>
func.func @test_create_mask() -> vector<1x16xi1> {
  %c0 = arith.constant 0 : index
  %c20 = arith.constant 20 : index
  %0 = vector.create_mask %c0, %c20 : vector<1x16xi1>
  return %0 : vector<1x16xi1>
}

// -----
// CHECK-LABEL: test_extract_strided_slice
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<8x16xf32>) -> vector<8x8xf32>
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<8x16xf32> to vector<128xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
//       CHECK: [8, 9, 10, 11, 12, 13, 14, 15,
//       CHECK: 24, 25, 26, 27, 28, 29, 30, 31,
//       CHECK: 40, 41, 42, 43, 44, 45, 46, 47,
//       CHECK: 56, 57, 58, 59, 60, 61, 62, 63,
//       CHECK: 72, 73, 74, 75, 76, 77, 78, 79,
//       CHECK: 88, 89, 90, 91, 92, 93, 94, 95,
//       CHECK: 104, 105, 106, 107, 108, 109, 110, 111,
//       CHECK: 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf32>, vector<128xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<8x8xf32>
//       CHECK: return %[[RES]] : vector<8x8xf32>
func.func @test_extract_strided_slice_1(%arg0 : vector<8x16xf32>) -> vector<8x8xf32> {
  %0 = vector.extract_strided_slice %arg0 { sizes = [8, 8], strides = [1, 1], offsets = [0, 8]}
     : vector<8x16xf32> to vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----
// CHECK-LABEL: test_extract_strided_slice_2
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x32x8xf32>) -> vector<1x8x8xf32>
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x32x8xf32> to vector<512xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
//       CHECK: [448, 449, 450, 451, 452, 453, 454, 455,
//       CHECK: 456, 457, 458, 459, 460, 461, 462, 463,
//       CHECK: 464, 465, 466, 467, 468, 469, 470, 471,
//       CHECK: 472, 473, 474, 475, 476, 477, 478, 479,
//       CHECK: 480, 481, 482, 483, 484, 485, 486, 487,
//       CHECK: 488, 489, 490, 491, 492, 493, 494, 495,
//       CHECK: 496, 497, 498, 499, 500, 501, 502, 503,
//       CHECK: 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xf32>, vector<512xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<1x8x8xf32>
//       CHECK: return %[[RES]] : vector<1x8x8xf32>
func.func @test_extract_strided_slice_2(%arg0 : vector<2x32x8xf32>) -> vector<1x8x8xf32> {
  %0 = vector.extract_strided_slice %arg0 { offsets = [1, 24], strides = [1, 1], sizes = [1, 8] }
    : vector<2x32x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// -----
// CHECK-LABEL: test_vector_shuffle
//  CHECK-SAME: (%[[ORIG_ARG1:.*]]: vector<4x4xf32>, %[[ORIG_ARG2:.*]]: vector<4x4xf32>) -> vector<8x4xf32> {
//       CHECK: %[[ARG2:.*]] = vector.shape_cast %[[ORIG_ARG2]] : vector<4x4xf32> to vector<16xf32>
//       CHECK: %[[ARG1:.*]] = vector.shape_cast %[[ORIG_ARG1]] : vector<4x4xf32> to vector<16xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG1]], %[[ARG2]]
//       CHECK: [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
//       CHECK: 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<32xf32> to vector<8x4xf32>
//       CHECK: return %[[RES]] : vector<8x4xf32>
func.func @test_vector_shuffle(%arg0: vector<4x4xf32>, %arg1: vector<4x4xf32>) -> vector<8x4xf32> {
  %0 = vector.shuffle %arg0, %arg1 [0, 4, 1, 5, 2, 6, 3, 7] : vector<4x4xf32>, vector<4x4xf32>
  return %0 : vector<8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_extract
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x4xf32>) -> vector<8x4xf32>
// CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
// CHECK: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
// CHECK: 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<64xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<32xf32> to vector<8x4xf32>
// CHECK: return %[[RES]] : vector<8x4xf32>
func.func @test_vector_extract(%arg0: vector<2x8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.extract %arg0[1]: vector<8x4xf32> from vector<2x8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_insert
// CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<8x4xf32>) -> vector<2x8x4xf32>
// CHECK: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[ARG_SRC:.*]] = vector.shape_cast %[[SRC]] : vector<8x4xf32> to vector<32xf32>
// CHECK: %[[SHUFFLE0:.*]] = vector.shuffle %[[ARG_SRC]], %[[ARG_SRC]]
// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
// CHECK: 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 0, 0, 0, 0, 0, 0,
// CHECK: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
// CHECK: vector<32xf32>, vector<32xf32>
// CHECK: %[[SHUFFLE1:.*]] = vector.shuffle %[[ARG_DEST]], %[[SHUFFLE0]]
// CHECK: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
// CHECK-SAME: 88, 89, 90, 91, 92, 93, 94, 95, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
// CHECK-SAME: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<64xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE1]] : vector<64xf32> to vector<2x8x4xf32>
// CHECK: return %[[RES]] : vector<2x8x4xf32>
func.func @test_vector_insert(%arg0: vector<2x8x4xf32>, %arg1: vector<8x4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.insert %arg1, %arg0[0]: vector<8x4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_insert_2d_idx
// CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<4xf32>) -> vector<2x8x4xf32>
// CHECK: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[SHUFFLE0:.*]] = vector.shuffle %[[SRC]], %[[SRC]] [0, 1, 2, 3, 0, 0
// CHECK: , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<4xf32>, vector<4xf32>
// CHECK: %[[SHUFFLE1:.*]] = vector.shuffle %[[ARG_DEST]], %[[SHUFFLE0]]
// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 66, 67, 16, 17, 18, 19, 20, 21,
// CHECK-SAME: 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
// CHECK-SAME: 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<64xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE1]] : vector<64xf32> to vector<2x8x4xf32>
// CHECK: return %[[RES]] : vector<2x8x4xf32>
func.func @test_vector_insert_2d_idx(%arg0: vector<2x8x4xf32>, %arg1: vector<4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.insert %arg1, %arg0[0, 3]: vector<4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8xf32>) -> vector<8x2xf32>
// CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8xf32> to vector<16xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
// CHECK: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<16xf32>, vector<16xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
// CHECK: return %[[RES]] : vector<8x2xf32>
func.func @test_vector_transpose(%arg: vector<2x8xf32>) -> vector<8x2xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<2x8xf32> to vector<8x2xf32>
  return %0 : vector<8x2xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose_16x16
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
func.func @test_vector_transpose_16x16(%arg: vector<16x16xf32>) -> vector<16x16xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  return %0 : vector<16x16xf32>
}

// -----
// CHECK-LABEL: test_vector_store_load_4x4
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[R0:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK:       %[[L0:.*]] = vector.load %{{.*}}[%[[C0]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK:       %[[T0:.*]] = vector.shuffle %[[L0]], %[[L0]] [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<4xf32>, vector<4xf32>
// CHECK:       %[[R1:.*]] = vector.shuffle %[[R0]], %[[T0]] [16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
//
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[I1:.*]] = arith.addi %[[C0]], %[[C1]] : index
// CHECK:       %[[L1:.*]] = vector.load %{{.*}}[%[[I1]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK:       %[[T1:.*]] = vector.shuffle %[[L1]], %[[L1]] [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<4xf32>, vector<4xf32>
// CHECK:       %[[R2:.*]] = vector.shuffle %[[R1]], %[[T1]] [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
//
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[I2:.*]] = arith.addi %[[C0]], %[[C2]] : index
// CHECK:       %[[L2:.*]] = vector.load %{{.*}}[%[[I2]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK:       %[[T2:.*]] = vector.shuffle %[[L2]], %[[L2]] [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<4xf32>, vector<4xf32>
// CHECK:       %[[R3:.*]] = vector.shuffle %[[R2]], %[[T2]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
//
// CHECK:       %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[I3:.*]] = arith.addi %[[C0]], %[[C3]] : index
// CHECK:       %[[L3:.*]] = vector.load %{{.*}}[%[[I3]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
// CHECK:       %[[T3:.*]] = vector.shuffle %[[L3]], %[[L3]] [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<4xf32>, vector<4xf32>
// CHECK:       %[[R4:.*]] = vector.shuffle %[[R3]], %[[T3]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19] : vector<16xf32>, vector<16xf32>
//
// CHECK:       %[[V:.*]] = vector.shape_cast %{{.*}} : vector<4x4xf32> to vector<16xf32>
// CHECK:       %[[S0:.*]] = vector.shuffle %[[V]], %[[V]] [0, 1, 2, 3] : vector<16xf32>, vector<16xf32>
// CHECK:       vector.store %[[S0]], %{{.*}}[%[[C0]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
//
// CHECK:       %[[S1:.*]] = vector.shuffle %[[V]], %[[V]] [4, 5, 6, 7] : vector<16xf32>, vector<16xf32>
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[I1:.*]] = arith.addi %[[C0]], %[[C1]] : index
// CHECK:       vector.store %[[S1]], %{{.*}}[%[[I1]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
//
// CHECK:       %[[S2:.*]] = vector.shuffle %[[V]], %[[V]] [8, 9, 10, 11] : vector<16xf32>, vector<16xf32>
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[I2:.*]] = arith.addi %[[C0]], %[[C2]] : index
// CHECK:       vector.store %[[S2]], %{{.*}}[%[[I2]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>
//
// CHECK:       %[[S3:.*]] = vector.shuffle %[[V]], %[[V]] [12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
// CHECK:       %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[I3:.*]] = arith.addi %[[C0]], %[[C3]] : index
// CHECK:       vector.store %[[S3]], %{{.*}}[%[[I3]], %[[C0]]] : memref<4x4xf32>, vector<4xf32>

func.func @test_vector_store_load_4x4(%buffer: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %buffer[%c0, %c0] : memref<4x4xf32>, vector<4x4xf32>
  vector.store %0, %buffer[%c0, %c0] : memref<4x4xf32>, vector<4x4xf32>
  return
}
