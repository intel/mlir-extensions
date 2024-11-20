// RUN: imex-opt %s -split-input-file -imex-xegpu-hoist-transpose | FileCheck %s

// CHECK-LABEL: func.func @test_hoist_transpose_0(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<64x64xf16>) -> vector<16x16xf16> {
// CHECK: %[[T1:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
// CHECK: %[[T2:.*]] = vector.transpose %[[T1]], [1, 0] : vector<32x16xf16> to vector<16x32xf16>
// CHECK: %[[T3:.*]] = vector.extract_strided_slice %[[T2]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<16x32xf16> to vector<16x16xf16>
// CHECK: %[[T4:.*]] = vector.extract_strided_slice %[[T2]] {offsets = [0, 16], sizes = [16, 16], strides = [1, 1]} : vector<16x32xf16> to vector<16x16xf16>
func.func @test_hoist_transpose_0(%arg0: memref<64x64xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf16>
  %2 = vector.extract_strided_slice %1 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
  %3 = vector.extract_strided_slice %1 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
  %4 = vector.transpose %2, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %5 = vector.transpose %3, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %6 = arith.addf %4, %5 : vector<16x16xf16>
  return %6 : vector<16x16xf16>
}


// -----
// CHECK-LABEL: func.func @test_hoist_transpose_1(
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z]+]]: memref<64x64xf16>) -> vector<16x16xf16> {
// CHECK: %[[T1:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
// CHECK: %[[T2:.*]] = vector.extract %[[T1]][0] : vector<32x16xf16> from vector<2x32x16xf16>
// CHECK: %[[T3:.*]] = vector.extract %[[T1]][1] : vector<32x16xf16> from vector<2x32x16xf16>
// CHECK: %[[T4:.*]] = vector.transpose %[[T2]], [1, 0] : vector<32x16xf16> to vector<16x32xf16>
// CHECK: %[[T5:.*]] = vector.extract_strided_slice %[[T4]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<16x32xf16> to vector<16x16xf16>
// CHECK: %[[T6:.*]] = vector.extract_strided_slice %[[T4]] {offsets = [0, 16], sizes = [16, 16], strides = [1, 1]} : vector<16x32xf16> to vector<16x16xf16>
// CHECK: %[[T7:.*]] = vector.transpose %[[T3]], [1, 0] : vector<32x16xf16> to vector<16x32xf16>
// CHECK: %[[T8:.*]] = vector.extract_strided_slice %[[T7]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<16x32xf16> to vector<16x16xf16>
// CHECK: %[[T9:.*]] = vector.extract_strided_slice %[[T7]] {offsets = [0, 16], sizes = [16, 16], strides = [1, 1]} : vector<16x32xf16> to vector<16x16xf16>
func.func @test_hoist_transpose_1(%arg0: memref<64x64xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
  %2 = vector.extract %1[0] : vector<32x16xf16> from vector<2x32x16xf16>
  %3 = vector.extract %1[1] : vector<32x16xf16> from vector<2x32x16xf16>
  %4 = vector.extract_strided_slice %2 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
  %5 = vector.extract_strided_slice %2 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
  %6 = vector.extract_strided_slice %3 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
  %7 = vector.extract_strided_slice %3 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
  %8 = vector.transpose %4, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %9 = vector.transpose %5, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %10 = vector.transpose %6, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %11 = vector.transpose %7, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %12 = arith.addf %8, %9 : vector<16x16xf16>
  %13 = arith.addf %10, %11 : vector<16x16xf16>
  %14 = arith.addf %12, %13 : vector<16x16xf16>
  return %14 : vector<16x16xf16>
}
