// RUN: imex-opt %s -split-input-file -imex-xegpu-optimize-transpose | FileCheck %s

// CHECK-LABEL: @test_no_scf
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C0]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<8x16x2xf16>
// CHECK: xegpu.dpas %[[ARG1]], %[[T1]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>

func.func @test_no_scf(%arg0 : memref<64x64xf16>, %arg1 : vector<8x16xf16>)  -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<16x16xf16>
  %2 = vector.transpose %1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %3 = vector.shape_cast %2 {packed} : vector<16x16xf16> to vector<256xf16>
  %4 = vector.shuffle %3, %3 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
  %5 = vector.shape_cast %4 {packed} : vector<256xf16> to vector<8x16x2xf16>
  %6 = xegpu.dpas %arg1, %5 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return %6 : vector<8x16xf32>
}

// -----
// NOTE: This function is not changed by the pass because "packed" attribute is not present.
//
// CHECK-LABEL: func.func @test_no_scf_no_packed_layout(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C0]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<16x16xf16>
// CHECK: %[[T2:.*]] = vector.transpose %[[T1]], [1, 0] : vector<16x16xf16> to vector<16x16xf16>
// CHECK: %[[T3:.*]] = vector.shape_cast %[[T2]] {packed} : vector<16x16xf16> to vector<256xf16>
// CHECK: %[[T4:.*]] = vector.shuffle %[[T3]], %[[T3]] [{{.*}}] {packed} : vector<256xf16>, vector<256xf16>
// CHECK: %[[T5:.*]] = vector.shape_cast %[[T4]] : vector<256xf16> to vector<8x16x2xf16>
// CHECK: xegpu.dpas %[[ARG1]], %[[T5]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
func.func @test_no_scf_no_packed_layout(%arg0 : memref<64x64xf16>, %arg1 : vector<8x16xf16>)  -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<16x16xf16>
  %2 = vector.transpose %1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %3 = vector.shape_cast %2 {packed} : vector<16x16xf16> to vector<256xf16>
  %4 = vector.shuffle %3, %3 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
  %5 = vector.shape_cast %4 : vector<256xf16> to vector<8x16x2xf16>
  %6 = xegpu.dpas %arg1, %5 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return %6 : vector<8x16xf32>
}


// -----
// CHECK-LABEL: func.func @test_scf_for(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %{{.*}}] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %[[T1]], %{{.*}} = %{{.*}}) -> (!xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32>) {
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[ARG3]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<8x16x2xf16>
// CHECK: %[[T4:.*]] = xegpu.dpas %[[ARG1]], %[[T3]], %{{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
func.func @test_scf_for(%arg0 : memref<64x64xf16>, %arg1 : vector<8x16xf16>)  -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<0.0> : vector<128xf32>
  %cst_c = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  %result:2 = scf.for %k = %c0 to %c64 step %c16 iter_args(%arg2 = %0, %arg3 = %cst_c) -> (!xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32> ) {
    %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<16x16xf16>
    %2 = vector.transpose %1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %3 = vector.shape_cast %2 {packed} : vector<16x16xf16> to vector<256xf16>
    %4 = vector.shuffle %3, %3 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %5 = vector.shape_cast %4 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %6 = xegpu.dpas %arg1, %5, %cst_c : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %7 = xegpu.update_nd_offset %arg2, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
    scf.yield %7, %6 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32>
  }
  return %result#1 : vector<8x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_scf_for_preop(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %{{.*}}] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %[[T1]], %{{.*}} = %{{.*}}) -> (!xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32>) {
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[ARG3]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<8x16x2xf16>
// CHECK: %[[T4:.*]] = math.exp %[[T3]] : vector<8x16x2xf16>
// CHECK: %[[T5:.*]] = math.log2 %[[T4]] : vector<8x16x2xf16>
// CHECK: xegpu.dpas %[[ARG1]], %[[T5]], %{{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
func.func @test_scf_for_preop(%arg0 : memref<64x64xf16>, %arg1 : vector<8x16xf16>)  -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<0.0> : vector<128xf32>
  %cst_c = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  %result:2 = scf.for %k = %c0 to %c64 step %c16 iter_args(%arg2 = %0, %arg3 = %cst_c) -> (!xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32> ) {
    %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<16x16xf16>
    %2 = vector.transpose %1, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %3 = vector.shape_cast %2 {packed} : vector<16x16xf16> to vector<256xf16>
    %4 = vector.shuffle %3, %3 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %5 = vector.shape_cast %4 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %8 = math.exp %5 : vector<8x16x2xf16>
    %9 = math.log2 %8 : vector<8x16x2xf16>
    %6 = xegpu.dpas %arg1, %9, %cst_c : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %7 = xegpu.update_nd_offset %arg2, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
    scf.yield %7, %6 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32>
  }
  return %result#1 : vector<8x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_scf_for_large_loads(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %{{.*}}] : memref<64x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %[[T1]], %{{.*}} = %{{.*}}) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32>) {
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[ARG3]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<8x32x2xf16>
// CHECK: %[[T4:.*]] = vector.extract_strided_slice %[[T3]] {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
// CHECK: %[[T5:.*]] = vector.extract_strided_slice %[[T3]] {offsets = [0, 16, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
// CHECK: %[[T6:.*]] = xegpu.dpas %{{.*}}, %[[T4]], %{{.*}} : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK: %[[T7:.*]] = xegpu.dpas %{{.*}}, %[[T5]], %[[T6]] : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
func.func @test_scf_for_large_loads(%arg0: memref<64x64xf16>, %arg1: vector<8x16xf16>) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
  %0 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
  %1 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  %2:2 = scf.for %arg2 = %c0 to %c64 step %c16 iter_args(%arg3 = %1, %arg4 = %0) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32>) {
    %3 = xegpu.load_nd %arg3  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<32x16xf16>
    %4 = vector.transpose %3, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
    %5 = vector.shape_cast %4 {packed} : vector<16x32xf16> to vector<512xf16>
    %6 = vector.shuffle %5, %5 [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 64, 96, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70, 102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111, 80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89, 121, 90, 122, 91, 123, 92, 124, 93, 125, 94, 126, 95, 127, 128, 160, 129, 161, 130, 162, 131, 163, 132, 164, 133, 165, 134, 166, 135, 167, 136, 168, 137, 169, 138, 170, 139, 171, 140, 172, 141, 173, 142, 174, 143, 175, 144, 176, 145, 177, 146, 178, 147, 179, 148, 180, 149, 181, 150, 182, 151, 183, 152, 184, 153, 185, 154, 186, 155, 187, 156, 188, 157, 189, 158, 190, 159, 191, 192, 224, 193, 225, 194, 226, 195, 227, 196, 228, 197, 229, 198, 230, 199, 231, 200, 232, 201, 233, 202, 234, 203, 235, 204, 236, 205, 237, 206, 238, 207, 239, 208, 240, 209, 241, 210, 242, 211, 243, 212, 244, 213, 245, 214, 246, 215, 247, 216, 248, 217, 249, 218, 250, 219, 251, 220, 252, 221, 253, 222, 254, 223, 255, 256, 288, 257, 289, 258, 290, 259, 291, 260, 292, 261, 293, 262, 294, 263, 295, 264, 296, 265, 297, 266, 298, 267, 299, 268, 300, 269, 301, 270, 302, 271, 303, 272, 304, 273, 305, 274, 306, 275, 307, 276, 308, 277, 309, 278, 310, 279, 311, 280, 312, 281, 313, 282, 314, 283, 315, 284, 316, 285, 317, 286, 318, 287, 319, 320, 352, 321, 353, 322, 354, 323, 355, 324, 356, 325, 357, 326, 358, 327, 359, 328, 360, 329, 361, 330, 362, 331, 363, 332, 364, 333, 365, 334, 366, 335, 367, 336, 368, 337, 369, 338, 370, 339, 371, 340, 372, 341, 373, 342, 374, 343, 375, 344, 376, 345, 377, 346, 378, 347, 379, 348, 380, 349, 381, 350, 382, 351, 383, 384, 416, 385, 417, 386, 418, 387, 419, 388, 420, 389, 421, 390, 422, 391, 423, 392, 424, 393, 425, 394, 426, 395, 427, 396, 428, 397, 429, 398, 430, 399, 431, 400, 432, 401, 433, 402, 434, 403, 435, 404, 436, 405, 437, 406, 438, 407, 439, 408, 440, 409, 441, 410, 442, 411, 443, 412, 444, 413, 445, 414, 446, 415, 447, 448, 480, 449, 481, 450, 482, 451, 483, 452, 484, 453, 485, 454, 486, 455, 487, 456, 488, 457, 489, 458, 490, 459, 491, 460, 492, 461, 493, 462, 494, 463, 495, 464, 496, 465, 497, 466, 498, 467, 499, 468, 500, 469, 501, 470, 502, 471, 503, 472, 504, 473, 505, 474, 506, 475, 507, 476, 508, 477, 509, 478, 510, 479, 511] {packed} : vector<512xf16>, vector<512xf16>
    %7 = vector.shape_cast %6 {packed} : vector<512xf16> to vector<8x32x2xf16>
    %8 = vector.extract_strided_slice %7 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    %9 = vector.extract_strided_slice %7 {offsets = [0, 16, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    %10 = xegpu.dpas %arg1, %8, %0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %11 = xegpu.dpas %arg1, %9, %10 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %12 = xegpu.update_nd_offset %arg3, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
    scf.yield %12, %11 : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 1 : i64>>, vector<8x16xf32>
  }
  return %2#1 : vector<8x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_i32(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xi32>) -> vector<8x32xi32> {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %{{.*}}] : memref<64x64xi32> -> !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]] <{transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<8x32xi32>
// CHECK: return %[[T1]] : vector<8x32xi32>
func.func @test_i32(%arg0: memref<64x64xi32>) -> vector<8x32xi32> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xi32> -> !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<32x8xi32>
  %2 = vector.transpose %1, [1, 0] : vector<32x8xi32> to vector<8x32xi32>
  return %2 : vector<8x32xi32>
}

// -----
// CHECK-LABEL: func.func @test_i64(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xi64>) -> vector<4x8xi64> {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %{{.*}}] : memref<64x64xi64> -> !xegpu.tensor_desc<8x4xi64, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]] <{transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<8x4xi64, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<4x8xi64>
// return %[[T1]] : vector<4x8xi64>
func.func @test_i64(%arg0: memref<64x64xi64>) -> vector<4x8xi64> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xi64> -> !xegpu.tensor_desc<8x4xi64, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x4xi64, #xegpu.block_tdesc_attr<array_length = 1 : i64>> -> vector<8x4xi64>
  %2 = vector.transpose %1, [1, 0] : vector<8x4xi64> to vector<4x8xi64>
  return %2 : vector<4x8xi64>
}
