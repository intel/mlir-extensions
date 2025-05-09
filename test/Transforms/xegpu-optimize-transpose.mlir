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
// CHECK-LABEL: func.func @test_no_scf_array_len(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C0]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C16]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T2:.*]] = xegpu.load_nd %[[T0]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
// CHECK: %[[T4:.*]] = xegpu.load_nd %[[T1]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
func.func @test_no_scf_array_len(%arg0 : memref<64x64xf16>, %arg1 : vector<8x16xf16>)  -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
  %7 = vector.extract %1[0] : vector<16x16xf16> from vector<2x16x16xf16>
  %8 = vector.extract %1[1] : vector<16x16xf16> from vector<2x16x16xf16>
  %2 = vector.transpose %7, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %3 = vector.shape_cast %2 {packed} : vector<16x16xf16> to vector<256xf16>
  %4 = vector.shuffle %3, %3 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
  %5 = vector.shape_cast %4 {packed} : vector<256xf16> to vector<8x16x2xf16>
  %6 = xegpu.dpas %arg1, %5 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  %9 = vector.transpose %8, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %10 = vector.shape_cast %9 {packed} : vector<16x16xf16> to vector<256xf16>
  %11 = vector.shuffle %10, %10 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
  %12 = vector.shape_cast %11 {packed} : vector<256xf16> to vector<8x16x2xf16>
  %13 = xegpu.dpas %arg1, %12 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  %14 = arith.addf %6, %13 : vector<8x16xf32>
  return %14 : vector<8x16xf32>
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
// CHECK-LABEL: func.func @test_scf_for_array_len(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C0]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C16]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T3:.*]]:3 = scf.for %{{.*}} = %[[C0]] to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %[[ARG4:.*]] = %[[T1]], %[[ARG5:.*]] = %[[T2]]) -> (vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>) {
// CHECK: %[[T4:.*]] = xegpu.load_nd %[[ARG4]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
// CHECK: %[[T6:.*]] = xegpu.load_nd %[[ARG5]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
// CHECK: %[[T8:.*]] = xegpu.update_nd_offset %[[ARG4]], [%[[C32]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T9:.*]] = xegpu.update_nd_offset %[[ARG5]], [%[[C32]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: scf.yield %{{.*}}, %[[T8]], %[[T9]] : vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
func.func @test_scf_for_array_len(%arg0 : memref<64x64xf16>, %arg1 : vector<8x16xf16>)  -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<0.0> : vector<128xf32>
  %cst_c = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %result:2 = scf.for %k = %c0 to %c64 step %c32 iter_args(%arg3 = %cst_c, %arg2 = %0) -> (vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> ) {
    %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
    %8 = vector.extract %1[0] : vector<16x16xf16> from vector<2x16x16xf16>
    %9 = vector.extract %1[1] : vector<16x16xf16> from vector<2x16x16xf16>
    %2 = vector.transpose %8, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %3 = vector.shape_cast %2 {packed} : vector<16x16xf16> to vector<256xf16>
    %4 = vector.shuffle %3, %3 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %5 = vector.shape_cast %4 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %6 = xegpu.dpas %arg1, %5, %cst_c : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %10 = vector.transpose %9, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %11 = vector.shape_cast %10 {packed} : vector<16x16xf16> to vector<256xf16>
    %12 = vector.shuffle %11, %11 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %13 = vector.shape_cast %12 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %14 = xegpu.dpas %arg1, %13, %6 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %7 = xegpu.update_nd_offset %arg2, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
    scf.yield %14, %7: vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  }
  return %result#0 : vector<8x16xf32>
}

// -----
// CHECK-LABEL: @test_nested_scf_for_array_len(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>, %[[ARG2:[a-zA-Z0-9]+]]: memref<64x64xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK: scf.for {{.*}} {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %[[C0]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}, %[[C16]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T2:.*]]:3 = scf.for {{.*}} iter_args(%[[ARG5:.*]] = %[[CST]], %[[ARG6:.*]] = %[[T0]], %[[ARG7:.*]] = %[[T1]]) -> (vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>) {
// CHECK-DAG: %[[T4:.*]] = xegpu.load_nd %[[ARG6]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
// CHECK-DAG: %[[T6:.*]] = xegpu.load_nd %[[ARG7]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
// CHECK-DAG: %[[T8:.*]] = xegpu.update_nd_offset %[[ARG6]], [{{.*}}] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK-DAG: %[[T9:.*]] = xegpu.update_nd_offset %[[ARG7]], [{{.*}}] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: scf.yield %{{.*}}, %[[T8]], %[[T9]] : vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T3:.*]] = xegpu.create_nd_tdesc %[[ARG2]][{{.*}}] : memref<64x64xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
// CHECK: xegpu.store_nd %[[T2]]#0, %{{.*}}  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
func.func @test_nested_scf_for_array_len(%arg0: memref<64x64xf16>, %arg1: vector<8x16xf16>, %arg2: memref<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  scf.for %arg3 = %c0 to %c64 step %c8 {
    %0 = xegpu.create_nd_tdesc %arg0[%arg3, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
    %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
    %1 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
    %2:2 = scf.for %arg4 = %c0 to %c64 step %c32 iter_args(%arg5 = %1, %arg6 = %0) -> (vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>) {
      %4 = xegpu.load_nd %arg6  : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
      %5 = vector.extract %4[0] : vector<16x16xf16> from vector<2x16x16xf16>
      %6 = vector.extract %4[1] : vector<16x16xf16> from vector<2x16x16xf16>
      %7 = vector.transpose %5, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
      %8 = vector.shape_cast %7 {packed} : vector<16x16xf16> to vector<256xf16>
      %9 = vector.shuffle %8, %8 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %10 = vector.shape_cast %9 {packed} : vector<256xf16> to vector<8x16x2xf16>
      %11 = xegpu.dpas %arg1, %10, %1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %12 = vector.transpose %6, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
      %13 = vector.shape_cast %12 {packed} : vector<16x16xf16> to vector<256xf16>
      %14 = vector.shuffle %13, %13 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
      %15 = vector.shape_cast %14 {packed} : vector<256xf16> to vector<8x16x2xf16>
      %16 = xegpu.dpas %arg1, %15, %11 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %17 = xegpu.update_nd_offset %arg6, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      scf.yield %16, %17 : vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
    }
    %3 = xegpu.create_nd_tdesc %arg2[%arg3, %c0] : memref<64x64xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
    xegpu.store_nd %2#0, %3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<array_length = 1 : i64>>
  }
  return
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
// CHECK-LABEL: func.func @test_scf_for_preop_array_len(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C0]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C16]]] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T3:.*]]:3 = scf.for %{{.*}} = %[[C0]] to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %[[ARG4:.*]] = %[[T1]], %[[ARG5:.*]] = %[[T2]]) -> (vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>) {
// CHECK: %[[T4:.*]] = xegpu.load_nd %[[ARG4]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
// CHECK: %[[T5:.*]] = math.exp %[[T4]] : vector<8x16x2xf16>
// CHECK: %[[T6:.*]] = math.log2 %[[T5]] : vector<8x16x2xf16>
// CHECK: %[[T8:.*]] = xegpu.load_nd %[[ARG5]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16x2xf16>
// CHECK: %[[T9:.*]] = math.exp %[[T8]] : vector<8x16x2xf16>
// CHECK: %[[T10:.*]] = math.log2 %[[T9]] : vector<8x16x2xf16>
// CHECK: %[[T12:.*]] = xegpu.update_nd_offset %[[ARG4]], [%[[C32]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T13:.*]] = xegpu.update_nd_offset %[[ARG5]], [%[[C32]], %[[C0]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: scf.yield %{{.*}}, %[[T12]], %[[T13]] : vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
func.func @test_scf_for_preop_array_len(%arg0 : memref<64x64xf16>, %arg1 : vector<8x16xf16>)  -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<0.0> : vector<128xf32>
  %cst_c = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %result:2 = scf.for %k = %c0 to %c64 step %c32 iter_args(%arg3 = %cst_c, %arg2 = %0) -> (vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> ) {
    %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
    %8 = vector.extract %1[0] : vector<16x16xf16> from vector<2x16x16xf16>
    %9 = vector.extract %1[1] : vector<16x16xf16> from vector<2x16x16xf16>
    %2 = vector.transpose %8, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %3 = vector.shape_cast %2 {packed} : vector<16x16xf16> to vector<256xf16>
    %4 = vector.shuffle %3, %3 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %5 = vector.shape_cast %4 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %15 = math.exp %5 : vector<8x16x2xf16>
    %16 = math.log2 %15 : vector<8x16x2xf16>
    %6 = xegpu.dpas %arg1, %16, %cst_c : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %10 = vector.transpose %9, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %11 = vector.shape_cast %10 {packed} : vector<16x16xf16> to vector<256xf16>
    %12 = vector.shuffle %11, %11 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %13 = vector.shape_cast %12 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %17 = math.exp %13 : vector<8x16x2xf16>
    %18 = math.log2 %17 : vector<8x16x2xf16>
    %14 = xegpu.dpas %arg1, %18, %6 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %7 = xegpu.update_nd_offset %arg2, [%c32, %c0] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
    scf.yield %14, %7: vector<8x16xf32>, !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  }
  return %result#0 : vector<8x16xf32>
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
// CHECK-LABEL: func.func @test_scf_for_large_loads_array_len(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<64x64xf16>, %[[ARG1:[a-zA-Z0-9]+]]: vector<8x16xf16>) -> vector<8x16xf32> {
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C0]]] : memref<64x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C16]]] : memref<64x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T3:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %[[ARG4:.*]] = %[[T1]], %[[ARG5:.*]] = %[[T2]]) -> (vector<8x16xf32>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>) {
// CHECK: %[[T4:.*]] = xegpu.load_nd %[[ARG4]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x32x2xf16>
// CHECK: %[[T9:.*]] = xegpu.load_nd %[[ARG5]] <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x32x2xf16>
// CHECK: %[[T14:.*]] = xegpu.update_nd_offset %[[ARG4]], [%[[C32]], %[[C0]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: %[[T15:.*]] = xegpu.update_nd_offset %[[ARG5]], [%[[C32]], %[[C0]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
// CHECK: scf.yield %{{.*}}, %[[T14]], %[[T15]] : vector<8x16xf32>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
func.func @test_scf_for_large_loads_array_len(%arg0: memref<64x64xf16>, %arg1: vector<8x16xf16>) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
  %0 = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
  %1 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<64x64xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %2:2 = scf.for %arg2 = %c0 to %c64 step %c16 iter_args(%arg4 = %0, %arg3 = %1) -> (vector<8x16xf32>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>) {
    %3 = xegpu.load_nd %arg3  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x16xf16>
    %13 = vector.extract %3[0] : vector<32x16xf16> from vector<2x32x16xf16>
    %14 = vector.extract %3[1] : vector<32x16xf16> from vector<2x32x16xf16>
    %4 = vector.transpose %13, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
    %5 = vector.shape_cast %4 {packed} : vector<16x32xf16> to vector<512xf16>
    %6 = vector.shuffle %5, %5 [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 64, 96, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70, 102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111, 80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89, 121, 90, 122, 91, 123, 92, 124, 93, 125, 94, 126, 95, 127, 128, 160, 129, 161, 130, 162, 131, 163, 132, 164, 133, 165, 134, 166, 135, 167, 136, 168, 137, 169, 138, 170, 139, 171, 140, 172, 141, 173, 142, 174, 143, 175, 144, 176, 145, 177, 146, 178, 147, 179, 148, 180, 149, 181, 150, 182, 151, 183, 152, 184, 153, 185, 154, 186, 155, 187, 156, 188, 157, 189, 158, 190, 159, 191, 192, 224, 193, 225, 194, 226, 195, 227, 196, 228, 197, 229, 198, 230, 199, 231, 200, 232, 201, 233, 202, 234, 203, 235, 204, 236, 205, 237, 206, 238, 207, 239, 208, 240, 209, 241, 210, 242, 211, 243, 212, 244, 213, 245, 214, 246, 215, 247, 216, 248, 217, 249, 218, 250, 219, 251, 220, 252, 221, 253, 222, 254, 223, 255, 256, 288, 257, 289, 258, 290, 259, 291, 260, 292, 261, 293, 262, 294, 263, 295, 264, 296, 265, 297, 266, 298, 267, 299, 268, 300, 269, 301, 270, 302, 271, 303, 272, 304, 273, 305, 274, 306, 275, 307, 276, 308, 277, 309, 278, 310, 279, 311, 280, 312, 281, 313, 282, 314, 283, 315, 284, 316, 285, 317, 286, 318, 287, 319, 320, 352, 321, 353, 322, 354, 323, 355, 324, 356, 325, 357, 326, 358, 327, 359, 328, 360, 329, 361, 330, 362, 331, 363, 332, 364, 333, 365, 334, 366, 335, 367, 336, 368, 337, 369, 338, 370, 339, 371, 340, 372, 341, 373, 342, 374, 343, 375, 344, 376, 345, 377, 346, 378, 347, 379, 348, 380, 349, 381, 350, 382, 351, 383, 384, 416, 385, 417, 386, 418, 387, 419, 388, 420, 389, 421, 390, 422, 391, 423, 392, 424, 393, 425, 394, 426, 395, 427, 396, 428, 397, 429, 398, 430, 399, 431, 400, 432, 401, 433, 402, 434, 403, 435, 404, 436, 405, 437, 406, 438, 407, 439, 408, 440, 409, 441, 410, 442, 411, 443, 412, 444, 413, 445, 414, 446, 415, 447, 448, 480, 449, 481, 450, 482, 451, 483, 452, 484, 453, 485, 454, 486, 455, 487, 456, 488, 457, 489, 458, 490, 459, 491, 460, 492, 461, 493, 462, 494, 463, 495, 464, 496, 465, 497, 466, 498, 467, 499, 468, 500, 469, 501, 470, 502, 471, 503, 472, 504, 473, 505, 474, 506, 475, 507, 476, 508, 477, 509, 478, 510, 479, 511] {packed} : vector<512xf16>, vector<512xf16>
    %7 = vector.shape_cast %6 {packed} : vector<512xf16> to vector<8x32x2xf16>
    %8 = vector.extract_strided_slice %7 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    %9 = vector.extract_strided_slice %7 {offsets = [0, 16, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    %10 = xegpu.dpas %arg1, %8, %0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %11 = xegpu.dpas %arg1, %9, %10 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %15 = vector.transpose %14, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
    %16 = vector.shape_cast %15 {packed} : vector<16x32xf16> to vector<512xf16>
    %17 = vector.shuffle %16, %16 [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 64, 96, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70, 102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111, 80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89, 121, 90, 122, 91, 123, 92, 124, 93, 125, 94, 126, 95, 127, 128, 160, 129, 161, 130, 162, 131, 163, 132, 164, 133, 165, 134, 166, 135, 167, 136, 168, 137, 169, 138, 170, 139, 171, 140, 172, 141, 173, 142, 174, 143, 175, 144, 176, 145, 177, 146, 178, 147, 179, 148, 180, 149, 181, 150, 182, 151, 183, 152, 184, 153, 185, 154, 186, 155, 187, 156, 188, 157, 189, 158, 190, 159, 191, 192, 224, 193, 225, 194, 226, 195, 227, 196, 228, 197, 229, 198, 230, 199, 231, 200, 232, 201, 233, 202, 234, 203, 235, 204, 236, 205, 237, 206, 238, 207, 239, 208, 240, 209, 241, 210, 242, 211, 243, 212, 244, 213, 245, 214, 246, 215, 247, 216, 248, 217, 249, 218, 250, 219, 251, 220, 252, 221, 253, 222, 254, 223, 255, 256, 288, 257, 289, 258, 290, 259, 291, 260, 292, 261, 293, 262, 294, 263, 295, 264, 296, 265, 297, 266, 298, 267, 299, 268, 300, 269, 301, 270, 302, 271, 303, 272, 304, 273, 305, 274, 306, 275, 307, 276, 308, 277, 309, 278, 310, 279, 311, 280, 312, 281, 313, 282, 314, 283, 315, 284, 316, 285, 317, 286, 318, 287, 319, 320, 352, 321, 353, 322, 354, 323, 355, 324, 356, 325, 357, 326, 358, 327, 359, 328, 360, 329, 361, 330, 362, 331, 363, 332, 364, 333, 365, 334, 366, 335, 367, 336, 368, 337, 369, 338, 370, 339, 371, 340, 372, 341, 373, 342, 374, 343, 375, 344, 376, 345, 377, 346, 378, 347, 379, 348, 380, 349, 381, 350, 382, 351, 383, 384, 416, 385, 417, 386, 418, 387, 419, 388, 420, 389, 421, 390, 422, 391, 423, 392, 424, 393, 425, 394, 426, 395, 427, 396, 428, 397, 429, 398, 430, 399, 431, 400, 432, 401, 433, 402, 434, 403, 435, 404, 436, 405, 437, 406, 438, 407, 439, 408, 440, 409, 441, 410, 442, 411, 443, 412, 444, 413, 445, 414, 446, 415, 447, 448, 480, 449, 481, 450, 482, 451, 483, 452, 484, 453, 485, 454, 486, 455, 487, 456, 488, 457, 489, 458, 490, 459, 491, 460, 492, 461, 493, 462, 494, 463, 495, 464, 496, 465, 497, 466, 498, 467, 499, 468, 500, 469, 501, 470, 502, 471, 503, 472, 504, 473, 505, 474, 506, 475, 507, 476, 508, 477, 509, 478, 510, 479, 511] {packed} : vector<512xf16>, vector<512xf16>
    %18 = vector.shape_cast %17 {packed} : vector<512xf16> to vector<8x32x2xf16>
    %19 = vector.extract_strided_slice %18 {offsets = [0, 0, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    %20 = vector.extract_strided_slice %18 {offsets = [0, 16, 0], sizes = [8, 16, 2], strides = [1, 1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    %21 = xegpu.dpas %arg1, %19, %11 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %22 = xegpu.dpas %arg1, %20, %21 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %12 = xegpu.update_nd_offset %arg3, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
    scf.yield %22, %12 : vector<8x16xf32>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  }
  return %2#0 : vector<8x16xf32>
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

// -----
func.func @test_transpose_8x16xf16(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  %in = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %data = xegpu.load_nd %in : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %transpose = vector.transpose %data, [1, 0] : vector<8x16xf16> to vector<16x8xf16>
  %cast = vector.shape_cast %transpose : vector<16x8xf16> to vector<8x16xf16>
  %out = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  xegpu.store_nd %cast, %out : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  return

  //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
  //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %{{.*}}[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] <{packed}> : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  //CHECK: %[[r2:.*]] = vector.shape_cast %[[r1]] : vector<4x16x2xf16> to vector<128xf16>
  //CHECK: %[[r3:.*]] = vector.bitcast %[[r2]] : vector<128xf16> to vector<64xf32>
  //CHECK: %[[r4:.*]] = vector.shape_cast %[[r3]] : vector<64xf32> to vector<4x16xf32>
  //CHECK: %[[alloc:.*]] = memref.alloc() : memref<4096xf32, 3>
  //CHECK: %[[r22:.*]] = xegpu.create_tdesc %[[alloc]], %{{.*}} : memref<4096xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 4 : i64>>
  //CHECK: xegpu.store %[[r4]], %[[r22]], %[[cst]] <{transpose}> : vector<4x16xf32>, !xegpu.tensor_desc<16x4xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 4 : i64>>, vector<16xi1>
  //CHECK: %[[r23:.*]] = xegpu.create_nd_tdesc %[[alloc]][{{.*}}] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
  //CHECK: %[[r24:.*]] = xegpu.load_nd %[[r23]]  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
  //CHECK: %[[r25:.*]] = vector.bitcast %[[r24]] : vector<64xf32> to vector<128xf16>
  //CHECK: %[[r26:.*]] = vector.shape_cast %[[r25]] : vector<128xf16> to vector<8x16xf16>
  //CHECK: %[[r27:.*]] = xegpu.create_nd_tdesc %{{.*}}[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  //CHECK: xegpu.store_nd %[[r26]], %[[r27]]  : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>

}

// -----
func.func @test_transpose_16x16xf16(%arg0: memref<16x16xf16>, %arg1: memref<8x32xf16>) {
  %in = xegpu.create_nd_tdesc %arg0[0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %data = xegpu.load_nd %in : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %transpose = vector.transpose %data, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %cast = vector.shape_cast %transpose : vector<16x16xf16> to vector<8x32xf16>
  %out = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
  xegpu.store_nd %cast, %out : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

  //CHECK: %cst = arith.constant dense<true> : vector<16xi1>
  //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %{{.*}}[0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
  //CHECK: %[[r2:.*]] = vector.shape_cast %[[r1]] : vector<8x16x2xf16> to vector<256xf16>
  //CHECK: %[[r3:.*]] = vector.bitcast %[[r2]] : vector<256xf16> to vector<128xf32>
  //CHECK: %[[r4:.*]] = vector.shape_cast %[[r3]] : vector<128xf32> to vector<8x16xf32>
  //CHECK: %[[alloc:.*]] = memref.alloc() : memref<8192xf32, 3>
  //CHECK: %[[r22:.*]] = xegpu.create_tdesc %[[alloc]], %{{.*}} : memref<8192xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
  //CHECK: xegpu.store %[[r4]], %[[r22]], %[[cst]] <{transpose}> : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<16xi1>
  //CHECK: %[[r23:.*]] = xegpu.create_nd_tdesc %[[alloc]][{{.*}}] : memref<8192xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
  //CHECK: %[[r24:.*]] = xegpu.load_nd %[[r23]]  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
  //CHECK: %[[r25:.*]] = vector.bitcast %[[r24]] : vector<64xf32> to vector<128xf16>
  //CHECK: %[[r26:.*]] = vector.shape_cast %[[r25]] : vector<128xf16> to vector<8x16xf16>
  //CHECK: %[[r28:.*]] = xegpu.create_nd_tdesc %[[alloc]][{{.*}}] : memref<8192xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
  //CHECK: %[[r29:.*]] = xegpu.load_nd %[[r28]]  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
  //CHECK: %[[r30:.*]] = vector.bitcast %[[r29]] : vector<64xf32> to vector<128xf16>
  //CHECK: %[[r31:.*]] = vector.shape_cast %[[r30]] : vector<128xf16> to vector<8x16xf16>
  //CHECK: %[[r32:.*]] = vector.shuffle %[[r26]], %[[r31]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf16>, vector<8x16xf16>
  //CHECK: %[[r33:.*]] = vector.shape_cast %[[r32]] : vector<16x16xf16> to vector<8x32xf16>
  //CHECK: %[[r34:.*]] = xegpu.create_nd_tdesc %{{.*}}[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x32xf16>
  //CHECK: xegpu.store_nd %[[r33]], %[[r34]]  : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

  return
}

// -----
func.func @test_transpose_8x16xf32(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>) {
  %in = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  %data = xegpu.load_nd %in : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %transpose = vector.transpose %data, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
  %cast = vector.shape_cast %transpose : vector<16x8xf32> to vector<8x16xf32>
  %out = xegpu.create_nd_tdesc %arg1[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %cast, %out : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

  //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<16xi1>
  //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %{{.*}}[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]]  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  //CHECK: %[[alloc:.*]] = memref.alloc() : memref<8192xf32, 3>
  //CHECK: %[[r19:.*]] = xegpu.create_tdesc %[[alloc]], %{{.*}} : memref<8192xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
  //CHECK: xegpu.store %[[r1]], %[[r19]], %[[cst]] <{transpose}> : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<16xi1>
  //CHECK: %[[r20:.*]] = xegpu.create_nd_tdesc %[[alloc]][{{.*}}] : memref<8192xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
  //CHECK: %[[r21:.*]] = xegpu.load_nd %[[r20]]  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
  //CHECK: %[[r22:.*]] = vector.shape_cast %[[r21]] : vector<64xf32> to vector<8x8xf32>
  //CHECK: %[[r24:.*]] = xegpu.create_nd_tdesc %[[alloc]][{{.*}}] : memref<8192xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
  //CHECK: %[[r25:.*]] = xegpu.load_nd %[[r24]]  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
  //CHECK: %[[r26:.*]] = vector.shape_cast %[[r25]] : vector<64xf32> to vector<8x8xf32>
  //CHECK: %[[r27:.*]] = vector.shuffle %[[r22]], %[[r26]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x8xf32>, vector<8x8xf32>
  //CHECK: %[[r28:.*]] = vector.shape_cast %[[r27]] : vector<16x8xf32> to vector<8x16xf32>
  //CHECK: %[[r29:.*]] = xegpu.create_nd_tdesc %{{.*}}[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  //CHECK: xegpu.store_nd %[[r28]], %[[r29]]  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
//CHECK: func.func @test_transpose(%[[arg0:.*]]: memref<16x16xf16>, %[[arg1:.*]]: memref<8x32xf16>)
func.func @test_transpose(%arg0: memref<16x16xf16>, %arg1: memref<8x32xf16>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %id = gpu.subgroup_id : index
  %y = arith.muli %id, %c8 : index
  %in = xegpu.create_nd_tdesc %arg0[0, %y] : memref<16x16xf16> -> !xegpu.tensor_desc<16x8xf16>
  %data = xegpu.load_nd %in : !xegpu.tensor_desc<16x8xf16> -> vector<16x8xf16>
  %transposed = vector.transpose %data, [1, 0] : vector<16x8xf16> to vector<8x16xf16>
  %y2 = arith.muli %id, %c16 : index
  %out = xegpu.create_nd_tdesc %arg1[0, %y2]: memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  xegpu.store_nd %transposed, %out : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  return

  //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<8xi1>
  //CHECK: %[[cst_0:.*]] = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56]> : vector<8xindex>
  //CHECK: %[[c64:.*]] = arith.constant 64 : index
  //CHECK: %[[c8:.*]] = arith.constant 8 : index
  //CHECK: %[[c16:.*]] = arith.constant 16 : index
  //CHECK: %[[r0:.*]] = gpu.subgroup_id : index
  //CHECK: %[[r1:.*]] = arith.muli %[[r0]], %[[c8]] : index
  //CHECK: %[[r2:.*]] = xegpu.create_nd_tdesc %[[arg0]][0, %[[r1]]] : memref<16x16xf16> -> !xegpu.tensor_desc<16x8xf16>
  //CHECK: %[[r3:.*]] = xegpu.load_nd %[[r2]] <{packed}> : !xegpu.tensor_desc<16x8xf16> -> vector<8x8x2xf16>
  //CHECK: %[[r4:.*]] = vector.shape_cast %[[r3]] : vector<8x8x2xf16> to vector<128xf16>
  //CHECK: %[[r5:.*]] = vector.bitcast %[[r4]] : vector<128xf16> to vector<64xf32>
  //CHECK: %[[r6:.*]] = vector.shape_cast %[[r5]] : vector<64xf32> to vector<8x8xf32>
  //CHECK: %[[alloc:.*]] = memref.alloc() : memref<4096xf32, 3>
  //CHECK: %[[r7:.*]] = gpu.subgroup_id : index
  //CHECK: %[[r8:.*]] = arith.muli %[[r7]], %[[c64]] : index
  //CHECK: %[[r9:.*]] = vector.broadcast %[[r8]] : index to vector<8xindex>
  //CHECK: %[[r10:.*]] = arith.addi %[[r9]], %[[cst_0]] : vector<8xindex>
  //CHECK: %[[r11:.*]] = xegpu.create_tdesc %[[alloc]], %[[r10]] : memref<4096xf32, 3>, vector<8xindex> -> !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
  //CHECK: xegpu.store %[[r6]], %[[r11]], %[[cst]] <{transpose}> : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<8xi1>
  //CHECK: %[[r12:.*]] = xegpu.create_nd_tdesc %[[alloc]][%[[r8]]] : memref<4096xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>>
  //CHECK: %[[r13:.*]] = xegpu.load_nd %[[r12]]  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = false>> -> vector<64xf32>
  //CHECK: %[[r14:.*]] = vector.bitcast %[[r13]] : vector<64xf32> to vector<128xf16>
  //CHECK: %[[r15:.*]] = vector.shape_cast %[[r14]] : vector<128xf16> to vector<8x16xf16>
  //CHECK: %[[r16:.*]] = arith.muli %[[r0]], %[[c16]] : index
  //CHECK: %[[r17:.*]] = xegpu.create_nd_tdesc %[[arg1]][0, %[[r16]]] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  //CHECK: xegpu.store_nd %[[r15]], %[[r17]]  : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
}

// -----
  //CHECK: func.func @test_load_update_nd_offset(%[[arg0:.*]]: memref<16x16xf16>, %[[arg1:.*]]: memref<16x32xf16>, %[[arg2:.*]]: memref<16x32xf32>)
  func.func @test_load_update_nd_offset(%arg0: memref<16x16xf16>, %arg1: memref<16x32xf16>, %arg2: memref<16x32xf32>) {
    %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    %1 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16>
    %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    //CHECK: %{{.*}} = xegpu.load_nd %{{.*}} <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %4 = vector.transpose %3, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %5 = vector.shape_cast %4 {packed} : vector<16x16xf16> to vector<256xf16>
    %6 = vector.shuffle %5, %5 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %7 = vector.shape_cast %6 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %8 = xegpu.dpas %2, %7 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    %9 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %8, %9  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    %10 = xegpu.update_nd_offset %0, [8, 0] : !xegpu.tensor_desc<8x16xf16>
    %11 = xegpu.update_nd_offset %1, [0, 16] : !xegpu.tensor_desc<16x16xf16>
    %12 = xegpu.load_nd %10  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    //CHECK: %{{.*}} = xegpu.load_nd %{{.*}} <{transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    %13 = xegpu.load_nd %11  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %14 = vector.transpose %13, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %15 = vector.shape_cast %14 {packed} : vector<16x16xf16> to vector<256xf16>
    %16 = vector.shuffle %15, %15 [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
    %17 = vector.shape_cast %16 {packed} : vector<256xf16> to vector<8x16x2xf16>
    %18 = xegpu.dpas %12, %17 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    %19 = xegpu.update_nd_offset %9, [0, 16] : !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %18, %19  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    return
  }

// -----

// Here the same xegpu.create_nd_tdesc is used four times as iter operand in the
// same scf.for.
//CHECK: gpu.func @add_bf16_EC831D15_4614D61C_861
#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (16)>
#map2 = affine_map<() -> (2)>
module attributes {gpu.container_module} {
  func.func @add_bf16_EC831D15_4614D61C_861_entry(%arg0: memref<2x16x384x384xbf16>, %arg1: memref<2x1x384x384xbf16>, %arg2: memref<2x16x384x384xbf16>) attributes {gemm_tiles_x = dense<1> : vector<4xi64>, gemm_tiles_y = dense<[2, 64, 24, 48]> : vector<4xi64>, habana_runner.num_inputs = 2 : i64, habana_runner.tests = [{inputs = [dense<1.000000e+00> : tensor<2x16x384x384xbf16>, dense<2.000000e+00> : tensor<2x1x384x384xbf16>], outputs = [dense<3.000000e+00> : tensor<2x16x384x384xbf16>]}], physical_nd_range = dense<[2, 24]> : vector<2xi64>, region_partition = 1 : i64, region_size = 24 : i64, syn.fusion_successful, syn.gemm_pipeline, syn.large_grf = false, syn.tensor_signature = (tensor<2x16x384x384xbf16>, tensor<2x1x384x384xbf16>) -> tensor<2x16x384x384xbf16>, synFusionGenOps = 8 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 236196.20000000001 : f64} {
    %c48 = arith.constant 48 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @add_bf16_EC831D15_4614D61C_861::@add_bf16_EC831D15_4614D61C_861 blocks in (%c48, %c1, %c1) threads in (%c48, %c1, %c1)  args(%arg0 : memref<2x16x384x384xbf16>, %arg1 : memref<2x1x384x384xbf16>, %arg2 : memref<2x16x384x384xbf16>)
    return
  }
  gpu.module @add_bf16_EC831D15_4614D61C_861 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @add_bf16_EC831D15_4614D61C_861(%arg0: memref<2x16x384x384xbf16>, %arg1: memref<2x1x384x384xbf16>, %arg2: memref<2x16x384x384xbf16>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 48, 1, 1>, known_grid_size = array<i32: 48, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c12 = arith.constant 12 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c192 = arith.constant 192 : index
      %c96 = arith.constant 96 : index
      %c16 = arith.constant 16 : index
      %c3 = arith.constant 3 : index
      %c32 = arith.constant 32 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      %c24 = arith.constant 24 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.shrsi %block_id_x, %c5 : index
      %1 = arith.addi %0, %c1 : index
      %2 = arith.muli %1, %c8 : index
      %3 = arith.addi %block_id_x, %2 : index
      %4 = arith.shrsi %3, %c5 : index
      %5 = arith.muli %4, %c24 : index
      %6 = arith.subi %block_id_x, %5 : index
      %7 = arith.shrsi %4, %c1 : index
      %8 = arith.muli %7, %c2 : index
      %9 = arith.subi %4, %8 : index
      %10 = arith.shrsi %9, %c1 : index
      %11 = arith.muli %10, %c2 : index
      %12 = arith.subi %9, %11 : index
      %13 = arith.remsi %6, %c24 : index
      %14 = arith.remsi %13, %c24 : index
      %15 = arith.remsi %thread_id_x, %c12 : index
      %16 = arith.divsi %thread_id_x, %c12 : index
      %17 = arith.shrsi %16, %c2 : index
      %18 = arith.muli %17, %c4 : index
      %19 = arith.subi %16, %18 : index
      %20 = arith.remsi %15, %c12 : index
      %21 = arith.muli %20, %c32 overflow<nsw> : index
      %22 = arith.muli %12, %c96 overflow<nsw> : index
      %23 = arith.muli %14, %c4 overflow<nsw> : index
      %24 = arith.addi %22, %23 : index
      %25 = arith.addi %24, %19 : index
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %26 = xegpu.create_nd_tdesc %arg1[%arg3, %c0, %25, %21] : memref<2x1x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
        scf.for %arg4 = %c0 to %c16 step %c4 {
          %27 = arith.addi %arg4, %c3 : index
          %28 = arith.addi %arg4, %c2 : index
          %29 = arith.addi %arg4, %c1 : index
          %30 = xegpu.create_nd_tdesc %arg0[%arg3, %27, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %31 = xegpu.create_nd_tdesc %arg0[%arg3, %28, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %32 = xegpu.create_nd_tdesc %arg0[%arg3, %29, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %33 = xegpu.create_nd_tdesc %arg0[%arg3, %arg4, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %34 = xegpu.create_nd_tdesc %arg2[%arg3, %27, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %35 = xegpu.create_nd_tdesc %arg2[%arg3, %28, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %36 = xegpu.create_nd_tdesc %arg2[%arg3, %29, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %37 = xegpu.create_nd_tdesc %arg2[%arg3, %arg4, %25, %21] : memref<2x16x384x384xbf16> -> !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          %38:12 = scf.for %arg5 = %c0 to %c2 step %c1 iter_args(%arg6 = %26, %arg7 = %26, %arg8 = %26, %arg9 = %26, %arg10 = %30, %arg11 = %31, %arg12 = %32, %arg13 = %33, %arg14 = %34, %arg15 = %35, %arg16 = %36, %arg17 = %37) -> (!xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>) {
            %39 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %40 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %41 = xegpu.load_nd %arg8 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %42 = xegpu.load_nd %arg9 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %43 = xegpu.load_nd %arg10 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %44 = xegpu.load_nd %arg11 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %45 = xegpu.load_nd %arg12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %46 = xegpu.load_nd %arg13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>> -> vector<1x32xbf16>
            %47 = arith.addf %43, %39 : vector<1x32xbf16>
            %48 = arith.addf %44, %40 : vector<1x32xbf16>
            %49 = arith.addf %45, %41 : vector<1x32xbf16>
            %50 = arith.addf %46, %42 : vector<1x32xbf16>
            xegpu.store_nd %47, %arg14 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x32xbf16>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            xegpu.store_nd %48, %arg15 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x32xbf16>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            xegpu.store_nd %49, %arg16 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x32xbf16>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            xegpu.store_nd %50, %arg17 <{l1_hint = #xegpu.cache_hint<uncached>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<1x32xbf16>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %51 = xegpu.update_nd_offset %arg6, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %52 = xegpu.update_nd_offset %arg7, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %53 = xegpu.update_nd_offset %arg8, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %54 = xegpu.update_nd_offset %arg9, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %55 = xegpu.update_nd_offset %arg10, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %56 = xegpu.update_nd_offset %arg11, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %57 = xegpu.update_nd_offset %arg12, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %58 = xegpu.update_nd_offset %arg13, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %59 = xegpu.update_nd_offset %arg14, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %60 = xegpu.update_nd_offset %arg15, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %61 = xegpu.update_nd_offset %arg16, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            %62 = xegpu.update_nd_offset %arg17, [%c192, %c0] : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
            scf.yield %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62 : !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<1x32xbf16, #xegpu.block_tdesc_attr<array_length = 1 : i64, boundary_check = true>>
          }
        } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 4 : index, syn.mm_dim = 1 : i64, syn.parall_level = 2 : i64, upperBoundMap = #map1}
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, syn.mm_dim = 1 : i64, syn.parall_level = 2 : i64, upperBoundMap = #map2}
      gpu.return
    }
  }
}
