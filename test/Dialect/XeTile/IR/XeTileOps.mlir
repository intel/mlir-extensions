// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_init_tile({{.*}}) {
func.func @test_init_tile(%src: memref<1024x1024xf32>) {

  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf32> -> !xetile.tile<32x64xf32>
  %1 = xetile.init_tile %src[8, 16] : memref<1024x1024xf32> -> !xetile.tile<32x64xf32>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf32> -> !xetile.tile<32x64xf32>
  %2 = xetile.init_tile %src[%c128, %c256] : memref<1024x1024xf32> -> !xetile.tile<32x64xf32>

  return
}

// CHECK-LABEL: func @test_init_coop_tile({{.*}}) {
func.func @test_init_coop_tile(%src: !xetile.tile<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 64 : index

  // CHECK: xetile.init_coop_tile
  // CHECK-SAME: !xetile.tile<64x64xf32>, index, index -> !xetile.tile<8x8xf32>
  %1 = xetile.init_coop_tile %src, %c0, %c1
    : !xetile.tile<64x64xf32>, index, index -> !xetile.tile<8x8xf32>

  return
}


// CHECK-LABEL: func @test_load_tile({{.*}}) {
func.func @test_load_tile(%src: !xetile.tile<64x32xf32>) {
  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<64x32xf32> -> vector<64x32xf32>
  %1 = xetile.load_tile %src : !xetile.tile<64x32xf32> -> vector<64x32xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: inner_blocks = [8, 16] : !xetile.tile<64x32xf32> -> vector<8x2x8x16xf32>
  %2 = xetile.load_tile %src inner_blocks = [8, 16] : !xetile.tile<64x32xf32> -> vector<8x2x8x16xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: TRANSPOSE = true : !xetile.tile<64x32xf32> -> vector<32x64xf32>
  %3 = xetile.load_tile %src TRANSPOSE = true : !xetile.tile<64x32xf32> -> vector<32x64xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: inner_blocks = [8, 16] TRANSPOSE = true : !xetile.tile<64x32xf32> -> vector<2x8x16x8xf32>
  %4 = xetile.load_tile %src inner_blocks = [8, 16] TRANSPOSE = true : !xetile.tile<64x32xf32> -> vector<2x8x16x8xf32>

  return
}

// CHECK-LABEL: func @test_store_tile({{.*}}) {
func.func @test_store_tile(%value1 : vector<64x32xf32>,
  %value2 : vector<8x2x8x16xf32>, %dst: !xetile.tile<64x32xf32>) {

  // CHECK: xetile.store_tile
  // CHECK-SAME: (!xetile.tile<64x32xf32>,  vector<64x32xf32>)
  xetile.store_tile %dst, %value1 : (!xetile.tile<64x32xf32>,  vector<64x32xf32>)

  // CHECK: xetile.store_tile
  // CHECK-SAME: inner_blocks = [8, 16] : (!xetile.tile<64x32xf32>, vector<8x2x8x16xf32>)
  xetile.store_tile %dst, %value2 inner_blocks = [8,16] : (!xetile.tile<64x32xf32>,  vector<8x2x8x16xf32>)

  return
}

// CHECK-LABEL: func @test_coop_prefetch_tile({{.*}}) {
func.func @test_coop_prefetch_tile(%src: !xetile.tile<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 64 : index

  // CHECK: xetile.init_coop_tile
  // CHECK-SAME: !xetile.tile<64x64xf32>, index, index -> !xetile.tile<8x8xf32>
  %1 = xetile.init_coop_tile %src, %c0, %c1
    : !xetile.tile<64x64xf32>, index, index -> !xetile.tile<8x8xf32>

  // CHECK: xetile.prefetch_tile
  // CHECK-SAME: (!xetile.tile<8x8xf32>)
  xetile.prefetch_tile %1 : (!xetile.tile<8x8xf32>)

  return
}


// CHECK-LABEL: func @test_tile_mma({{.*}}) {
func.func @test_tile_mma(%a: !xetile.tile<64x32xf32>, %b: !xetile.tile<32x128xf32>, %c : !xetile.tile<64x128xf32>) {

  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<64x32xf32> -> vector<64x32xf32>
  %a_vec = xetile.load_tile %a : !xetile.tile<64x32xf32> -> vector<64x32xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<32x128xf32> -> vector<32x128xf32>
  %b_vec = xetile.load_tile %b : !xetile.tile<32x128xf32> -> vector<32x128xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<64x128xf32> -> vector<64x128xf32>
  %c_vec = xetile.load_tile %c : !xetile.tile<64x128xf32> -> vector<64x128xf32>

  // CHECK: xetile.tile_mma
  // CHECK-SAME: (vector<64x32xf32>,  vector<32x128xf32>,  vector<64x128xf32>) -> vector<64x128xf32>
  %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec
    : (vector<64x32xf32>, vector<32x128xf32>, vector<64x128xf32>) -> vector<64x128xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: inner_blocks = [8, 8] : !xetile.tile<64x32xf32> -> vector<8x4x8x8xf32>
  %a_vec_1 = xetile.load_tile %a inner_blocks = [8, 8]
    : !xetile.tile<64x32xf32> -> vector<8x4x8x8xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: inner_blocks = [8, 16] : !xetile.tile<32x128xf32> -> vector<4x8x8x16xf32>
  %b_vec_1 = xetile.load_tile %b inner_blocks = [8,16]
    : !xetile.tile<32x128xf32> -> vector<4x8x8x16xf32>

  // CHECK: xetile.load_tile
  // CHECK-SAME: inner_blocks = [8, 16] : !xetile.tile<64x128xf32> -> vector<8x8x8x16xf32>
  %c_vec_1 = xetile.load_tile %c inner_blocks = [8,16]
    : !xetile.tile<64x128xf32> -> vector<8x8x8x16xf32>

  // CHECK: xetile.tile_mma
  // CHECK-SAME: (vector<8x4x8x8xf32>,  vector<4x8x8x16xf32>, vector<8x8x8x16xf32>) -> vector<8x8x8x16xf32>
  %c_new_1 = xetile.tile_mma %a_vec_1, %b_vec_1, %c_vec_1 a_inner_blocks=[8,8] b_inner_blocks=[8,16]
    : (vector<8x4x8x8xf32>, vector<4x8x8x16xf32>, vector<8x8x8x16xf32>) -> vector<8x8x8x16xf32>

  return
}


// CHECK-LABEL: func @test_update_tile_offset({{.*}}) {
func.func @test_update_tile_offset(%tile: !xetile.tile<32x32xf32>) {

  %offset_x = arith.constant 0 : index
  %offset_y = arith.constant 96 : index

  // CHECK: xetile.update_tile_offset
  // CHECK-SAME: (!xetile.tile<32x32xf32>,  index,  index)
  xetile.update_tile_offset %tile, %offset_x, %offset_y : (!xetile.tile<32x32xf32>, index, index)

  return
}
