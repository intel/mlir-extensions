// RUN: imex-opt --xetile-tiling %s | FileCheck %s


// CHECK-LABEL: func @test_gemm({{.*}}) {
func.func @test_gemm(%A: memref<4096x4096xi8>, %B: memref<4096x4096xi8>, %C: memref<4096x4096xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c4096 = arith.constant 4096 : index
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %m = arith.muli %block_id_x, %c128 : index
  %n = arith.muli %block_id_y, %c256 : index
  // intialize C tile and load it
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<4096x4096xi32> -> !xetile.tile<16x16x8x16xi32>
  %c_init_tile = xetile.init_tile %C[%m, %n] : memref<4096x4096xi32> -> !xetile.tile<128x256xi32>
  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<16x16x8x16xi32> -> vector<16x16x8x16xi32>
  %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x256xi32> -> vector<128x256xi32>
  // initalize A and B tiles
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<4096x4096xi8> -> !xetile.tile<16x16x8x16xi8>
  %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<4096x4096xi8> -> !xetile.tile<128x256xi8>
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<4096x4096xi8> -> !xetile.tile<16x16x16x16xi8>
  %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xi8> -> !xetile.tile<256x256xi8>
  // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
  // CHECK: (!xetile.tile<16x16x8x16xi8>, !xetile.tile<16x16x16x16xi8>, vector<16x16x8x16xi32>)
  %out:3 = scf.for %k = %c0 to %c4096 step %c256
    iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
    -> (!xetile.tile<128x256xi8>, !xetile.tile<256x256xi8>, vector<128x256xi32>) {

    // load A and B tiles
    // CHECK: xetile.load_tile
    // CHECK-SAME: !xetile.tile<16x16x8x16xi8> -> vector<16x16x8x16xi8>
    %a_value = xetile.load_tile %a_tile : !xetile.tile<128x256xi8> -> vector<128x256xi8>
    // CHECK: xetile.load_tile
    // CHECK-SAME: !xetile.tile<16x16x16x16xi8> -> vector<16x16x16x16xi8>
    %b_value = xetile.load_tile %b_tile : !xetile.tile<256x256xi8> -> vector<256x256xi8>
    // perform dpas and accumulate
    // CHECK: xetile.tile_mma
    // CHECK-SAME: vector<16x16x8x16xi8>,  vector<16x16x16x16xi8>,  vector<16x16x8x16xi32> -> vector<16x16x8x16xi32>
    %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
      : vector<128x256xi8>, vector<256x256xi8>, vector<128x256xi32> -> vector<128x256xi32>
    // update the offsets for A and B tiles
    // CHECK: xetile.update_tile_offset
    // CHECK-SAME: !xetile.tile<16x16x8x16xi8>, index, index -> !xetile.tile<16x16x8x16xi8>
    %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c256]
      : !xetile.tile<128x256xi8>, index, index -> !xetile.tile<128x256xi8>
    // CHECK: xetile.update_tile_offset
    // CHECK-SAME: !xetile.tile<16x16x16x16xi8>, index, index -> !xetile.tile<16x16x16x16xi8>
    %b_next_tile = xetile.update_tile_offset %b_tile, [%c256, %c0]
      : !xetile.tile<256x256xi8>, index, index -> !xetile.tile<256x256xi8>
    // partial C tile result
    // CHECK: !xetile.tile<16x16x8x16xi8>, !xetile.tile<16x16x16x16xi8>, vector<16x16x8x16xi32>
    scf.yield %a_next_tile, %b_next_tile, %c_new_value
      : !xetile.tile<128x256xi8>, !xetile.tile<256x256xi8>, vector<128x256xi32>
  }
  // store the final accumulated C tile result back to memory
  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<16x16x8x16xi32>, !xetile.tile<16x16x8x16xi32>
  xetile.store_tile %out#2, %c_init_tile : vector<128x256xi32>, !xetile.tile<128x256xi32>
  return
}
