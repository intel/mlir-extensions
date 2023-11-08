// RUN: imex-opt --xetile-tiling %s | FileCheck %s


// CHECK-LABEL: func @test_gemm({{.*}}) {
func.func @test_gemm(%A: memref<1024x1024xi8>, %B: memref<1024x1024xi8>, %C: memref<1024x1024xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // %c8 = arith.constant 8 : index
  // %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %m = arith.muli %block_id_x, %c64 : index
  %n = arith.muli %block_id_y, %c64 : index
  // intialize C tile and load it
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xi32> -> !xetile.tile<8x4x8x16xi32>
  %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<64x64xi32>
  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<8x4x8x16xi32> -> vector<8x4x8x16xi32>
  %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x64xi32> -> vector<64x64xi32>
  // initalize A and B tiles
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xi8> -> !xetile.tile<8x4x8x16xi8>
  %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xi8> -> !xetile.tile<64x64xi8>
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xi8> -> !xetile.tile<4x4x16x16xi8>
  %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xi8> -> !xetile.tile<64x64xi8>
  // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
  // CHECK: scf.for
  // CHECK-SAME: !xetile.tile<8x4x8x16xi8>, !xetile.tile<4x4x16x16xi8>, vector<8x4x8x16xi32>
  %out:3 = scf.for %k = %c0 to %c1024 step %c64
    iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
    -> (!xetile.tile<64x64xi8>, !xetile.tile<64x64xi8>, vector<64x64xi32>) {

    // load A and B tiles
    // CHECK: xetile.load_tile
    // CHECK-SAME: !xetile.tile<8x4x8x16xi8> -> vector<8x4x8x16xi8>
    %a_value = xetile.load_tile %a_tile : !xetile.tile<64x64xi8> -> vector<64x64xi8>
    // CHECK: xetile.load_tile
    // CHECK-SAME: !xetile.tile<4x4x16x16xi8> -> vector<4x4x16x16xi8>
    %b_value = xetile.load_tile %b_tile : !xetile.tile<64x64xi8> -> vector<64x64xi8>
    // perform dpas and accumulate
    // CHECK: xetile.tile_mma
    // CHECK-SAME: vector<8x4x8x16xi8>,  vector<4x4x16x16xi8>,  vector<8x4x8x16xi32> -> vector<8x4x8x16xi32>
    %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<64x64xi8>, vector<64x64xi8>, vector<64x64xi32> -> vector<64x64xi32>
    // update the offsets for A and B tiles
    // CHECK: xetile.update_tile_offset
    // CHECK-SAME: !xetile.tile<8x4x8x16xi8>, index, index -> !xetile.tile<8x4x8x16xi8>
    %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64]
      : !xetile.tile<64x64xi8>, index, index -> !xetile.tile<64x64xi8>
    // CHECK: xetile.update_tile_offset
    // CHECK-SAME: !xetile.tile<4x4x16x16xi8>, index, index -> !xetile.tile<4x4x16x16xi8>
    %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0]
      : !xetile.tile<64x64xi8>, index, index -> !xetile.tile<64x64xi8>
    // partial C tile result
    // CHECK: scf.yield
    // CHECK-SAME: !xetile.tile<8x4x8x16xi8>, !xetile.tile<4x4x16x16xi8>, vector<8x4x8x16xi32>
    scf.yield %a_next_tile, %b_next_tile, %c_new_value
      : !xetile.tile<64x64xi8>, !xetile.tile<64x64xi8>, vector<64x64xi32>
  }
  // store the final accumulated C tile result back to memory
  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<8x4x8x16xi32>, !xetile.tile<8x4x8x16xi32>
  xetile.store_tile %out#2, %c_init_tile {innner_blocks = [8, 16]}: vector<64x64xi32>, !xetile.tile<64x64xi32>
  return
}
