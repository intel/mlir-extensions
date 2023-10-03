// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s


// CHECK-LABEL: func @test_gemm({{.*}}) {
func.func @test_gemm(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %m = arith.muli %block_id_x, %c8 : index
  %n = arith.muli %block_id_y, %c16 : index
  // intialize C tile and load it
  // CHECK : xetile.init_tile
  // CHECK-SAME : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
  %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<8x16xf32>
  // CHECK : xetile.load_tile
  // CHECK-SAME : !xetile.tile<8x16xf32> -> vector<8x16xf32>
  %c_init_value = xetile.load_tile %c_init_tile  : !xetile.tile<8x16xf32> -> vector<8x16xf32>
  // initalize A and B tiles
  // CHECK : xetile.init_tile
  // CHECK-SAME : memref<1024x1024xf16> -> !xetile.tile<8x16xf16>
  %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<8x16xf16>
  // CHECK : xetile.init_tile
  // CHECK-SAME : memref<1024x1024xf16> -> !xetile.tile<16x16xf16>
  %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16> -> !xetile.tile<16x16xf16>
  // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
  %out:3 = scf.for %k = %c0 to %c1024 step %c16
    iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
    -> (!xetile.tile<8x16xf16>, !xetile.tile<16x16xf16>, vector<8x16xf32>) {

    // load A and B tiles
    // CHECK : xetile.load_tile
    // CHECK-SAME : !xetile.tile<8x16xf16> -> vector<8x16xf16>
    %a_value = xetile.load_tile %a_tile  : !xetile.tile<8x16xf16> -> vector<8x16xf16>
    // CHECK : xetile.load_tile
    // CHECK-SAME : !xetile.tile<16x16xf16> -> vector<16x16xf16>
    %b_value = xetile.load_tile %b_tile  : !xetile.tile<16x16xf16> -> vector<16x16xf16>
    // perform dpas and accumulate
    // CHECK : xetile.tile_mma
    // CHECK-SAME : (vector<8x16xf16>,  vector<16x16xf16>,  vector<8x16xf32>) -> vector<8x16xf32>
    %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
      : (vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32>) -> vector<8x16xf32>
    // update the offsets for A and B tiles
    // CHECK : xetile.update_tile_offset
    // CHECK-SAME : (!xetile.tile<8x16xf16>,  index,  index) -> !xetile.tile<8x16xf16>
    %a_next_tile = xetile.update_tile_offset %a_tile, %c0, %c16
      : (!xetile.tile<8x16xf16>, index, index) -> !xetile.tile<8x16xf16>
    // CHECK : xetile.update_tile_offset
    // CHECK-SAME : (!xetile.tile<16x16xf16>,  index,  index) -> !xetile.tile<16x16xf16>
    %b_next_tile = xetile.update_tile_offset %b_tile, %c16, %c0
      : (!xetile.tile<16x16xf16>, index, index) -> !xetile.tile<16x16xf16>
    // partial C tile result
    scf.yield %a_next_tile, %b_next_tile, %c_new_value
      : !xetile.tile<8x16xf16>, !xetile.tile<16x16xf16>, vector<8x16xf32>
  }
  // store the final accumulated C tile result back to memory
  // CHECK : xetile.store_tile
  // CHECK-SAME : (vector<8x16xf32>, !xetile.tile<8x16xf32>)
  xetile.store_tile %out#2, %c_init_tile: (vector<8x16xf32>, !xetile.tile<8x16xf32>)
  return
}
