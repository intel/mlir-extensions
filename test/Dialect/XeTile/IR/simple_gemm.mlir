// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

#sg_map_a = #xetile.sg_map< wi_layout = [2, 8], wi_data = [1, 2]>
#wg_map_a = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
#tile_attr_a = #xetile.tile_attr<wg_map = #wg_map_a, sg_map = #sg_map_a>

#sg_map_b = #xetile.sg_map< wi_layout = [1, 16], wi_data = [1, 1]>
#wg_map_b = #xetile.wg_map<sg_layout = [2, 2], sg_data = [128, 32]>
#tile_attr_b = #xetile.tile_attr<wg_map = #wg_map_b, sg_map = #sg_map_b>

#sg_map_c = #xetile.sg_map< wi_layout = [1, 16], wi_data = [1, 1]>
#wg_map_c = #xetile.wg_map<sg_layout = [4, 4], sg_data = [32, 32]>
#tile_attr_c = #xetile.tile_attr<wg_map = #wg_map_c, sg_map = #sg_map_c>

// CHECK-LABEL: func @test_gemm({{.*}}) {
func.func @test_gemm(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // %c8 = arith.constant 8 : index
  // %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %m = arith.muli %block_id_x, %c128 : index
  %n = arith.muli %block_id_y, %c128 : index
  // intialize C tile and load it
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf32> -> !xetile.tile<128x128xf32, #xetile.tile_attr<sg_map = <wi_layout = [1, 16],
  // CHECK-SAME: wi_data = [1, 1]>, wg_map = <sg_layout = [4, 4], sg_data = [32, 32]>>>
  %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<128x128xf32, #tile_attr_c>
  // CHECK:  xetile.load_tile
  // CHECK-SAME: : !xetile.tile<128x128xf32, #xetile.tile_attr<sg_map =
  // CHECK-SAME: <wi_layout = [1, 16], wi_data = [1, 1]>, wg_map = <sg_layout = [4, 4], sg_data = [32, 32]>>> -> vector<128x128xf32>
  %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<128x128xf32, #tile_attr_c> -> vector<128x128xf32>
  // initalize A and B tiles
  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map = <wi_layout = [2, 8],
  // CHECK-SAME: wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #tile_attr_a>
  // CHECK:  xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map = <wi_layout = [1, 16],
  // CHECK-SAME: wi_data = [1, 1]>, wg_map = <sg_layout = [2, 2], sg_data = [128, 32]>>>
  %b_init_tile = xetile.init_tile %B[%c0, %n]  : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #tile_attr_b>
  // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
  %out:3 = scf.for %k = %c0 to %c1024 step %c128
    iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
    -> (!xetile.tile<128x128xf16, #tile_attr_a>, !xetile.tile<128x128xf16, #tile_attr_b>, vector<128x128xf32>) {

    // load A and B tiles
    // CHECK: xetile.load_tile
    // CHECK-SAME: : !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map
    // CHECK-SAME: = <wi_layout = [2, 8], wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>> -> vector<128x128xf16>
    %a_value = xetile.load_tile %a_tile : !xetile.tile<128x128xf16, #tile_attr_a> -> vector<128x128xf16>
    // CHECK:  xetile.load_tile
    // CHECK-SAME: : !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map
    // CHECK-SAME: = <wi_layout = [1, 16], wi_data = [1, 1]>, wg_map = <sg_layout = [2, 2], sg_data = [128, 32]>>> -> vector<128x128xf16>
    %b_value = xetile.load_tile %b_tile : !xetile.tile<128x128xf16,  #tile_attr_b> -> vector<128x128xf16>
    // perform dpas and accumulate
    // CHECK: xetile.tile_mma
    // CHECK-SAME: vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>
    %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>
    // update the offsets for A and B tiles
    // CHECK:  xetile.update_tile_offset
    // CHECK-SAME: !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map = <wi_layout = [2, 8], wi_data = [1, 2]>,
    // CHECK-SAME: wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
    %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c128] : !xetile.tile<128x128xf16,  #tile_attr_a>
    // CHECK: xetile.update_tile_offset
    // CHECK-SAME: !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map = <wi_layout = [1, 16], wi_data = [1, 1]>,
    // CHECK-SAME: wg_map = <sg_layout = [2, 2], sg_data = [128, 32]>>>
    %b_next_tile = xetile.update_tile_offset %b_tile, [%c128, %c0] : !xetile.tile<128x128xf16, #tile_attr_b>
    // partial C tile result
    scf.yield %a_next_tile, %b_next_tile, %c_new_value
      : !xetile.tile<128x128xf16, #tile_attr_a>, !xetile.tile<128x128xf16,  #tile_attr_b>, vector<128x128xf32>
  }
  // store the final accumulated C tile result back to memory
  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<128x128xf32>, !xetile.tile<128x128xf32, #xetile.tile_attr<sg_map = <wi_layout = [1, 16],
  // CHECK-SAME: wi_data = [1, 1]>, wg_map = <sg_layout = [4, 4], sg_data = [32, 32]>>>
  xetile.store_tile %out#2, %c_init_tile : vector<128x128xf32>, !xetile.tile<128x128xf32, #tile_attr_c>
  return
}
