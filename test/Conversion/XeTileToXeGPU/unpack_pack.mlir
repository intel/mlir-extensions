// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse -verify-diagnostics %s -o - | FileCheck %s

gpu.module @unpack_pack_non_compatible {
  gpu.func @unpack_pack_non_compatible(%arg0: memref<88x32xf16>, %arg1: memref<88x32xf16>) {
    %c0 = arith.constant 0 : index
    %0 = xetile.init_tile %arg0[%c0, %c0] : memref<88x32xf16> -> !xetile.tile<88x32xf16, #xetile.tile_attr<inner_blocks = [22, 16]>>
    %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<88x32xf16, #xetile.tile_attr<inner_blocks = [22, 16]>> -> vector<4x2x22x16xf16>
    %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 22, 16>}  : vector<4x2x22x16xf16> -> vector<88x32xf16>
    %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 8, 16>}  : vector<88x32xf16> -> vector<11x2x8x16xf16>
    %4 = xetile.init_tile %arg1[%c0, %c0] : memref<88x32xf16> -> !xetile.tile<88x32xf16, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %3,  %4 : vector<11x2x8x16xf16>, !xetile.tile<88x32xf16, #xetile.tile_attr<inner_blocks = [8, 16]>>
    gpu.return
  }
}

// Since the unpack/pack are lowered independently, the unpack op is lowered to
// a series of shuffles followed by a shape cast and the pack op is lowered to
// a series of extract_strided_slice.

// CHECK: gpu.func @unpack_pack_non_compatible
// CHECK-COUNT-4:  vector.shuffle {{.*}} vector<22x16xf16>
// CHECK:          vector.shape_cast {{.*}} vector<88x32xf16>
// CHECK-COUNT-22: vector.extract_strided_slice {{.*}} vector<8x16xf16>

// -----

gpu.module @unpack_pack_compatible {
  gpu.func @unpack_pack_compatible(%arg0: memref<64x32xf16>, %arg1: memref<64x32xf16>) {
    %c0 = arith.constant 0 : index
    %0 = xetile.init_tile %arg0[%c0, %c0] : memref<64x32xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<2x2x32x16xf16>
    %2 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 16>}  : vector<2x2x32x16xf16> -> vector<64x32xf16>
    %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 8, 16>}  : vector<64x32xf16> -> vector<8x2x8x16xf16>
    %4 = xetile.init_tile %arg1[%c0, %c0] : memref<64x32xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %3,  %4 : vector<8x2x8x16xf16>, !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 16]>>
    gpu.return
  }
}

// Since the unpack/pack are lowered jointly, there will be a series of
// extract_strided_slice from the unpack inner blocks (32x16) to the pack inner
// blocks (8x16).

// CHECK: gpu.func @unpack_pack_compatible
// CHECK-COUNT-16: vector.extract_strided_slice {{.*}} vector<32x16xf16> to vector<8x16xf16>
