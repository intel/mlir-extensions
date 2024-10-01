// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse -verify-diagnostics %s -o -| FileCheck %s

gpu.module @test_kernel {
  gpu.func @create_mask(%arg0: vector<32x32xf16>, %arg1: vector<32x32xf16>, %arg2: memref<32x32xf16>) {
    %c32 = arith.constant 32 : index
    %c20 = arith.constant 20 : index
    %0 = vector.create_mask %c32, %c20, %c32, %c20 : vector<32x2x1x16xi1>
    %1 = xetile.tile_pack %arg0 {inner_blocks = array<i64: 1, 16>}: vector<32x32xf16> -> vector<32x2x1x16xf16>
    %2 = xetile.tile_pack %arg1 {inner_blocks = array<i64: 1, 16>}: vector<32x32xf16> -> vector<32x2x1x16xf16>
    %3 = arith.select %0, %1, %2 : vector<32x2x1x16xi1>, vector<32x2x1x16xf16>
    %4 = xetile.tile_unpack %3 {inner_blocks = array<i64: 1, 16>}: vector<32x2x1x16xf16> -> vector<32x32xf16>
    %5 = xetile.init_tile %arg2[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %6 = xetile.tile_pack %4 {inner_blocks = array<i64: 8, 32>}: vector<32x32xf16> -> vector<4x1x8x32xf16>
    xetile.store_tile %6,  %5 : vector<4x1x8x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    gpu.return
  }
}

// convert-xetile-to-xegpu generates 64 create_mask ops. But since all the rows
// have the same masks, cse leaves only two masks corresponding to a row.

//CHECK: gpu.func @create_mask
//CHECK-DAG: %[[C1:.*]] = arith.constant 1
//CHECK-DAG: %[[C16:.*]] = arith.constant 16
//CHECK-DAG: %[[MASK1_VAL:.*]] = arith.constant 20
//CHECK:     %[[MASK2_VAL:.*]] = arith.subi %[[MASK1_VAL]], %c16
//CHECK:     %[[MASK1:.*]] = vector.create_mask %[[C1]], %[[MASK1_VAL]] : vector<1x16xi1>
//CHECK:     %[[MASK2:.*]] = vector.create_mask %[[C1]], %[[MASK2_VAL]] : vector<1x16xi1>
//CHECK-NOT: %[[MASK2:.*]] = vector.create_mask
//CHECK:     arith.select %[[MASK1]], {{.*}} : vector<1x16xi1>, vector<1x16xf16>
//CHECK:     arith.select %[[MASK2]], {{.*}} : vector<1x16xi1>, vector<1x16xf16>

// -----

gpu.module @test_kernel_2 {
  gpu.func @create_mask_2(%arg0: vector<32x32xf16>, %arg1: vector<32x32xf16>, %arg2: memref<32x32xf16>) {
    %c32 = arith.constant 32 : index
    %c20 = arith.constant 20 : index
    %0 = vector.create_mask %c20, %c32, %c20, %c32 : vector<32x2x1x16xi1>
    %1 = xetile.tile_pack %arg0 {inner_blocks = array<i64: 1, 16>}  : vector<32x32xf16> -> vector<32x2x1x16xf16>
    %2 = xetile.tile_pack %arg1 {inner_blocks = array<i64: 1, 16>}  : vector<32x32xf16> -> vector<32x2x1x16xf16>
    %3 = arith.select %0, %1, %2 : vector<32x2x1x16xi1>, vector<32x2x1x16xf16>
    %4 = xetile.tile_unpack %3 {inner_blocks = array<i64: 1, 16>}  : vector<32x2x1x16xf16> -> vector<32x32xf16>
    %5 = xetile.init_tile %arg2[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %6 = xetile.tile_pack %4 {inner_blocks = array<i64: 8, 32>}  : vector<32x32xf16> -> vector<4x1x8x32xf16>
    xetile.store_tile %6,  %5 : vector<4x1x8x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    gpu.return
  }
}

//CHECK: gpu.func @create_mask_2
//CHECK:     %[[UB:.*]] = arith.constant 20
//CHECK:     %[[C0:.*]] = arith.constant 0
//CHECK:     %[[CMP0:.*]] = arith.cmpi slt, %[[C0]], %[[UB]] : index
//CHECK:     %[[SPLAT0:.*]] = vector.splat %[[CMP0]] : vector<1x16xi1>
//CHECK-COUNT-31: arith.cmpi
//CHECK:     arith.select %[[SPLAT0]], {{.*}} : vector<1x16xi1>, vector<1x16xf16>
//CHECK:     arith.select %[[SPLAT0]], {{.*}} : vector<1x16xi1>, vector<1x16xf16>
//CHECK-COUNT-62: arith.select
