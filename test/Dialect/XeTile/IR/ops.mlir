// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

#sg_map = #xetile.sg_map<mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>
#wg_map = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
#xe_map = #xetile.xe_map<wg = #wg_map, sg = #sg_map>

// init_tile with a static shaped memref
// CHECK-LABEL: func @test_init_tile_using_static_memref({{.*}}) {
func.func @test_init_tile_using_static_memref(%src: memref<1024x1024xf16>) {

  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
  %1 = xetile.init_tile %src[8, 16] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
  %2 = xetile.init_tile %src[%c128, %c256] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
  %3 = xetile.init_tile %src[512, %c128] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<32x64xf16, inner_blocks = [32, 16]>
  %4 = xetile.init_tile %src[512, %c128] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, inner_blocks = [32, 16]>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>>
  %5 = xetile.init_tile %src[0, 0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xe_map>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<128x128xf16, inner_blocks = [32, 16],
  // CHECK-SAME: #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>>
  %6 = xetile.init_tile %src[0, 0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, inner_blocks = [32, 16], #xe_map>

  return
}

// init tile with a dynmaic shaped memref
// CHECK-LABEL: func @test_init_tile_using_dynamic_memref({{.*}}) {
func.func @test_init_tile_using_dynamic_memref(%src: memref<?x?xf16>, %dim0_size : index, %dim1_size : index,
    %dim0_stride : index, %dim1_stride : index ) {

  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<?x?xf16> -> !xetile.tile<32x64xf16>
  %1 = xetile.init_tile %src[8, 16], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : memref<?x?xf16> -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<?x?xf16> -> !xetile.tile<32x64xf16>
  %2 = xetile.init_tile %src[%c128, %c256], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : memref<?x?xf16> -> !xetile.tile<32x64xf16>


  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<?x?xf16> -> !xetile.tile<32x64xf16>
  %3 = xetile.init_tile %src[%c128, 64], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : memref<?x?xf16> -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<?x?xf16> -> !xetile.tile<128x128xf16, #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>>
  %4 = xetile.init_tile %src[0, 0], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : memref<?x?xf16> -> !xetile.tile<128x128xf16, #xe_map>

  return
}

// init tile with an addr
// CHECK-LABEL: func @test_init_tile_using_addr({{.*}}) {
func.func @test_init_tile_using_addr(%src: i64, %dim0_size : index, %dim1_size : index,
    %dim0_stride : index, %dim1_stride : index ) {

  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  // CHECK: xetile.init_tile
  // CHECK-SAME: i64 -> !xetile.tile<32x64xf16>
  %1 = xetile.init_tile %src[8, 16], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : i64 -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile
  // CHECK-SAME: i64 -> !xetile.tile<32x64xf16>
  %2 = xetile.init_tile %src[%c128, %c256], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : i64 -> !xetile.tile<32x64xf16>


  // CHECK: xetile.init_tile
  // CHECK-SAME: i64 -> !xetile.tile<32x64xf16>
  %3 = xetile.init_tile %src[%c128, 64], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : i64 -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile %arg0[0, 0], [%arg1, %arg2], [%arg3, %arg4]
  // CHECK-SAME: i64 -> !xetile.tile<128x128xf16, #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>>
  %4 = xetile.init_tile %src[0, 0], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : i64 -> !xetile.tile<128x128xf16, #xe_map>

  return
}


// CHECK-LABEL: func @test_load_tile({{.*}}) {
func.func @test_load_tile(%src: !xetile.tile<64x32xf16>, %src1 : !xetile.tile<128x128xf16, #xe_map>) {
  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }  : !xetile.tile<64x32xf16> -> vector<64x32xf16>
  %1 = xetile.load_tile %src : !xetile.tile<64x32xf16> -> vector<64x32xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }
  // CHECK-SAME: !xetile.tile<64x32xf16> -> vector<8x2x8x16xf16>
  %2 = xetile.load_tile %src : !xetile.tile<64x32xf16> -> vector<8x2x8x16xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { transpose = [1, 0], padding = 0.000000e+00 : f32 }  : !xetile.tile<64x32xf16> -> vector<32x64xf16>
  %3 = xetile.load_tile %src { transpose = [1, 0] } : !xetile.tile<64x32xf16> -> vector<32x64xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 1.000000e-01 : f32 }  : !xetile.tile<64x32xf16> -> vector<64x32xf16>
  %4 = xetile.load_tile %src { padding = 0.1 : f32 } : !xetile.tile<64x32xf16> -> vector<64x32xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { transpose = [1, 0], padding = 1.000000e-01 : f32 }
  // CHECK-SAME: !xetile.tile<64x32xf16> -> vector<2x8x16x8xf16>
  %5 = xetile.load_tile %src { transpose = [1, 0], padding = 0.1 : f32 }
    : !xetile.tile<64x32xf16> -> vector<2x8x16x8xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { transpose = [1, 0], padding = 1.000000e-01 : f32 }
  // CHECK-SAME: !xetile.tile<128x128xf16, #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>> -> vector<2x8x16x8xf16>
  %6 = xetile.load_tile %src1 { transpose = [1, 0], padding = 0.1 : f32 }
    : !xetile.tile<128x128xf16, #xe_map> -> vector<2x8x16x8xf16>

  return
}

// CHECK-LABEL: func @test_store_tile({{.*}}) {
func.func @test_store_tile(%value1 : vector<64x32xf16>,
  %value2 : vector<8x2x8x16xf16>, %value3 : vector<16x8x8x16xf16>, %dst: !xetile.tile<64x32xf16>, %dst1: !xetile.tile<128x128xf16, #xe_map>) {

  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<64x32xf16>, !xetile.tile<64x32xf16>
  xetile.store_tile %value1, %dst : vector<64x32xf16>, !xetile.tile<64x32xf16>

  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<8x2x8x16xf16>, !xetile.tile<64x32xf16>
  xetile.store_tile %value2, %dst : vector<8x2x8x16xf16>, !xetile.tile<64x32xf16>

  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<16x8x8x16xf16>, !xetile.tile<128x128xf16, #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>>
  xetile.store_tile %value3, %dst1 : vector<16x8x8x16xf16>, !xetile.tile<128x128xf16, #xe_map>

  return
}

// CHECK-LABEL: func @test_prefetch_tile({{.*}}) {
func.func @test_prefetch_tile(%src: !xetile.tile<64x64xf16>) {

  // CHECK: xetile.prefetch_tile
  // CHECK-SAME: !xetile.tile<64x64xf16>
  xetile.prefetch_tile %src : !xetile.tile<64x64xf16>

  return
}


// CHECK-LABEL: func @test_tile_mma({{.*}}) {
func.func @test_tile_mma(%a: !xetile.tile<64x32xf16>, %b: !xetile.tile<32x128xf16>, %c : !xetile.tile<64x128xf16>) {

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }  : !xetile.tile<64x32xf16> -> vector<64x32xf16>
  %a_vec = xetile.load_tile %a : !xetile.tile<64x32xf16> -> vector<64x32xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x128xf16> -> vector<32x128xf16>
  %b_vec = xetile.load_tile %b : !xetile.tile<32x128xf16> -> vector<32x128xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }  : !xetile.tile<64x128xf16> -> vector<64x128xf16>
  %c_vec = xetile.load_tile %c : !xetile.tile<64x128xf16> -> vector<64x128xf16>

  // CHECK: xetile.tile_mma
  // CHECK-SAME: vector<64x32xf16>,  vector<32x128xf16> -> vector<64x128xf16>
  %c_new = xetile.tile_mma %a_vec, %b_vec
    : vector<64x32xf16>, vector<32x128xf16> -> vector<64x128xf16>

  // CHECK: xetile.tile_mma
  // CHECK-SAME: vector<64x32xf16>,  vector<32x128xf16>,  vector<64x128xf16> -> vector<64x128xf16>
  %c_new_ = xetile.tile_mma %a_vec, %b_vec, %c_vec
    : vector<64x32xf16>, vector<32x128xf16>, vector<64x128xf16> -> vector<64x128xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }
  // CHECK-SAME: !xetile.tile<64x32xf16> -> vector<8x4x8x8xf16>
  %a_vec_1 = xetile.load_tile %a : !xetile.tile<64x32xf16> -> vector<8x4x8x8xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }
  // CHECK-SAME: !xetile.tile<32x128xf16> -> vector<4x8x8x16xf16>
  %b_vec_1 = xetile.load_tile %b : !xetile.tile<32x128xf16> -> vector<4x8x8x16xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: { padding = 0.000000e+00 : f32 }
  // CHECK-SAME: !xetile.tile<64x128xf16> -> vector<8x8x8x16xf16>
  %c_vec_1 = xetile.load_tile %c : !xetile.tile<64x128xf16> -> vector<8x8x8x16xf16>

  // CHECK: xetile.tile_mma
  // CHECK-SAME: vector<8x4x8x8xf16>, vector<4x8x8x16xf16> -> vector<8x8x8x16xf16>
  %c_new_1 = xetile.tile_mma %a_vec_1, %b_vec_1
    : vector<8x4x8x8xf16>, vector<4x8x8x16xf16> -> vector<8x8x8x16xf16>

  // CHECK: xetile.tile_mma
  // CHECK-SAME: vector<8x4x8x8xf16>, vector<4x8x8x16xf16>, vector<8x8x8x16xf16> -> vector<8x8x8x16xf16>
  %c_new_1_ = xetile.tile_mma %a_vec_1, %b_vec_1, %c_vec_1
    : vector<8x4x8x8xf16>, vector<4x8x8x16xf16>, vector<8x8x8x16xf16> -> vector<8x8x8x16xf16>

  return
}


// CHECK-LABEL: func @test_update_tile_offset({{.*}}) {
func.func @test_update_tile_offset(%tile: !xetile.tile<32x32xf16>, %tile1 : !xetile.tile<128x128xf16, #xe_map>) {

  %offset_x = arith.constant 0 : index
  %offset_y = arith.constant 96 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index

  // CHECK: xetile.update_tile_offset
  // CHECK-SAME: : !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>
  %1 = xetile.update_tile_offset %tile, [%offset_x, %offset_y]
    : !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>

  // CHECK: xetile.update_tile_offset
  // CHECK-SAME: !xetile.tile<128x128xf16, #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>>, index, index
  // CHECK-SAME: -> !xetile.tile<128x128xf16, #xetile.xe_map<wg = <sg_layout = [2, 2], sg_data = [32, 128]>,
  // CHECK-SAME: sg = <mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]>>>
  %2 = xetile.update_tile_offset %tile1, [%c128, %c0]
    : !xetile.tile<128x128xf16, #xe_map>, index, index -> !xetile.tile<128x128xf16, #xe_map>

  return
}
