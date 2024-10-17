// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

#sg_map = #xetile.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>
#wg_map = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128]>
#tile_attr = #xetile.tile_attr<wg_map = #wg_map, sg_map = #sg_map>
#tile_attr_w_inner_blocks = #xetile.tile_attr<inner_blocks = [8, 16]>
#tile_attr_w_order = #xetile.tile_attr<order = [0, 1]>


#wg_map_mma_a = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>
#wg_map_mma_b = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>
#wg_map_mma_c = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>

#wg_map_a = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 128]>
#wg_map_a2 = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 1]>

#wg_map_b = #xetile.wg_map<sg_layout = [16, 1], sg_data = [16, 1]>
#wg_map_b2 = #xetile.wg_map<sg_layout = [4, 4], sg_data = [64, 64]>

func.func @test_init_tile_for_slm(%a: memref<1024x1024xf16, 3>) {
  //CHECK: xetile.init_tile {{.*}}[8, 16] : memref<1024x1024xf16, 3> -> !xetile.tile<32x64xf16, #xetile.tile_attr<memory_space = 3 : i64>>
  %1 = xetile.init_tile %a[8, 16] : memref<1024x1024xf16, 3> -> !xetile.tile<32x64xf16, #xetile.tile_attr<memory_space = 3>>
  return
}

func.func @test_init_tile_for_global(%a: memref<1024x1024xf16, 0>) {
  //CHECK: xetile.init_tile {{.*}}[8, 16] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
  %1 = xetile.init_tile %a[8, 16] : memref<1024x1024xf16, 0> -> !xetile.tile<32x64xf16>
  return
}

func.func @test_init_tile_for_scattered(%a: memref<1024xf16>, %indices: vector<128xindex>) {
  //CHECK: xetile.init_tile {{.*}} : memref<1024xf16>, vector<128xindex> -> !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>
  %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<128xindex> -> !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>
  return
}

func.func @test_init_tile_for_scattered_2(%a: memref<f16>, %indices: vector<128xindex>) {
  //CHECK: xetile.init_tile {{.*}} : memref<f16>, vector<128xindex> -> !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>
  %1 = xetile.init_tile %a, %indices : memref<f16>, vector<128xindex> -> !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>
  return
}

func.func @test_init_tile_for_scattered_3(%a: memref<f16>, %indices: vector<128x24xindex>) {
  //CHECK: xetile.init_tile {{.*}} : memref<f16>, vector<128x24xindex> -> !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>
  %1 = xetile.init_tile %a, %indices : memref<f16>, vector<128x24xindex> -> !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>
  return
}

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
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
  %4 = xetile.init_tile %src[512, %c128] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map =
  // CHECK-SAME: <wi_layout = [2, 8], wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  %5 = xetile.init_tile %src[0, 0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #tile_attr>

  // CHECK: xetile.init_tile
  // CHECK-SAME: memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map =
  // CHECK-SAME: <wi_layout = [2, 8], wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  %6 = xetile.init_tile %src[0, 0] : memref<1024x1024xf16> -> !xetile.tile<128x128xf16, #tile_attr>

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
  // CHECK-SAME: memref<?x?xf16> -> !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map = <wi_layout = [2, 8],
  // CHECK-SAME: wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  %4 = xetile.init_tile %src[0, 0], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : memref<?x?xf16> -> !xetile.tile<128x128xf16, #tile_attr>

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

  // CHECK:  xetile.init_tile
  // CHECK-SAME: i64 -> !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map = <wi_layout = [2, 8],
  // CHECK-SAME: wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  %4 = xetile.init_tile %src[0, 0], [%dim0_size, %dim1_size], [%dim0_stride, %dim1_stride]
    : i64 -> !xetile.tile<128x128xf16, #tile_attr>

  return
}


// CHECK-LABEL: func @test_load_tile({{.*}}) {
func.func @test_load_tile(%src: !xetile.tile<64x32xf16>, %src1 : !xetile.tile<128x128xf16, #tile_attr>,
                          %src2 : !xetile.tile<64x64xf16, #tile_attr_w_inner_blocks>,
                          %src3 : !xetile.tile<64x32xf16, #tile_attr_w_order>) {
  // CHECK: xetile.load_tile
  // CHECK-SAME: : !xetile.tile<64x32xf16> -> vector<64x32xf16>
  %1 = xetile.load_tile %src : !xetile.tile<64x32xf16> -> vector<64x32xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: {padding = 1.000000e-01 : f32}  : !xetile.tile<64x32xf16> -> vector<64x32xf16>
  %4 = xetile.load_tile %src { padding = 0.1 : f32 } : !xetile.tile<64x32xf16> -> vector<64x32xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: {padding = 1.000000e-01 : f32}  : !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map =
  // CHECK-SAME: <wi_layout = [2, 8], wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  // CHECK-SAME: -> vector<128x128xf16>
  %6 = xetile.load_tile %src1 {  padding = 0.1 : f32 }
    : !xetile.tile<128x128xf16, #tile_attr> -> vector<128x128xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: : !xetile.tile<64x64xf16,
  // CHECK-SAME: #xetile.tile_attr<inner_blocks = [8, 16]>> -> vector<8x4x8x16xf16>
  %7 = xetile.load_tile %src2 : !xetile.tile<64x64xf16, #tile_attr_w_inner_blocks>
    -> vector<8x4x8x16xf16>

  // CHECK:  xetile.load_tile
  // CHECK-SAME: : !xetile.tile<64x32xf16,
  // CHECK-SAME: #xetile.tile_attr<order = [0, 1]>> -> vector<64x32xf16>
  %8 = xetile.load_tile %src3 : !xetile.tile<64x32xf16, #tile_attr_w_order>
    -> vector<64x32xf16>

  return
}

// CHECK-LABEL: func @test_store_tile({{.*}}) {
func.func @test_store_tile(%value1 : vector<64x32xf16>,
  %value2 : vector<8x4x8x16xf16>, %value3 : vector<128x128xf16>, %dst: !xetile.tile<64x32xf16>,
  %dst1 : !xetile.tile<128x128xf16, #tile_attr>,
  %dst2 : !xetile.tile<64x64xf16, #tile_attr_w_inner_blocks>,
  %dst3 : !xetile.tile<64x32xf16, #tile_attr_w_order>) {

  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<64x32xf16>, !xetile.tile<64x32xf16>
  xetile.store_tile %value1, %dst : vector<64x32xf16>, !xetile.tile<64x32xf16>

  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<128x128xf16>, !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map =
  // CHECK-SAME: <wi_layout = [2, 8], wi_data = [1, 2]>, wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  xetile.store_tile %value3, %dst1 : vector<128x128xf16>, !xetile.tile<128x128xf16, #tile_attr>

  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<8x4x8x16xf16>, !xetile.tile<64x64xf16, #xetile.tile_attr<inner_blocks = [8, 16]>>
  xetile.store_tile %value2, %dst2 : vector<8x4x8x16xf16>, !xetile.tile<64x64xf16, #tile_attr_w_inner_blocks>

  // CHECK: xetile.store_tile
  // CHECK-SAME: vector<64x32xf16>, !xetile.tile<64x32xf16, #xetile.tile_attr<order = [0, 1]>>
  xetile.store_tile %value1, %dst3 : vector<64x32xf16>, !xetile.tile<64x32xf16, #tile_attr_w_order>

  return
}

// CHECK-LABEL: func @test_prefetch_tile({{.*}}) {
func.func @test_prefetch_tile(%src: !xetile.tile<64x64xf16>, %src1: !xetile.tile<128x128xf16>) {

  // CHECK: xetile.prefetch_tile
  // CHECK-SAME: !xetile.tile<64x64xf16>
  xetile.prefetch_tile %src : !xetile.tile<64x64xf16>

  // CHECK: xetile.prefetch_tile
  // CHECK-SAME: !xetile.tile<128x128xf16>
  xetile.prefetch_tile %src1 : !xetile.tile<128x128xf16>

  return
}


// CHECK-LABEL: func @test_tile_mma({{.*}}) {
func.func @test_tile_mma(%a: !xetile.tile<64x32xf16>, %b: !xetile.tile<32x128xf16>, %c : !xetile.tile<64x128xf16>,
            %a_tiled: !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks=[8,8]>>,
            %b_tiled: !xetile.tile<32x128xf16, #xetile.tile_attr<inner_blocks=[8, 16]>>,
            %c_tiled: !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks=[8, 16]>>) {

  // CHECK: xetile.load_tile
  // CHECK-SAME: : !xetile.tile<64x32xf16> -> vector<64x32xf16>
  %a_vec = xetile.load_tile %a : !xetile.tile<64x32xf16> -> vector<64x32xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: : !xetile.tile<32x128xf16> -> vector<32x128xf16>
  %b_vec = xetile.load_tile %b : !xetile.tile<32x128xf16> -> vector<32x128xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: : !xetile.tile<64x128xf16> -> vector<64x128xf16>
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
  // CHECK-SAME: !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 8]>> -> vector<8x4x8x8xf16>
  %a_vec_1 = xetile.load_tile %a_tiled: !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks=[8,8]>>
    -> vector<8x4x8x8xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<32x128xf16, #xetile.tile_attr<inner_blocks = [8, 16]>> -> vector<4x8x8x16xf16>
  %b_vec_1 = xetile.load_tile %b_tiled: !xetile.tile<32x128xf16, #xetile.tile_attr<inner_blocks=[8, 16]>>
   -> vector<4x8x8x16xf16>

  // CHECK: xetile.load_tile
  // CHECK-SAME: !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks = [8, 16]>> -> vector<8x8x8x16xf16>
  %c_vec_1 = xetile.load_tile %c_tiled: !xetile.tile<64x128xf16, #xetile.tile_attr<inner_blocks=[8, 16]>>
   -> vector<8x8x8x16xf16>

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

func.func @test_load_gather(%a: memref<1024xf16>, %indices: vector<128xindex>) {
  %mask = arith.constant dense<1> : vector<128xi1>
  %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<128xindex> -> !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>
  //CHECK: xetile.load {{.*}} : !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>, vector<128xi1> -> vector<128xf16>
  %2 = xetile.load %1, %mask : !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>, vector<128xi1> -> vector<128xf16>
  return
}

func.func @test_load_gather_2(%a: memref<1024xf16>, %indices: vector<128x24xindex>) {
  %mask = arith.constant dense<1> : vector<128x24xi1>
  %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<128x24xindex> -> !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>
  //CHECK: xetile.load {{.*}} : !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>, vector<128x24xi1> -> vector<128x24xf16>
  %2 = xetile.load %1, %mask : !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>, vector<128x24xi1> -> vector<128x24xf16>
  return
}

func.func @test_store_scatter(%a: memref<1024xf16>, %indices: vector<128xindex>, %data: vector<128xf16>) {
  %mask = arith.constant dense<1> : vector<128xi1>
  %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<128xindex> -> !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>
  // CHECK: xetile.store {{.*}} : vector<128xf16>, !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>, vector<128xi1>
  xetile.store %data, %1, %mask : vector<128xf16>, !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>, vector<128xi1>
  return
}

func.func @test_store_scatter_2(%a: memref<1024xf16>, %indices: vector<128x24xindex>, %data: vector<128x24xf16>) {
  %mask = arith.constant dense<1> : vector<128x24xi1>
  %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<128x24xindex> -> !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>
  // CHECK: xetile.store {{.*}} : vector<128x24xf16>, !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>, vector<128x24xi1>
  xetile.store %data, %1, %mask : vector<128x24xf16>, !xetile.tile<128x24xf16, #xetile.tile_attr<scattered = true>>, vector<128x24xi1>
  return
}

// CHECK-LABEL: func @test_update_tile_offset({{.*}}) {
func.func @test_update_tile_offset(%tile: !xetile.tile<32x32xf16>, %tile1 : !xetile.tile<128x128xf16, #tile_attr>) {

  %offset_x = arith.constant 0 : index
  %offset_y = arith.constant 96 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index

  // CHECK: xetile.update_tile_offset {{.*}} : !xetile.tile<32x32xf16>
  %1 = xetile.update_tile_offset %tile, [%offset_x, %offset_y] : !xetile.tile<32x32xf16>

  // CHECK: xetile.update_tile_offset
  // CHECK-SAME: !xetile.tile<128x128xf16, #xetile.tile_attr<sg_map = <wi_layout = [2, 8], wi_data = [1, 2]>,
  // CHECK-SAME: wg_map = <sg_layout = [2, 2], sg_data = [32, 128]>>>
  %2 = xetile.update_tile_offset %tile1, [%c128, %c0] : !xetile.tile<128x128xf16, #tile_attr>

  return
}

func.func @test_update_tile_offset_scattered(%a: memref<1024xf16>, %indices: vector<128xindex>, %data: vector<128xf16>) {
  %step = arith.constant dense<2> : vector<128xindex>
  %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<128xindex> -> !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>
  // CHECK: xetile.update_tile_offset {{.*}} : !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>, vector<128xindex>
  %2 = xetile.update_tile_offset %1, %step : !xetile.tile<128xf16, #xetile.tile_attr<scattered = true>>, vector<128xindex>
  return
}

// CHECK-LABEL: func @test_tile_pack({{.*}}) {
func.func @test_tile_pack(%source : vector<32x64xf16>) {
  // CHECK: xetile.tile_pack
  // CHECK-SAME: {inner_blocks = array<i64: 16, 16>}
  // CHECK-SAME: vector<32x64xf16> -> vector<2x4x16x16xf16>
  %1 = xetile.tile_pack %source {inner_blocks = array<i64: 16, 16>} : vector<32x64xf16> -> vector<2x4x16x16xf16>
  return
}

// CHECK-LABEL: func @test_tile_unpack({{.*}}) {
func.func @test_tile_unpack(%source : vector<2x4x16x16xf16>) {
  // CHECK: xetile.tile_unpack
  // CHECK-SAME: {inner_blocks = array<i64: 16, 16>}
  // CHECK-SAME: vector<2x4x16x16xf16> -> vector<32x64xf16>
  %1 = xetile.tile_unpack %source {inner_blocks = array<i64: 16, 16>} : vector<2x4x16x16xf16> -> vector<32x64xf16>
  return
}

// CHECK-LABEL: func @test_atomic_rmw({{.*}}) {
func.func @test_atomic_rmw(%tile : !xetile.tile<8x16xf16>, %value : vector<8x16xf16>) {
  // CHECK:  xetile.atomic_rmw addf
  // CHECK-SAME: vector<8x16xf16>, !xetile.tile<8x16xf16> -> vector<8x16xf16>
  %1 = xetile.atomic_rmw "addf" %value, %tile : vector<8x16xf16>, !xetile.tile<8x16xf16> -> vector<8x16xf16>
  return
}

func.func @test_transpose(%source: vector<8x16xf16>) {
  // CHECK: xetile.transpose {{.*}} [1, 0] : vector<8x16xf16> -> vector<16x8xf16>
  %1 = xetile.transpose %source, [1, 0] : vector<8x16xf16> -> vector<16x8xf16>
  return
}

func.func @test_reduce(%source: vector<8x16xf16>) {
  // CHECK: xetile.reduction {{.*}} [0] : vector<8x16xf16> -> vector<1x16xf16>
  %1 = xetile.reduction <add>, %source [0] : vector<8x16xf16> -> vector<1x16xf16>
  return
}

func.func @test_reduce_map(%source: vector<256x128xf16>) {
  // CHECK: xetile.reduction {{.*}} [1] {map1 = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 128]>, map2 = #xetile.wg_map<sg_layout = [32, 1], sg_data = [8, 1]>} : vector<256x128xf16> -> vector<256x1xf16>
  %1 = xetile.reduction <add>, %source [1] {map1 = #wg_map_a, map2 = #wg_map_a2} : vector<256x128xf16> -> vector<256x1xf16>
  return
}


func.func @test_broadcast(%source: vector<1x16xf16>) {
  // CHECK: xetile.broadcast {{.*}} [0] : vector<1x16xf16> -> vector<8x16xf16>
  %1 = xetile.broadcast %source [0] : vector<1x16xf16> -> vector<8x16xf16>
  return
}

func.func @test_broadcast_map(%source: vector<256x1xf16>) {
  // CHECK: xetile.broadcast {{.*}} [1] {map1 = #xetile.wg_map<sg_layout = [16, 1], sg_data = [16, 1]>, map2 = #xetile.wg_map<sg_layout = [4, 4], sg_data = [64, 64]>} : vector<256x1xf16> -> vector<256x256xf16>
  %1 = xetile.broadcast %source [1] {map1 = #wg_map_b, map2 = #wg_map_b2} : vector<256x1xf16> -> vector<256x256xf16>
  return
}


func.func @test_tile_mma_map(%a : vector<256x256xf16>, %b : vector<256x256xf16>, %c : vector<256x256xf32>) {
// CHECK: xetile.tile_mma
// CHECK-SAME : {wg_map_a =#xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>, wg_map_b =#xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>, wg_map_c =#xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>}
 %result = xetile.tile_mma %a, %b, %c {wg_map_a = #wg_map_mma_a, wg_map_b = #wg_map_mma_b, wg_map_c = #wg_map_mma_c} :
     vector<256x256xf16>, vector<256x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
  return
}
