// RUN: imex-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// -----
func.func @test_init_tile_invalid_order(%src: memref<1024x1024xf16>) {
   // Memref is row major but tile is column major
   // expected-error@+1 {{memref operand is expected to have a column-major layout}}
  %1 = xetile.init_tile %src[8, 16] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<order = [0, 1]>>
  return
}

// -----
func.func @test_init_tile_with_invalid_order(%a: memref<1024x1024xf16, affine_map<(d0, d1) -> (d1*1024 + d0)>>) {
   // Memref is column major but tile is row major
   // expected-error@+1 {{memref operand is expected to have a row-major layout}}
  %1 = xetile.init_tile %a[8, 16] : memref<1024x1024xf16, affine_map<(d0, d1) -> (d1*1024 + d0)>> -> !xetile.tile<32x64xf16>
  return
}

// -----
func.func @test_init_tile_with_invalid_strided_layout(%a: memref<512x1024xf16, strided<[1, 256], offset: ?>>) {
   // Memref is column major but tile is row major
   // expected-error@+1 {{memref operand is expected to have a row-major layout}}
  %1 = xetile.init_tile %a[8, 16] : memref<512x1024xf16, strided<[1, 256], offset: ?>> -> !xetile.tile<32x64xf16>
  return
}

// -----
func.func @test_init_tile_invalid_order_using_address(%src : i64) {
   // Expected row major access
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  // expected-error@+1 {{memref operand is expected to have a row-major layout}}
  %1 = xetile.init_tile %src[%c256, %c64], [%c1024, %c1024], [%c1, %c1024] : i64 -> !xetile.tile<32x64xf16, #xetile.tile_attr<order = [1, 0]>>
  return
}

// -----
func.func @test_init_tile_using_address(%src : i64) {
   // Expected column major access
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  // expected-error@+1 {{memref operand is expected to have a column-major layout}}
  %1 = xetile.init_tile %src[%c256, %c64], [%c512, %c1024], [%c1024, %c1] : i64 -> !xetile.tile<32x64xf16, #xetile.tile_attr<order = [0, 1]>>
  return
}

// -----
func.func @init_tile_static_memref_with_invalid_dynamic_shape(%source : memref<1024x1024xf32>,
    %dim0_size : index, %dim1_size : index) {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // for source memref with static shape, dynamic shape arguments should not be present
    // expected-error@+1 {{dynamic sizes are not allowed with a static shaped memref as source}}
    %1 = xetile.init_tile %source[0, 0], [%dim0_size, %dim1_size], [%c1024, %c1]
        : memref<1024x1024xf32> -> !xetile.tile<64x64xf32>
}

// -----
func.func @tile_mma_input_elem_type_mismatch(%a_vec : vector<8x8xf32>,
    %b_vec : vector<8x8xf16>, %c_vec : vector<8x8xf32>) {
    // the two input vectors must have the same rank
    // expected-error@+1 {{A and B inputs must have the same type.}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec : vector<8x8xf32>, vector<8x8xf16>, vector<8x8xf32> -> vector<8x8xf32>
}

// -----
func.func @tile_mma_output_elem_type_mismatch(%a_vec : vector<8x8xf32>,
    %b_vec : vector<8x8xf32>, %c_vec : vector<8x8xf16>) {
    // the two input vectors must have the same rank
    // expected-error@+1 {{C and output vector must have the same type.}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec : vector<8x8xf32>, vector<8x8xf32>, vector<8x8xf16> -> vector<8x8xf32>
}

// -----
func.func @tile_mma_incompatible_mma_shapes(%a_vec : vector<8x16xf16>,
    %b_vec : vector<8x8xf16>, %c_vec : vector<8x8xf32>) {
    // the two input vectors must have the same element type
    // expected-error@+1 {{incompatible A, B and output sizes for 2D tile mma op. 2D tile mma should have the shape (m x k) x (k x n) = (m x n).}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec : vector<8x16xf16>, vector<8x8xf16>, vector<8x8xf32> -> vector<8x8xf32>
}

// -----
func.func @tile_mma_input_c_shape_mismatch(%a_vec : vector<8x16xf16>,
    %b_vec : vector<16x8xf16>, %c_vec : vector<16x8xf32>) {
    // the two input vectors must have the same element type
    // expected-error@+1 {{input C must have the same shape as output.}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec : vector<8x16xf16>, vector<16x8xf16>, vector<16x8xf32> -> vector<8x8xf32>
}

// -----
func.func @test_init_tile_with_mismatch_memory_space(%a: memref<1024x1024xf16, 3>) {
   // expected-error@+1 {{memory space of the tile doesn't match with the source}}
  %1 = xetile.init_tile %a[8, 16] : memref<1024x1024xf16, 3> -> !xetile.tile<32x64xf16>
  return
}

// -----
// expected-error@+1 {{expect integer array of size 2 for wi_layout}}
#sg_map_2 = #xetile.sg_map< wi_layout = [2, 8, 2], wi_data = [1, 2]>
// expected-error@+1 {{expect integer array of size 2 for wi_data}}
#sg_map_3 = #xetile.sg_map< wi_layout = [2, 8], wi_data = [1, 2, 1]>
// expected-error@+1 {{expect integer array of size 2 for sg_layout}}
#wg_map_1 = #xetile.wg_map<sg_layout = [4], sg_data = [32, 128]>
// expected-error@+1 {{expect integer array of size 2 for sg_data}}
#wg_map_2 = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128, 32]>
// expected-error@+1 {{expect integer array of size 2 for order}}
#wg_map_3 = #xetile.tile_attr<order = [0, 1, 2]>


// -----
func.func @test_transpose(%source: vector<8x16xf16>) {
  // expected-error@+1 {{Incorrect transpose permutation}}
  %1 = xetile.transpose %source, [0, 1] : vector<8x16xf16> -> vector<16x8xf16>
  return
}

// -----
func.func @test_reduce(%source: vector<8x16xf16>) {
  // expected-error@+1 {{reduction dimension of result must have size 1}}
  %1 = xetile.reduction <add>, %source [0] : vector<8x16xf16> -> vector<2x16xf16>
  return
}

// -----
func.func @test_broadcast(%source: vector<2x16xf16>) {
  // expected-error@+1 {{broadcast dimension of source must have size 1}}
  %1 = xetile.broadcast %source [0] : vector<2x16xf16> -> vector<8x16xf16>
  return
}

// -----
func.func @test_convert_layout(%a: vector<64x64xf32>) {
  // expected-error@+1 {{data is not evenly distributed to each subgroup}}
  %1 = xetile.convert_layout %a {wg_map_result = #xetile.wg_map<sg_layout = [4, 8], sg_data = [16, 16]>, wg_map_source = #xetile.wg_map<sg_layout = [8, 4], sg_data = [16, 16]>} : vector<64x64xf32>
  return
}
// -----
func.func @test_convert_layout(%a: vector<64x64xf32>) {
  // expected-error@+1 {{data is not evenly distributed to each subgroup}}
  %1 = xetile.convert_layout %a {wg_map_result = #xetile.wg_map<sg_layout = [4, 8], sg_data = [32, 8]>, wg_map_source = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 32]>} : vector<64x64xf32>
  return
}

// -----
func.func @test_invalid_wg_map(%a: vector<64x64xf32>, %b: memref<64x64xf32>) {
  %t = xetile.init_tile %b[0, 0] : memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 32]>>>
  // expected-error@+1 {{data is not evenly distributed to each subgroup}}
  xetile.store_tile %a, %t : vector<64x64xf32>, !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 32]>>>
  return
}

// -----
func.func @test_invalid_wg_map(%a: vector<64x64xf32>, %b: memref<64x64xf32>) {
  %t = xetile.init_tile %b[0, 0] : memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 32]>>>
  // expected-error@+1 {{data is not evenly distributed to each subgroup}}
  xetile.store_tile %a, %t : vector<64x64xf32>, !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 32]>>>
  return
}
