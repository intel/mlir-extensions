// RUN: imex-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics


// -----
func.func @init_tile_with_invalid_offsets(%source : memref<64x64xf32>, %offset : index) {
    // the offsets of the init_tile must be 2D
    // expected-error@+1 {{number of offsets must be 2}}
    %1 = xetile.init_tile %source[%offset, %offset, %offset]
        : memref<64x64xf32> -> !xetile.tile<8x8xf32>
}

// -----
func.func @init_tile_static_memref_with_invalid_dynamic_shape(%source : memref<1024x1024xf32>,
    %dim0_size : index, %dim1_size : index) {
    // for source memref with static shape, dynamic shape arguments should not be present
    // expected-error@+1 {{dynamic shape or strides are not allowed with a static shaped memref as source}}
    %1 = xetile.init_tile %source[0, 0], [%dim0_size, %dim1_size]
        : memref<1024x1024xf32> -> !xetile.tile<64x64xf32>
}

// -----
func.func @init_tile_dynamic_memref_with_invalid_dynamic_shape(%source : memref<?x?xf32>,
    %dim0_size : index, %dim1_size : index, %dim0_stride : index, %dim1_stride : index) {
    // for source memref with dynamic shape, dynamic shape arguments should be 2D
    // expected-error@+1 {{memref with a dynamic shape is used as source but dynamic shape argument missing or it is not 2D}}
    %1 = xetile.init_tile %source[0, 0], [%dim0_size], [%dim0_stride, %dim1_stride]
        : memref<?x?xf32> -> !xetile.tile<64x64xf32>
}

// -----
func.func @init_tile_dynamic_memref_with_invalid_dynamic_strides(%source : memref<?x?xf32>,
    %dim0_size : index, %dim1_size : index, %dim0_stride : index, %dim1_stride : index) {
    // for source memref with dynamic shape, dynamic strides arguments should be 2D
    // expected-error@+1 {{memref with a dynamic shape is used as source but dynamic strides argument missing or it is not 2D}}
    %1 = xetile.init_tile %source[0, 0], [%dim0_size, %dim1_size], [%dim0_stride]
        : memref<?x?xf32> -> !xetile.tile<64x64xf32>
}


// -----
func.func @init_tile_address_with_invalid_dynamic_shape(%source : i64, %dim0_size : index, %dim1_size : index,
    %dim0_stride : index, %dim1_stride : index) {
    // for source address, dynamic shape arguments should be 2D
    // expected-error@+1 {{address is used as source but dynamic shape argument is missing or it is not 2D}}
    %1 = xetile.init_tile %source[0, 0], [%dim0_size], [%dim0_stride, %dim1_stride]
        : i64 -> !xetile.tile<64x64xf32>
}

// -----
func.func @init_tile_address_with_invalid_dynamic_strides(%source : i64, %dim0_size : index, %dim1_size : index,
    %dim0_stride : index, %dim1_stride : index) {
    // for source address, dynamic strides arguments should be 2D
    // expected-error@+1 {{address is used as source but dynamic strides argument is missing or it is not 2D}}
    %1 = xetile.init_tile %source[0, 0], [%dim0_size, %dim1_size], [%dim0_stride]
        : i64 -> !xetile.tile<64x64xf32>
}

// -----
func.func @load_tile_with_invalid_transpose(%tile : !xetile.tile<64x32xf32>) {
    // TRANSPOSE must be 2D
    // expected-error@+1 {{transpose must be two dimensional}}
    %1 = xetile.load_tile %tile { transpose = [1, 0 , 0] }
        : !xetile.tile<64x32xf32> -> vector<32x64xf32>
}


// -----
func.func @tile_mma_incompatible_ranks(%a_vec : vector<8x8x8x8xf32>,
    %b_vec : vector<8x8xf32>, %c_vec : vector<8x8x8x8xf32>) {
    // the two input vectors must have the same rank
    // expected-error@+1 {{A and B inputs must have the same rank.}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec : vector<8x8x8x8xf32>, vector<8x8xf32>, vector<8x8x8x8xf32>
        -> vector<8x8x8x8xf32>
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
func.func @tile_mma_incompatible_mma_shapes_4d(%a_vec : vector<8x16x8x32xf16>,
    %b_vec : vector<16x8x8x8xf16>, %c_vec : vector<8x8x8x8xf32>) {
    // the two input vectors must have the same element type
    // expected-error@+1 {{incompatible A, B and output sizes for 4D tile mma op. 4D tile mma should have the shape (m x k x Bm x Bk) x (k x n x Bk x Bn) = (m x n x Bm x Bn).}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec
    : vector<8x16x8x32xf16>, vector<16x8x8x8xf16>, vector<8x8x8x8xf32> -> vector<8x8x8x8xf32>
}

// -----
func.func @tile_mma_incompatible_mma_shapes_2d(%a_vec : vector<8x16xf16>,
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
// expected-error@+1 {{expect integer array of size 2 for mma_block_size}}
#sg_map_1 = #xetile.sg_map<mma_block_size = [8, 16, 4], wi_layout = [2, 8], wi_data = [1, 2]>
// expected-error@+1 {{expect integer array of size 2 for wi_layout}}
#sg_map_2 = #xetile.sg_map<mma_block_size = [8, 16], wi_layout = [2, 8, 2], wi_data = [1, 2]>
// expected-error@+1 {{expect integer array of size 2 for wi_data}}
#sg_map_3 = #xetile.sg_map<mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2, 1]>
// expected-error@+1 {{expect integer array of size 2 for sg_layout}}
#wg_map_1 = #xetile.wg_map<sg_layout = [4], sg_data = [32, 128]>
// expected-error@+1 {{expect integer array of size 2 for sg_data}}
#wg_map_2 = #xetile.wg_map<sg_layout = [2, 2], sg_data = [32, 128, 32]>
// expected-error@+1 {{expect integer array of size 2 for inner_blocks}}
#tile1 = !xetile.tile<64x64xf16, inner_blocks = [8, 16, 8]>
