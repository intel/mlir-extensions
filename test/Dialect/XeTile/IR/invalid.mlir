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
func.func @load_tile_with_invalid_inner_blocks(%tile : !xetile.tile<64x64xf32>) {
    // INNER_BLOCKS must be 2D
    // expected-error@+1 {{inner_blocks must be two dimensional}}
    %1 = xetile.load_tile %tile { inner_blocks = [8,16,4] }
        : !xetile.tile<64x64xf32> -> vector<8x4x8x16xf32>
}

// -----
func.func @load_tile_with_invalid_transpose(%tile : !xetile.tile<64x32xf32>) {
    // TRANSPOSE must be 2D
    // expected-error@+1 {{transpose must be two dimensional}}
    %1 = xetile.load_tile %tile { transpose = [1, 0 , 0] }
        : !xetile.tile<64x32xf32> -> vector<32x64xf32>
}

// -----
func.func @load_tile_with_invalid_output_rank(%tile : !xetile.tile<64x64xf32>) {
    // if the INNER_BLOCKS is specified in tile_load output must be 4D
    // expected-error@+1 {{output must be 4-dimensional if inner_blocks is specified}}
    %1 = xetile.load_tile %tile { inner_blocks = [8,16] }
        : !xetile.tile<64x64xf32> -> vector<8x4xf32>

}

// -----
func.func @tile_mma_input_rank_mismatch(%a_vec : vector<8x8x8x8xf32>,
    %b_vec : vector<8x8x8xf32>, %c_vec : vector<8x8x8x8xf32>) {
    // the two input vectors must have the same rank
    // expected-error@+1 {{rank mismatch in tile mma inputs}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec {a_inner_blocks = [8, 8], b_inner_blocks = [8, 8]}
    : (vector<8x8x8x8xf32>, vector<8x8x8xf32>, vector<8x8x8x8xf32>) -> vector<8x8x8x8xf32>
}

// -----
func.func @tile_mma_input_elem_type_mismatch(%a_vec : vector<8x8x8x8xf16>,
    %b_vec : vector<8x8x8x8xf32>, %c_vec : vector<8x8x8x8xf32>) {
    // the two input vectors must have the same element type
    // expected-error@+1 {{element type mismatch in tile mma inputs}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec {a_inner_blocks = [8, 8], b_inner_blocks = [8, 8]}
    : (vector<8x8x8x8xf16>, vector<8x8x8x8xf32>, vector<8x8x8x8xf32>) -> vector<8x8x8x8xf32>
}
