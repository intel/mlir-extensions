// RUN: imex-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics


// -----
func.func @init_tile_with_unranked_memref(%source : memref<?x?xf32>) {
    // the source memref must be ranked
    // expected-error@+1 {{base memref does not have a static shape or stride layout information.}}
    %1 = xetile.init_tile %source[0, 0]
        : memref<?x?xf32> -> !xetile.tile<32x64xf32>

    return
}



// -----
func.func @init_tile_with_invalid_offsets(%source : memref<64x64x64xf32>, %offset : index) {
    // the offsets of the init_tile must be 2D
    // expected-error@+1 {{offsets of the init_tile must be 2D.}}
    %1 = xetile.init_tile %source[%offset, %offset, %offset]
        : memref<64x64x64xf32> -> !xetile.tile<8x8xf32>

    return
}


// -----
func.func @load_tile_with_invalid_inner_blocks(%tile : !xetile.tile<64x64xf32>) {

    // inner_blocks must be 2D
    // expected-error@+1 {{inner_blocks must be two dimensional if specified}}
    %1 = xetile.load_tile %tile inner_blocks = [8,16,4]
        : !xetile.tile<64x64xf32> -> vector<8x4x8x16xf32>

    return
}

// -----
func.func @load_tile_with_invalid_output_rank(%tile : !xetile.tile<64x64xf32>) {

    // if the inner_blocks is specified in tile_load output must be 4D
    // expected-error@+1 {{output must be 4-dimensional if inner_blocks is specified}}
    %1 = xetile.load_tile %tile inner_blocks = [8,16]
        : !xetile.tile<64x64xf32> -> vector<8x4xf32>

    return
}

// -----
func.func @tile_mma_input_rank_mismatch(%a_vec : vector<8x8x8x8xf32>,
    %b_vec : vector<8x8x8xf32>, %c_vec : vector<8x8x8x8xf32>) {

    // the two input vectors must have the same rank
    // expected-error@+1 {{rank mismatch in tile mma inputs}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec
    : (vector<8x8x8x8xf32>, vector<8x8x8xf32>, vector<8x8x8x8xf32>) -> vector<8x8x8x8xf32>

     return
}

// -----
func.func @tile_mma_input_elem_type_mismatch(%a_vec : vector<8x8x8x8xf16>,
    %b_vec : vector<8x8x8x8xf32>, %c_vec : vector<8x8x8x8xf32>) {
    // the two input vectors must have the same element type
    // expected-error@+1 {{element type mismatch in tile mma inputs}}
    %c_new = xetile.tile_mma %a_vec, %b_vec, %c_vec
    : (vector<8x8x8x8xf16>, vector<8x8x8x8xf32>, vector<8x8x8x8xf32>) -> vector<8x8x8x8xf32>

    return
}
