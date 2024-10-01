// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  // CHECK: sg_tiled_gemm(%[[ARG0:.*]]: memref<1024x1024xf16>, %[[ARG1:.*]]: memref<1024x1024xf16>)
  gpu.func @sg_tiled_gemm(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    // CHECK: %[[C64:.*]] = arith.constant 64 : index
    %c64 = arith.constant 64 : index

    // CHECK: %[[REG0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[C0]], %[[C64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    %0 = xetile.init_tile %arg0[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>

    // CHECK: %[[REG1:.*]] = xegpu.load_nd %[[REG0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    // CHECK: %[[REG2:.*]] = vector.extract %[[REG1]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    // CHECK: %[[REG3:.*]] = vector.extract %[[REG1]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    %1 = xetile.load_tile %0 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>


    // CHECK: %[[REG4:.*]] = xegpu.create_nd_tdesc %arg1[%[[C64]], %[[C0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    // CHECK: %[[C32:.*]] = arith.constant 32 : index
    // CHECK: %[[REG5:.*]] = xegpu.create_nd_tdesc %arg1[%[[C64]], %[[C32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    %2 = xetile.init_tile %arg1[%c64, %c0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>

    // CHECK: %[[REG6:.*]] = xegpu.load_nd %[[REG4]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    // CHECK: %[[REG7:.*]] = vector.extract %[[REG6]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    // CHECK: %[[REG8:.*]] = vector.extract %[[REG6]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    // CHECK: %[[REG9:.*]] = xegpu.load_nd %[[REG5]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    // CHECK: %[[REG10:.*]] = vector.extract %[[REG9]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    // CHECK: %[[REG11:.*]] = vector.extract %[[REG9]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    %3 = xetile.load_tile %2 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x4x32x16xf16>

    //CHECK: %[[REG12:.*]] = vector.extract_strided_slice %[[REG2]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[REG13:.*]] = vector.extract_strided_slice %[[REG2]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[REG14:.*]] = vector.extract_strided_slice %[[REG2]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[REG15:.*]] = vector.extract_strided_slice %[[REG2]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[REG16:.*]] = vector.extract_strided_slice %[[REG3]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[REG17:.*]] = vector.extract_strided_slice %[[REG3]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[REG18:.*]] = vector.extract_strided_slice %[[REG3]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[REG19:.*]] = vector.extract_strided_slice %[[REG3]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    %4 = xetile.tile_unpack %1 {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
    %5 = xetile.tile_pack %4 {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>

    // CHECK: %[[REG20:.*]] = vector.extract_strided_slice %[[REG7]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    // CHECK: %[[REG21:.*]] = vector.extract_strided_slice %[[REG7]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    // CHECK: %[[REG22:.*]] = vector.extract_strided_slice %[[REG8]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    // CHECK: %[[REG23:.*]] = vector.extract_strided_slice %[[REG8]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    // CHECK: %[[REG24:.*]] = vector.extract_strided_slice %[[REG10]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    // CHECK: %[[REG25:.*]] = vector.extract_strided_slice %[[REG10]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    // CHECK: %[[REG26:.*]] = vector.extract_strided_slice %[[REG11]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    // CHECK: %[[REG27:.*]] = vector.extract_strided_slice %[[REG11]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    %6 = xetile.tile_unpack %3 {inner_blocks = array<i64: 32, 16>}  : vector<1x4x32x16xf16> -> vector<32x64xf16>
    %7 = xetile.tile_pack %6 {inner_blocks = array<i64: 16, 16>}  : vector<32x64xf16> -> vector<2x4x16x16xf16>

    // CHECK: %[[REG28:.*]] = xegpu.dpas %[[REG12]], %[[REG20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG29:.*]] = xegpu.dpas %[[REG16]], %[[REG21]], %[[REG28]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG30:.*]] = xegpu.dpas %[[REG12]], %[[REG22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG31:.*]] = xegpu.dpas %[[REG16]], %[[REG23]], %[[REG30]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG32:.*]] = xegpu.dpas %[[REG12]], %[[REG24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG33:.*]] = xegpu.dpas %[[REG16]], %[[REG25]], %[[REG32]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG34:.*]] = xegpu.dpas %[[REG12]], %[[REG26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG35:.*]] = xegpu.dpas %[[REG16]], %[[REG27]], %[[REG34]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG36:.*]] = xegpu.dpas %[[REG13]], %[[REG20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG37:.*]] = xegpu.dpas %[[REG17]], %[[REG21]], %[[REG36]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG38:.*]] = xegpu.dpas %[[REG13]], %[[REG22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG39:.*]] = xegpu.dpas %[[REG17]], %[[REG23]], %[[REG38]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG40:.*]] = xegpu.dpas %[[REG13]], %[[REG24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG41:.*]] = xegpu.dpas %[[REG17]], %[[REG25]], %[[REG40]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG42:.*]] = xegpu.dpas %[[REG13]], %[[REG26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG43:.*]] = xegpu.dpas %[[REG17]], %[[REG27]], %[[REG42]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG44:.*]] = xegpu.dpas %[[REG14]], %[[REG20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG45:.*]] = xegpu.dpas %[[REG18]], %[[REG21]], %[[REG44]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG46:.*]] = xegpu.dpas %[[REG14]], %[[REG22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG47:.*]] = xegpu.dpas %[[REG18]], %[[REG23]], %[[REG46]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG48:.*]] = xegpu.dpas %[[REG14]], %[[REG24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG49:.*]] = xegpu.dpas %[[REG18]], %[[REG25]], %[[REG48]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG50:.*]] = xegpu.dpas %[[REG14]], %[[REG26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG51:.*]] = xegpu.dpas %[[REG18]], %[[REG27]], %[[REG50]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG52:.*]] = xegpu.dpas %[[REG15]], %[[REG20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG53:.*]] = xegpu.dpas %[[REG19]], %[[REG21]], %[[REG52]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG54:.*]] = xegpu.dpas %[[REG15]], %[[REG22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG55:.*]] = xegpu.dpas %[[REG19]], %[[REG23]], %[[REG54]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG56:.*]] = xegpu.dpas %[[REG15]], %[[REG24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG57:.*]] = xegpu.dpas %[[REG19]], %[[REG25]], %[[REG56]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // CHECK: %[[REG58:.*]] = xegpu.dpas %[[REG15]], %[[REG26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    // CHECK: %[[REG59:.*]] = xegpu.dpas %[[REG19]], %[[REG27]], %[[REG58]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %8 = xetile.tile_mma %5, %7 : vector<4x2x8x16xf16>, vector<2x4x16x16xf16> -> vector<4x4x8x16xf32>

    gpu.return
  }
}
