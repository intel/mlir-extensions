// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking="enable-2d-transform=true" \
// RUN: --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  //CHECK: s_tiled_gemm(%[[arg0:.*]]: memref<1024x1024xf16>, %[[arg1:.*]]: memref<1024x1024xf16>)
  gpu.func @s_tiled_gemm(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>) {
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index

    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
  	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    //CHECK: %[[r2:.*]] = vector.extract %[[r1]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r3:.*]] = vector.extract %[[r1]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r4:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r10:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r11:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r12:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c64]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r13:.*]] = xegpu.create_nd_tdesc %arg1[%[[c64]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
  	%3 = xetile.init_tile %b[%c64, %c0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

    //CHECK: %[[r14:.*]] = xegpu.load_nd %[[r12]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    //CHECK: %[[r15:.*]] = vector.extract %[[r14]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r16:.*]] = vector.extract %[[r14]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r17:.*]] = xegpu.load_nd %[[r13]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    //CHECK: %[[r18:.*]] = vector.extract %[[r17]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r19:.*]] = vector.extract %[[r17]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r20:.*]] = vector.extract_strided_slice %[[r15]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r21:.*]] = vector.extract_strided_slice %[[r15]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r22:.*]] = vector.extract_strided_slice %[[r16]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r23:.*]] = vector.extract_strided_slice %[[r16]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r24:.*]] = vector.extract_strided_slice %[[r18]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r18]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r19]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r19]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    %4 = xetile.load_tile %3 : !xetile.tile<32x64xf16> -> vector<32x64xf16>

    //CHECK: %[[r28:.*]] = xegpu.dpas %[[r4]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r29:.*]] = xegpu.dpas %[[r8]], %[[r21]], %[[r28]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r30:.*]] = xegpu.dpas %[[r4]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r31:.*]] = xegpu.dpas %[[r8]], %[[r23]], %[[r30]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r32:.*]] = xegpu.dpas %[[r4]], %[[r24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r33:.*]] = xegpu.dpas %[[r8]], %[[r25]], %[[r32]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r34:.*]] = xegpu.dpas %[[r4]], %[[r26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r35:.*]] = xegpu.dpas %[[r8]], %[[r27]], %[[r34]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r36:.*]] = xegpu.dpas %[[r5]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r37:.*]] = xegpu.dpas %[[r9]], %[[r21]], %[[r36]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r38:.*]] = xegpu.dpas %[[r5]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r39:.*]] = xegpu.dpas %[[r9]], %[[r23]], %[[r38]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r40:.*]] = xegpu.dpas %[[r5]], %[[r24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r41:.*]] = xegpu.dpas %[[r9]], %[[r25]], %[[r40]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r42:.*]] = xegpu.dpas %[[r5]], %[[r26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r43:.*]] = xegpu.dpas %[[r9]], %[[r27]], %[[r42]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r44:.*]] = xegpu.dpas %[[r6]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r45:.*]] = xegpu.dpas %[[r10]], %[[r21]], %[[r44]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r46:.*]] = xegpu.dpas %[[r6]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r47:.*]] = xegpu.dpas %[[r10]], %[[r23]], %[[r46]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r48:.*]] = xegpu.dpas %[[r6]], %[[r24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r49:.*]] = xegpu.dpas %[[r10]], %[[r25]], %[[r48]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r50:.*]] = xegpu.dpas %[[r6]], %[[r26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r51:.*]] = xegpu.dpas %[[r10]], %[[r27]], %[[r50]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r52:.*]] = xegpu.dpas %[[r7]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r53:.*]] = xegpu.dpas %[[r11]], %[[r21]], %[[r52]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r54:.*]] = xegpu.dpas %[[r7]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r55:.*]] = xegpu.dpas %[[r11]], %[[r23]], %[[r54]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r56:.*]] = xegpu.dpas %[[r7]], %[[r24]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r57:.*]] = xegpu.dpas %[[r11]], %[[r25]], %[[r56]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r58:.*]] = xegpu.dpas %[[r7]], %[[r26]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r59:.*]] = xegpu.dpas %[[r11]], %[[r27]], %[[r58]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %6 = xetile.tile_mma %2, %4: vector<32x32xf16>, vector<32x64xf16> -> vector<32x64xf32>
  	gpu.return
  }
}
