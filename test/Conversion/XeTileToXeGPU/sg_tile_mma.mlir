// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
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
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>

    //CHECK: %[[r4:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c64]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r5:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c64]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
  	%3 = xetile.init_tile %b[%c64, %c0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

    //CHECK: %[[r6:.*]] = xegpu.load_nd %[[r4]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    //CHECK: %[[r7:.*]] = vector.extract %[[r6]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r8:.*]] = vector.extract %[[r6]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r9:.*]] = xegpu.load_nd %[[r5]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
    //CHECK: %[[r10:.*]] = vector.extract %[[r9]][0] : vector<32x16xf16> from vector<2x32x16xf16>
    //CHECK: %[[r11:.*]] = vector.extract %[[r9]][1] : vector<32x16xf16> from vector<2x32x16xf16>
    %4 = xetile.load_tile %3 : !xetile.tile<32x64xf16> -> vector<32x64xf16>


    //CHECK: %[[r12:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r13:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r14:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r18:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r19:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK: %[[r20:.*]] = vector.extract_strided_slice %[[r7]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r21:.*]] = vector.extract_strided_slice %[[r8]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r22:.*]] = vector.extract_strided_slice %[[r10]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r23:.*]] = vector.extract_strided_slice %[[r11]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r24:.*]] = vector.extract_strided_slice %[[r7]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r8]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r10]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r11]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK: %[[r28:.*]] = xegpu.dpas %[[r12]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r29:.*]] = xegpu.dpas %[[r13]], %[[r24]], %[[r28]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r30:.*]] = xegpu.dpas %[[r12]], %[[r21]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r31:.*]] = xegpu.dpas %[[r13]], %[[r25]], %[[r30]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r32:.*]] = xegpu.dpas %[[r12]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r33:.*]] = xegpu.dpas %[[r13]], %[[r26]], %[[r32]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r34:.*]] = xegpu.dpas %[[r12]], %[[r23]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r35:.*]] = xegpu.dpas %[[r13]], %[[r27]], %[[r34]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r36:.*]] = xegpu.dpas %[[r14]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r37:.*]] = xegpu.dpas %[[r15]], %[[r24]], %[[r36]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r38:.*]] = xegpu.dpas %[[r14]], %[[r21]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r39:.*]] = xegpu.dpas %[[r15]], %[[r25]], %[[r38]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r40:.*]] = xegpu.dpas %[[r14]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r41:.*]] = xegpu.dpas %[[r15]], %[[r26]], %[[r40]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r42:.*]] = xegpu.dpas %[[r14]], %[[r23]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r43:.*]] = xegpu.dpas %[[r15]], %[[r27]], %[[r42]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r44:.*]] = xegpu.dpas %[[r16]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r45:.*]] = xegpu.dpas %[[r17]], %[[r24]], %[[r44]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r46:.*]] = xegpu.dpas %[[r16]], %[[r21]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r47:.*]] = xegpu.dpas %[[r17]], %[[r25]], %[[r46]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r48:.*]] = xegpu.dpas %[[r16]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r49:.*]] = xegpu.dpas %[[r17]], %[[r26]], %[[r48]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r50:.*]] = xegpu.dpas %[[r16]], %[[r23]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r51:.*]] = xegpu.dpas %[[r17]], %[[r27]], %[[r50]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r52:.*]] = xegpu.dpas %[[r18]], %[[r20]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r53:.*]] = xegpu.dpas %[[r19]], %[[r24]], %[[r52]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r54:.*]] = xegpu.dpas %[[r18]], %[[r21]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r55:.*]] = xegpu.dpas %[[r19]], %[[r25]], %[[r54]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r56:.*]] = xegpu.dpas %[[r18]], %[[r22]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r57:.*]] = xegpu.dpas %[[r19]], %[[r26]], %[[r56]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r58:.*]] = xegpu.dpas %[[r18]], %[[r23]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    //CHECK: %[[r59:.*]] = xegpu.dpas %[[r19]], %[[r27]], %[[r58]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %6 = xetile.tile_mma %2, %4: vector<32x32xf16>, vector<32x64xf16> -> vector<32x64xf32>
  	gpu.return
  }
}
