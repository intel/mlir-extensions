// RUN: imex-opt --split-input-file --xetile-blocking --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  //CHECK: s_tiled_gemm(%[[arg0:.*]]: memref<1024x1024xf16>, %[[arg1:.*]]: memref<1024x1024xf16>)
  gpu.func @s_tiled_gemm(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    %c64 = arith.constant 64 : index

    //CHECK: %[[r0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c64]]] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16>
  	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[r1:.*]] = xegpu.load_nd %[[r0]] {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<32x32xf16> -> vector<32x16x2xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>

    //CHECK: %[[r2:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c64]], %[[c0]]] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16>
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[r3:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c64]], %[[c32]]] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16>
  	%3 = xetile.init_tile %b[%c64, %c0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

    //CHECK: %[[r4:.*]] = xegpu.load_nd %[[r2]] {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<32x32xf16> -> vector<16x32x2xf16>
    //CHECK: %[[r5:.*]] = xegpu.load_nd %[[r3]] {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<32x32xf16> -> vector<16x32x2xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r10:.*]] = vector.extract_strided_slice %[[r6]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r11:.*]] = vector.extract_strided_slice %[[r6]] {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r12:.*]] = vector.extract_strided_slice %[[r7]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r13:.*]] = vector.extract_strided_slice %[[r7]] {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r14:.*]] = vector.extract_strided_slice %[[r8]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r8]] {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r9]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r9]] {offsets = [0, 8], sizes = [8, 8], strides = [1, 1]} : vector<8x16x2xf16> to vector<8x8x2xf16>
    //CHECK: %[[r18:.*]] = vector.extract_strided_slice %[[r4]] {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<16x32x2xf16> to vector<8x32x2xf16>
    //CHECK: %[[r19:.*]] = vector.extract_strided_slice %[[r4]] {offsets = [8, 0], sizes = [8, 32], strides = [1, 1]} : vector<16x32x2xf16> to vector<8x32x2xf16>
    //CHECK: %[[r20:.*]] = vector.extract_strided_slice %[[r5]] {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<16x32x2xf16> to vector<8x32x2xf16>
    //CHECK: %[[r21:.*]] = vector.extract_strided_slice %[[r5]] {offsets = [8, 0], sizes = [8, 32], strides = [1, 1]} : vector<16x32x2xf16> to vector<8x32x2xf16>
    //CHECK: %[[r22:.*]] = vector.extract_strided_slice %[[r18]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r23:.*]] = vector.extract_strided_slice %[[r18]] {offsets = [0, 16], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r24:.*]] = vector.extract_strided_slice %[[r20]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r20]] {offsets = [0, 16], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r19]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r19]] {offsets = [0, 16], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r28:.*]] = vector.extract_strided_slice %[[r21]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    //CHECK: %[[r29:.*]] = vector.extract_strided_slice %[[r21]] {offsets = [0, 16], sizes = [8, 16], strides = [1, 1]} : vector<8x32x2xf16> to vector<8x16x2xf16>
    %4 = xetile.load_tile %3 : !xetile.tile<32x64xf16> -> vector<32x64xf16>


    //CHECK: %[[r30:.*]] = xegpu.dpas %[[r10]], %[[r22]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r31:.*]] = xegpu.dpas %[[r11]], %[[r26]], %[[r30]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r32:.*]] = xegpu.dpas %[[r10]], %[[r23]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r33:.*]] = xegpu.dpas %[[r11]], %[[r27]], %[[r32]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r34:.*]] = xegpu.dpas %[[r10]], %[[r24]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r35:.*]] = xegpu.dpas %[[r11]], %[[r28]], %[[r34]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r36:.*]] = xegpu.dpas %[[r10]], %[[r25]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r37:.*]] = xegpu.dpas %[[r11]], %[[r29]], %[[r36]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r38:.*]] = xegpu.dpas %[[r12]], %[[r22]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r39:.*]] = xegpu.dpas %[[r13]], %[[r26]], %[[r38]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r40:.*]] = xegpu.dpas %[[r12]], %[[r23]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r41:.*]] = xegpu.dpas %[[r13]], %[[r27]], %[[r40]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r42:.*]] = xegpu.dpas %[[r12]], %[[r24]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r43:.*]] = xegpu.dpas %[[r13]], %[[r28]], %[[r42]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r44:.*]] = xegpu.dpas %[[r12]], %[[r25]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r45:.*]] = xegpu.dpas %[[r13]], %[[r29]], %[[r44]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r46:.*]] = xegpu.dpas %[[r14]], %[[r22]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r47:.*]] = xegpu.dpas %[[r15]], %[[r26]], %[[r46]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r48:.*]] = xegpu.dpas %[[r14]], %[[r23]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r49:.*]] = xegpu.dpas %[[r15]], %[[r27]], %[[r48]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r50:.*]] = xegpu.dpas %[[r14]], %[[r24]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r51:.*]] = xegpu.dpas %[[r15]], %[[r28]], %[[r50]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r52:.*]] = xegpu.dpas %[[r14]], %[[r25]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r53:.*]] = xegpu.dpas %[[r15]], %[[r29]], %[[r52]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r54:.*]] = xegpu.dpas %[[r16]], %[[r22]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r55:.*]] = xegpu.dpas %[[r17]], %[[r26]], %[[r54]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r56:.*]] = xegpu.dpas %[[r16]], %[[r23]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r57:.*]] = xegpu.dpas %[[r17]], %[[r27]], %[[r56]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r58:.*]] = xegpu.dpas %[[r16]], %[[r24]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r59:.*]] = xegpu.dpas %[[r17]], %[[r28]], %[[r58]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK: %[[r60:.*]] = xegpu.dpas %[[r16]], %[[r25]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    //CHECK: %[[r61:.*]] = xegpu.dpas %[[r17]], %[[r29]], %[[r60]] {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %6 = xetile.tile_mma %2, %4: vector<32x32xf16>, vector<32x64xf16> -> vector<32x64xf32>
  	gpu.return
  }
}
