// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -o -| FileCheck %s
// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {
  // CHECK: gpu.func @test_gemm(%[[A:.*]]: memref<1024x1024xf16>, %[[B:.*]]: memref<1024x1024xf16>, %[[C:.*]]: memref<1024x1024xf32>)
  gpu.func @test_gemm(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {

    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index

    //CHECK: %[[r0:.*]] = gpu.block_id  x
    //CHECK: %[[r1:.*]] = gpu.block_id  y
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y

    //CHECK: %[[r0:.*]] = arith.muli %block_id_x, %c64 : index
    //CHECK: %[[r1:.*]] = arith.muli %block_id_y, %c64 : index
    %m = arith.muli %block_id_x, %c64 : index
    %n = arith.muli %block_id_y, %c64 : index


    //CHECK: %[[r2:.*]] = arith.addi %[[r0]], %[[c0]] : index
    //CHECK: %[[r3:.*]] = arith.addi %[[r1]], %[[c0]] : index
    //CHECK: %[[r4:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[r5:.*]] = arith.addi %[[r1]], %[[c16]] : index
    //CHECK: %[[r6:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[r7:.*]] = arith.addi %[[r1]], %[[c32]] : index
    //CHECK: %[[r8:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c48:.*]] = arith.constant 48 : index
    //CHECK: %[[r9:.*]] = arith.addi %[[r1]], %[[c48]] : index
    //CHECK: %[[r10:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[r11:.*]] = arith.addi %[[r0]], %[[c8]] : index
    //CHECK: %[[r12:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r11]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r13:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r11]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r14:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r11]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r15:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r11]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r16:.*]] = arith.addi %[[r0]], %[[c16]] : index
    //CHECK: %[[r17:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r16]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r18:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r16]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r19:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r16]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r20:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r16]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[r21:.*]] = arith.addi %[[r0]], %[[c24]] : index
    //CHECK: %[[r22:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r21]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r23:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r21]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r24:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r21]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r25:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r21]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r26:.*]] = arith.addi %[[r0]], %[[c32]] : index
    //CHECK: %[[r27:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r28:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r29:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r30:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c40:.*]] = arith.constant 40 : index
    //CHECK: %[[r31:.*]] = arith.addi %[[r0]], %[[c40]] : index
    //CHECK: %[[r32:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r31]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r33:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r31]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r34:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r31]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r35:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r31]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r36:.*]] = arith.addi %[[r0]], %[[c48]] : index
    //CHECK: %[[r37:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r36]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r38:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r36]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r39:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r36]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r40:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r36]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c56:.*]] = arith.constant 56 : index
    //CHECK: %[[r41:.*]] = arith.addi %[[r0]], %[[c56]] : index
    //CHECK: %[[r42:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r41]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r43:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r41]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r44:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r41]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r45:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r41]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r46:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r47:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r48:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r49:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r2]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r50:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r51:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r52:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r7]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r53:.*]] = xegpu.create_nd_tdesc %[[C]][%[[r26]], %[[r9]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<64x64xf32>

    //CHECK: %[[r54:.*]] = xegpu.load_nd %[[r46]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r55:.*]] = xegpu.load_nd %[[r47]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r56:.*]] = xegpu.load_nd %[[r48]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r57:.*]] = xegpu.load_nd %[[r49]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r58:.*]] = xegpu.load_nd %[[r50]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r59:.*]] = xegpu.load_nd %[[r51]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r60:.*]] = xegpu.load_nd %[[r52]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r61:.*]] = xegpu.load_nd %[[r53]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x64xf32> -> vector<64x64xf32>

    //CHECK: %[[r62:.*]] = xegpu.create_nd_tdesc %[[A]][%[[r2]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r63:.*]] = xegpu.create_nd_tdesc %[[A]][%[[r2]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r64:.*]] = xegpu.create_nd_tdesc %[[A]][%[[r26]], %[[c0]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r65:.*]] = xegpu.create_nd_tdesc %[[A]][%[[r26]], %[[c32]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>

    //CHECK: %[[r66:.*]] = xegpu.create_nd_tdesc %[[B]][%[[c0]], %[[r3]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r67:.*]] = xegpu.create_nd_tdesc %[[B]][%[[c0]], %[[r7]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r68:.*]] = xegpu.create_nd_tdesc %[[B]][%[[c32]], %[[r3]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r69:.*]] = xegpu.create_nd_tdesc %[[B]][%[[c32]], %[[r7]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>

    //CHECK: %[[r72:.*]]:16 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c1024]] step %[[c64]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[r62]], %[[arg5:.*]] = %[[r63]], %[[arg6:.*]] = %[[r64]], %[[arg7:.*]] = %[[r65]], %[[arg8:.*]] = %[[r66]],
    //CHECK-SAME: %[[arg9:.*]] = %[[r67]], %[[arg10:.*]] = %[[r68]], %[[arg11:.*]] = %[[r69]], %[[arg12:.*]] = %[[r54]], %[[arg13:.*]] = %[[r55]],
    //CHECK-SAME: %[[arg14:.*]] = %[[r56]], %[[arg15:.*]] = %[[r57]], %[[arg16:.*]] = %[[r58]], %[[arg17:.*]] = %[[r59]], %[[arg18:.*]] = %[[r60]],
    //CHECK-SAME: %[[arg19:.*]] = %[[r61]]) -> (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
    //CHECK-SAME: vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>) {
    %out:3 = scf.for %k = %c0 to %c1024 step %c64
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>) {
      //CHECK: %[[r177:.*]] = vector.extract_strided_slice %[[arg12]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r178:.*]] = vector.extract_strided_slice %[[arg12]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r179:.*]] = vector.extract_strided_slice %[[arg12]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r180:.*]] = vector.extract_strided_slice %[[arg12]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r181:.*]] = vector.extract_strided_slice %[[arg13]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r182:.*]] = vector.extract_strided_slice %[[arg13]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r183:.*]] = vector.extract_strided_slice %[[arg13]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r184:.*]] = vector.extract_strided_slice %[[arg13]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r185:.*]] = vector.extract_strided_slice %[[arg14]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r186:.*]] = vector.extract_strided_slice %[[arg14]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r187:.*]] = vector.extract_strided_slice %[[arg14]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r188:.*]] = vector.extract_strided_slice %[[arg14]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r189:.*]] = vector.extract_strided_slice %[[arg15]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r190:.*]] = vector.extract_strided_slice %[[arg15]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r191:.*]] = vector.extract_strided_slice %[[arg15]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r192:.*]] = vector.extract_strided_slice %[[arg15]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r193:.*]] = vector.extract_strided_slice %[[arg16]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r194:.*]] = vector.extract_strided_slice %[[arg16]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r195:.*]] = vector.extract_strided_slice %[[arg16]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r196:.*]] = vector.extract_strided_slice %[[arg16]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r197:.*]] = vector.extract_strided_slice %[[arg17]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r198:.*]] = vector.extract_strided_slice %[[arg17]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r199:.*]] = vector.extract_strided_slice %[[arg17]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r200:.*]] = vector.extract_strided_slice %[[arg17]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r201:.*]] = vector.extract_strided_slice %[[arg18]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r202:.*]] = vector.extract_strided_slice %[[arg18]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r203:.*]] = vector.extract_strided_slice %[[arg18]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r204:.*]] = vector.extract_strided_slice %[[arg18]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r205:.*]] = vector.extract_strided_slice %[[arg19]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r206:.*]] = vector.extract_strided_slice %[[arg19]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r207:.*]] = vector.extract_strided_slice %[[arg19]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r208:.*]] = vector.extract_strided_slice %[[arg19]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>


      //CHECK: %[[r105:.*]] = xegpu.load_nd %[[arg4]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r106:.*]] = vector.extract %[[r105]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r107:.*]] = vector.extract %[[r105]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r108:.*]] = xegpu.load_nd %[[arg5]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r109:.*]] = vector.extract %[[r108]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r110:.*]] = vector.extract %[[r108]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r111:.*]] = xegpu.load_nd %[[arg6]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r112:.*]] = vector.extract %[[r111]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r113:.*]] = vector.extract %[[r111]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r114:.*]] = xegpu.load_nd %[[arg7]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r115:.*]] = vector.extract %[[r114]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r116:.*]] = vector.extract %[[r114]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r117:.*]] = vector.extract_strided_slice %[[r106]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r118:.*]] = vector.extract_strided_slice %[[r106]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r119:.*]] = vector.extract_strided_slice %[[r106]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r120:.*]] = vector.extract_strided_slice %[[r106]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r121:.*]] = vector.extract_strided_slice %[[r107]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r122:.*]] = vector.extract_strided_slice %[[r107]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r123:.*]] = vector.extract_strided_slice %[[r107]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r124:.*]] = vector.extract_strided_slice %[[r107]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r125:.*]] = vector.extract_strided_slice %[[r109]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r126:.*]] = vector.extract_strided_slice %[[r109]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r127:.*]] = vector.extract_strided_slice %[[r109]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r128:.*]] = vector.extract_strided_slice %[[r109]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r129:.*]] = vector.extract_strided_slice %[[r110]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r130:.*]] = vector.extract_strided_slice %[[r110]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r131:.*]] = vector.extract_strided_slice %[[r110]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r132:.*]] = vector.extract_strided_slice %[[r110]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r133:.*]] = vector.extract_strided_slice %[[r112]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r134:.*]] = vector.extract_strided_slice %[[r112]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r135:.*]] = vector.extract_strided_slice %[[r112]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r136:.*]] = vector.extract_strided_slice %[[r112]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r137:.*]] = vector.extract_strided_slice %[[r113]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r138:.*]] = vector.extract_strided_slice %[[r113]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r139:.*]] = vector.extract_strided_slice %[[r113]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r140:.*]] = vector.extract_strided_slice %[[r113]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r141:.*]] = vector.extract_strided_slice %[[r115]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r142:.*]] = vector.extract_strided_slice %[[r115]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r143:.*]] = vector.extract_strided_slice %[[r115]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r144:.*]] = vector.extract_strided_slice %[[r115]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r145:.*]] = vector.extract_strided_slice %[[r116]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r146:.*]] = vector.extract_strided_slice %[[r116]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r147:.*]] = vector.extract_strided_slice %[[r116]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r148:.*]] = vector.extract_strided_slice %[[r116]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>

      //CHECK: %[[r149:.*]] = xegpu.load_nd %[[arg8]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r150:.*]] = vector.extract %[[r149]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r151:.*]] = vector.extract %[[r149]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r152:.*]] = xegpu.load_nd %[[arg9]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r153:.*]] = vector.extract %[[r152]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r154:.*]] = vector.extract %[[r152]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r155:.*]] = xegpu.load_nd %[[arg10]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r156:.*]] = vector.extract %[[r155]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r157:.*]] = vector.extract %[[r155]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r158:.*]] = xegpu.load_nd %[[arg11]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK: %[[r159:.*]] = vector.extract %[[r158]][0] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r160:.*]] = vector.extract %[[r158]][1] : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: %[[r161:.*]] = vector.extract_strided_slice %[[r150]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r162:.*]] = vector.extract_strided_slice %[[r150]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r163:.*]] = vector.extract_strided_slice %[[r151]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r164:.*]] = vector.extract_strided_slice %[[r151]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r165:.*]] = vector.extract_strided_slice %[[r153]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r166:.*]] = vector.extract_strided_slice %[[r153]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r167:.*]] = vector.extract_strided_slice %[[r154]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r168:.*]] = vector.extract_strided_slice %[[r154]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r169:.*]] = vector.extract_strided_slice %[[r156]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r170:.*]] = vector.extract_strided_slice %[[r156]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r171:.*]] = vector.extract_strided_slice %[[r157]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r172:.*]] = vector.extract_strided_slice %[[r157]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r173:.*]] = vector.extract_strided_slice %[[r159]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r174:.*]] = vector.extract_strided_slice %[[r159]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r175:.*]] = vector.extract_strided_slice %[[r160]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r176:.*]] = vector.extract_strided_slice %[[r160]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>


      //CHECK: %[[r209:.*]] = xegpu.dpas %[[r117]], %[[r161]], %[[r177]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r210:.*]] = xegpu.dpas %[[r121]], %[[r162]], %[[r209]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r211:.*]] = xegpu.dpas %[[r125]], %[[r169]], %[[r210]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r212:.*]] = xegpu.dpas %[[r129]], %[[r170]], %[[r211]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r213:.*]] = xegpu.dpas %[[r117]], %[[r163]], %[[r181]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r214:.*]] = xegpu.dpas %[[r121]], %[[r164]], %[[r213]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r215:.*]] = xegpu.dpas %[[r125]], %[[r171]], %[[r214]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r216:.*]] = xegpu.dpas %[[r129]], %[[r172]], %[[r215]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r217:.*]] = xegpu.dpas %[[r117]], %[[r165]], %[[r185]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r218:.*]] = xegpu.dpas %[[r121]], %[[r166]], %[[r217]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r219:.*]] = xegpu.dpas %[[r125]], %[[r173]], %[[r218]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r220:.*]] = xegpu.dpas %[[r129]], %[[r174]], %[[r219]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r221:.*]] = xegpu.dpas %[[r117]], %[[r167]], %[[r189]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r222:.*]] = xegpu.dpas %[[r121]], %[[r168]], %[[r221]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r223:.*]] = xegpu.dpas %[[r125]], %[[r175]], %[[r222]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r224:.*]] = xegpu.dpas %[[r129]], %[[r176]], %[[r223]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r225:.*]] = xegpu.dpas %[[r118]], %[[r161]], %[[r178]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r226:.*]] = xegpu.dpas %[[r122]], %[[r162]], %[[r225]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r227:.*]] = xegpu.dpas %[[r126]], %[[r169]], %[[r226]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r228:.*]] = xegpu.dpas %[[r130]], %[[r170]], %[[r227]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r229:.*]] = xegpu.dpas %[[r118]], %[[r163]], %[[r182]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r230:.*]] = xegpu.dpas %[[r122]], %[[r164]], %[[r229]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r231:.*]] = xegpu.dpas %[[r126]], %[[r171]], %[[r230]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r232:.*]] = xegpu.dpas %[[r130]], %[[r172]], %[[r231]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r233:.*]] = xegpu.dpas %[[r118]], %[[r165]], %[[r186]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r234:.*]] = xegpu.dpas %[[r122]], %[[r166]], %[[r233]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r235:.*]] = xegpu.dpas %[[r126]], %[[r173]], %[[r234]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r236:.*]] = xegpu.dpas %[[r130]], %[[r174]], %[[r235]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r237:.*]] = xegpu.dpas %[[r118]], %[[r167]], %[[r190]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r238:.*]] = xegpu.dpas %[[r122]], %[[r168]], %[[r237]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r239:.*]] = xegpu.dpas %[[r126]], %[[r175]], %[[r238]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r240:.*]] = xegpu.dpas %[[r130]], %[[r176]], %[[r239]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r241:.*]] = xegpu.dpas %[[r119]], %[[r161]], %[[r179]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r242:.*]] = xegpu.dpas %[[r123]], %[[r162]], %[[r241]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r243:.*]] = xegpu.dpas %[[r127]], %[[r169]], %[[r242]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r244:.*]] = xegpu.dpas %[[r131]], %[[r170]], %[[r243]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r245:.*]] = xegpu.dpas %[[r119]], %[[r163]], %[[r183]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r246:.*]] = xegpu.dpas %[[r123]], %[[r164]], %[[r245]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r247:.*]] = xegpu.dpas %[[r127]], %[[r171]], %[[r246]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r248:.*]] = xegpu.dpas %[[r131]], %[[r172]], %[[r247]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r249:.*]] = xegpu.dpas %[[r119]], %[[r165]], %[[r187]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r250:.*]] = xegpu.dpas %[[r123]], %[[r166]], %[[r249]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r251:.*]] = xegpu.dpas %[[r127]], %[[r173]], %[[r250]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r252:.*]] = xegpu.dpas %[[r131]], %[[r174]], %[[r251]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r253:.*]] = xegpu.dpas %[[r119]], %[[r167]], %[[r191]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r254:.*]] = xegpu.dpas %[[r123]], %[[r168]], %[[r253]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r255:.*]] = xegpu.dpas %[[r127]], %[[r175]], %[[r254]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r256:.*]] = xegpu.dpas %[[r131]], %[[r176]], %[[r255]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r257:.*]] = xegpu.dpas %[[r120]], %[[r161]], %[[r180]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r258:.*]] = xegpu.dpas %[[r124]], %[[r162]], %[[r257]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r259:.*]] = xegpu.dpas %[[r128]], %[[r169]], %[[r258]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r260:.*]] = xegpu.dpas %[[r132]], %[[r170]], %[[r259]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r261:.*]] = xegpu.dpas %[[r120]], %[[r163]], %[[r184]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r262:.*]] = xegpu.dpas %[[r124]], %[[r164]], %[[r261]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r263:.*]] = xegpu.dpas %[[r128]], %[[r171]], %[[r262]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r264:.*]] = xegpu.dpas %[[r132]], %[[r172]], %[[r263]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r265:.*]] = xegpu.dpas %[[r120]], %[[r165]], %[[r188]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r266:.*]] = xegpu.dpas %[[r124]], %[[r166]], %[[r265]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r267:.*]] = xegpu.dpas %[[r128]], %[[r173]], %[[r266]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r268:.*]] = xegpu.dpas %[[r132]], %[[r174]], %[[r267]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r269:.*]] = xegpu.dpas %[[r120]], %[[r167]], %[[r192]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r270:.*]] = xegpu.dpas %[[r124]], %[[r168]], %[[r269]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r271:.*]] = xegpu.dpas %[[r128]], %[[r175]], %[[r270]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r272:.*]] = xegpu.dpas %[[r132]], %[[r176]], %[[r271]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r273:.*]] = xegpu.dpas %[[r133]], %[[r161]], %[[r193]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r274:.*]] = xegpu.dpas %[[r137]], %[[r162]], %[[r273]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r275:.*]] = xegpu.dpas %[[r141]], %[[r169]], %[[r274]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r276:.*]] = xegpu.dpas %[[r145]], %[[r170]], %[[r275]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r277:.*]] = xegpu.dpas %[[r133]], %[[r163]], %[[r197]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r278:.*]] = xegpu.dpas %[[r137]], %[[r164]], %[[r277]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r279:.*]] = xegpu.dpas %[[r141]], %[[r171]], %[[r278]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r280:.*]] = xegpu.dpas %[[r145]], %[[r172]], %[[r279]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r281:.*]] = xegpu.dpas %[[r133]], %[[r165]], %[[r201]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r282:.*]] = xegpu.dpas %[[r137]], %[[r166]], %[[r281]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r283:.*]] = xegpu.dpas %[[r141]], %[[r173]], %[[r282]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r284:.*]] = xegpu.dpas %[[r145]], %[[r174]], %[[r283]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r285:.*]] = xegpu.dpas %[[r133]], %[[r167]], %[[r205]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r286:.*]] = xegpu.dpas %[[r137]], %[[r168]], %[[r285]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r287:.*]] = xegpu.dpas %[[r141]], %[[r175]], %[[r286]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r288:.*]] = xegpu.dpas %[[r145]], %[[r176]], %[[r287]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r289:.*]] = xegpu.dpas %[[r134]], %[[r161]], %[[r194]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r290:.*]] = xegpu.dpas %[[r138]], %[[r162]], %[[r289]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r291:.*]] = xegpu.dpas %[[r142]], %[[r169]], %[[r290]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r292:.*]] = xegpu.dpas %[[r146]], %[[r170]], %[[r291]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r293:.*]] = xegpu.dpas %[[r134]], %[[r163]], %[[r198]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r294:.*]] = xegpu.dpas %[[r138]], %[[r164]], %[[r293]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r295:.*]] = xegpu.dpas %[[r142]], %[[r171]], %[[r294]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r296:.*]] = xegpu.dpas %[[r146]], %[[r172]], %[[r295]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r297:.*]] = xegpu.dpas %[[r134]], %[[r165]], %[[r202]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r298:.*]] = xegpu.dpas %[[r138]], %[[r166]], %[[r297]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r299:.*]] = xegpu.dpas %[[r142]], %[[r173]], %[[r298]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r300:.*]] = xegpu.dpas %[[r146]], %[[r174]], %[[r299]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r301:.*]] = xegpu.dpas %[[r134]], %[[r167]], %[[r206]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r302:.*]] = xegpu.dpas %[[r138]], %[[r168]], %[[r301]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r303:.*]] = xegpu.dpas %[[r142]], %[[r175]], %[[r302]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r304:.*]] = xegpu.dpas %[[r146]], %[[r176]], %[[r303]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r305:.*]] = xegpu.dpas %[[r135]], %[[r161]], %[[r195]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r306:.*]] = xegpu.dpas %[[r139]], %[[r162]], %[[r305]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r307:.*]] = xegpu.dpas %[[r143]], %[[r169]], %[[r306]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r308:.*]] = xegpu.dpas %[[r147]], %[[r170]], %[[r307]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r309:.*]] = xegpu.dpas %[[r135]], %[[r163]], %[[r199]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r310:.*]] = xegpu.dpas %[[r139]], %[[r164]], %[[r309]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r311:.*]] = xegpu.dpas %[[r143]], %[[r171]], %[[r310]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r312:.*]] = xegpu.dpas %[[r147]], %[[r172]], %[[r311]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r313:.*]] = xegpu.dpas %[[r135]], %[[r165]], %[[r203]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r314:.*]] = xegpu.dpas %[[r139]], %[[r166]], %[[r313]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r315:.*]] = xegpu.dpas %[[r143]], %[[r173]], %[[r314]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r316:.*]] = xegpu.dpas %[[r147]], %[[r174]], %[[r315]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r317:.*]] = xegpu.dpas %[[r135]], %[[r167]], %[[r207]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r318:.*]] = xegpu.dpas %[[r139]], %[[r168]], %[[r317]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r319:.*]] = xegpu.dpas %[[r143]], %[[r175]], %[[r318]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r320:.*]] = xegpu.dpas %[[r147]], %[[r176]], %[[r319]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r321:.*]] = xegpu.dpas %[[r136]], %[[r161]], %[[r196]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r322:.*]] = xegpu.dpas %[[r140]], %[[r162]], %[[r321]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r323:.*]] = xegpu.dpas %[[r144]], %[[r169]], %[[r322]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r324:.*]] = xegpu.dpas %[[r148]], %[[r170]], %[[r323]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r325:.*]] = xegpu.dpas %[[r136]], %[[r163]], %[[r200]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r326:.*]] = xegpu.dpas %[[r140]], %[[r164]], %[[r325]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r327:.*]] = xegpu.dpas %[[r144]], %[[r171]], %[[r326]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r328:.*]] = xegpu.dpas %[[r148]], %[[r172]], %[[r327]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r329:.*]] = xegpu.dpas %[[r136]], %[[r165]], %[[r204]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r330:.*]] = xegpu.dpas %[[r140]], %[[r166]], %[[r329]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r331:.*]] = xegpu.dpas %[[r144]], %[[r173]], %[[r330]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r332:.*]] = xegpu.dpas %[[r148]], %[[r174]], %[[r331]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r333:.*]] = xegpu.dpas %[[r136]], %[[r167]], %[[r208]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r334:.*]] = xegpu.dpas %[[r140]], %[[r168]], %[[r333]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r335:.*]] = xegpu.dpas %[[r144]], %[[r175]], %[[r334]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r336:.*]] = xegpu.dpas %[[r148]], %[[r176]], %[[r335]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf32> -> vector<64x64xf32>

      //CHECK: %[[r337:.*]] = vector.shuffle %[[r212]], %[[r228]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r338:.*]] = vector.shuffle %[[r244]], %[[r260]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r339:.*]] = vector.shuffle %[[r337]], %[[r338]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r340:.*]] = vector.shuffle %[[r276]], %[[r292]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r341:.*]] = vector.shuffle %[[r308]], %[[r324]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r342:.*]] = vector.shuffle %[[r340]], %[[r341]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r343:.*]] = vector.shuffle %[[r216]], %[[r232]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r344:.*]] = vector.shuffle %[[r248]], %[[r264]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r345:.*]] = vector.shuffle %[[r343]], %[[r344]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r346:.*]] = vector.shuffle %[[r280]], %[[r296]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r347:.*]] = vector.shuffle %[[r312]], %[[r328]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r348:.*]] = vector.shuffle %[[r346]], %[[r347]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r349:.*]] = vector.shuffle %[[r220]], %[[r236]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r350:.*]] = vector.shuffle %[[r252]], %[[r268]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r351:.*]] = vector.shuffle %[[r349]], %[[r350]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r352:.*]] = vector.shuffle %[[r284]], %[[r300]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r353:.*]] = vector.shuffle %[[r316]], %[[r332]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r354:.*]] = vector.shuffle %[[r352]], %[[r353]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r355:.*]] = vector.shuffle %[[r224]], %[[r240]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r356:.*]] = vector.shuffle %[[r256]], %[[r272]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r357:.*]] = vector.shuffle %[[r355]], %[[r356]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r358:.*]] = vector.shuffle %[[r288]], %[[r304]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r359:.*]] = vector.shuffle %[[r320]], %[[r336]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r360:.*]] = vector.shuffle %[[r358]], %[[r359]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r361:.*]] = xegpu.update_nd_offset %[[arg4]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r362:.*]] = xegpu.update_nd_offset %[[arg5]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r363:.*]] = xegpu.update_nd_offset %[[arg6]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r364:.*]] = xegpu.update_nd_offset %[[arg7]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r365:.*]] = xegpu.update_nd_offset %[[arg8]], [%[[c64]], %[[c0]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r366:.*]] = xegpu.update_nd_offset %[[arg9]], [%[[c64]], %[[c0]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r367:.*]] = xegpu.update_nd_offset %[[arg10]], [%[[c64]], %[[c0]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r368:.*]] = xegpu.update_nd_offset %[[arg11]], [%[[c64]], %[[c0]]] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64] : !xetile.tile<64x64xf16>, index, index -> !xetile.tile<64x64xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0] : !xetile.tile<64x64xf16>, index, index -> !xetile.tile<64x64xf16>

      //CHECK: scf.yield %[[r361]], %[[r362]], %[[r363]], %[[r364]], %[[r365]], %[[r366]], %[[r367]], %[[r368]], %[[r339]], %[[r345]], %[[r351]], %[[r357]], %[[r342]], %[[r348]], %[[r354]], %[[r360]]
      //CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
      //CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
      //CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, vector<32x16xf32>, vector<32x16xf32>,
      //CHECK-SAME: vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>, vector<32x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>
    }
    //CHECK: %[[r73:.*]] = vector.extract_strided_slice %[[r72]]#8 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r74:.*]] = vector.extract_strided_slice %[[r72]]#8 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r75:.*]] = vector.extract_strided_slice %[[r72]]#8 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r76:.*]] = vector.extract_strided_slice %[[r72]]#8 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r77:.*]] = vector.extract_strided_slice %[[r72]]#9 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r78:.*]] = vector.extract_strided_slice %[[r72]]#9 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r79:.*]] = vector.extract_strided_slice %[[r72]]#9 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r80:.*]] = vector.extract_strided_slice %[[r72]]#9 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r81:.*]] = vector.extract_strided_slice %[[r72]]#10 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r82:.*]] = vector.extract_strided_slice %[[r72]]#10 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r83:.*]] = vector.extract_strided_slice %[[r72]]#10 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r84:.*]] = vector.extract_strided_slice %[[r72]]#10 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r85:.*]] = vector.extract_strided_slice %[[r72]]#11 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r86:.*]] = vector.extract_strided_slice %[[r72]]#11 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r87:.*]] = vector.extract_strided_slice %[[r72]]#11 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r88:.*]] = vector.extract_strided_slice %[[r72]]#11 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r89:.*]] = vector.extract_strided_slice %[[r72]]#12 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r90:.*]] = vector.extract_strided_slice %[[r72]]#12 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r91:.*]] = vector.extract_strided_slice %[[r72]]#12 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r92:.*]] = vector.extract_strided_slice %[[r72]]#12 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r93:.*]] = vector.extract_strided_slice %[[r72]]#13 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r94:.*]] = vector.extract_strided_slice %[[r72]]#13 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r95:.*]] = vector.extract_strided_slice %[[r72]]#13 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r96:.*]] = vector.extract_strided_slice %[[r72]]#13 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r97:.*]] = vector.extract_strided_slice %[[r72]]#14 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r98:.*]] = vector.extract_strided_slice %[[r72]]#14 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r99:.*]] = vector.extract_strided_slice %[[r72]]#14 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r100:.*]] = vector.extract_strided_slice %[[r72]]#14 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r101:.*]] = vector.extract_strided_slice %[[r72]]#15 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r102:.*]] = vector.extract_strided_slice %[[r72]]#15 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r103:.*]] = vector.extract_strided_slice %[[r72]]#15 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r104:.*]] = vector.extract_strided_slice %[[r72]]#15 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: xegpu.store_nd %[[r73]], %[[r4]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r77]], %[[r6]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r81]], %[[r8]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r85]], %[[r10]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r74]], %[[r12]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r78]], %[[r13]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r82]], %[[r14]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r86]], %[[r15]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r75]], %[[r17]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r79]], %[[r18]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r83]], %[[r19]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r87]], %[[r20]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r76]], %[[r22]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r80]], %[[r23]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r84]], %[[r24]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r88]], %[[r25]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r89]], %[[r27]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r93]], %[[r28]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r97]], %[[r29]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r101]], %[[r30]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r90]], %[[r32]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r94]], %[[r33]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r98]], %[[r34]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r102]], %[[r35]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r91]], %[[r37]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r95]], %[[r38]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r99]], %[[r39]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r103]], %[[r40]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r92]], %[[r42]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r96]], %[[r43]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r100]], %[[r44]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r104]], %[[r45]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    xetile.store_tile %out#2, %c_init_tile: vector<64x64xf32>, !xetile.tile<64x64xf32>

    gpu.return
  }
}
