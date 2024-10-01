// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -o -| FileCheck %s
gpu.module @test_kernel {
  //CHECK-LABEL: test_gemm
  //CHECK-SAME: %[[arg0:.*]]: memref<1024x1024xtf32>, %[[arg1:.*]]: memref<1024x1024xtf32>, %[[arg2:.*]]: memref<1024x1024xf32>
  gpu.func @test_gemm(%arg0: memref<1024x1024xtf32>, %arg1: memref<1024x1024xtf32>, %arg2: memref<1024x1024xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index

    //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
    //CHECK: %[[block_id_y:.*]] = gpu.block_id  y
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y

    //CHECK: %[[r0:.*]] = arith.muli %[[block_id_x]], %[[c64]] : index
    //CHECK: %[[r1:.*]] = arith.muli %[[block_id_y]], %[[c64]] : index
    %0 = arith.muli %block_id_x, %c64 : index
    %1 = arith.muli %block_id_y, %c64 : index

    //CHECK: %[[r2:.*]] = arith.addi %[[r0]], %[[c0]] : index
    //CHECK: %[[r3:.*]] = arith.addi %[[r1]], %[[c0]] : index
    //CHECK: %[[r4:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r2]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[r5:.*]] = arith.addi %[[r1]], %[[c16]] : index
    //CHECK: %[[r6:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r2]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[r7:.*]] = arith.addi %[[r0]], %[[c8]] : index
    //CHECK: %[[r8:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r7]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r9:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r7]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r10:.*]] = arith.addi %[[r0]], %[[c16]] : index
    //CHECK: %[[r11:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r10]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r12:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r10]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[r13:.*]] = arith.addi %[[r0]], %[[c24]] : index
    //CHECK: %[[r14:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r13]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r15:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r13]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r16:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r2]], %[[r3]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r17:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r2]], %[[r5]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>

    //CHECK: %[[r18:.*]] = xegpu.load_nd %[[r16]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK: %[[r19:.*]] = xegpu.load_nd %[[r17]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    %3 = xetile.load_tile %2 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf32> -> vector<32x32xf32>

    //CHECK: %[[r20:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r2]], %[[c0]]] : memref<1024x1024xtf32> -> !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    //CHECK: %[[r21:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r2]], %[[c16]]] : memref<1024x1024xtf32> -> !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
    %4 = xetile.init_tile %arg0[%0, %c0] : memref<1024x1024xtf32> -> !xetile.tile<32x32xtf32>

    //CHECK: %[[r22:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[r3]]] : memref<1024x1024xtf32> -> !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r23:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[r5]]] : memref<1024x1024xtf32> -> !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    %5 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xtf32> -> !xetile.tile<32x32xtf32>

    //CHECK: %[[r24:.*]]:6 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c1024]] step %[[c64]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[r20]], %[[arg5:.*]] = %[[r21]], %[[arg6:.*]] = %[[r22]], %[[arg7:.*]] = %[[r23]], %[[arg8:.*]] = %[[r18]], %[[arg9:.*]] = %[[r19]])
    //CHECK-SAME: !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>, vector<32x16xf32>, vector<32x16xf32>
    %6:3 = scf.for %arg3 = %c0 to %c1024 step %c64 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3) -> (!xetile.tile<32x32xtf32>, !xetile.tile<32x32xtf32>, vector<32x32xf32>) {
      //CHECK: %[[r65:.*]] = vector.extract_strided_slice %[[arg8]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r66:.*]] = vector.extract_strided_slice %[[arg8]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r67:.*]] = vector.extract_strided_slice %[[arg8]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r68:.*]] = vector.extract_strided_slice %[[arg8]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r69:.*]] = vector.extract_strided_slice %[[arg9]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r70:.*]] = vector.extract_strided_slice %[[arg9]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r71:.*]] = vector.extract_strided_slice %[[arg9]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
      //CHECK: %[[r72:.*]] = vector.extract_strided_slice %[[arg9]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>

      //CHECK: %[[r33:.*]] = xegpu.load_nd %[[arg4]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x8xtf32>
      //CHECK: %[[r34:.*]] = vector.extract %[[r33]][0] : vector<32x8xtf32> from vector<2x32x8xtf32>
      //CHECK: %[[r35:.*]] = vector.extract %[[r33]][1] : vector<32x8xtf32> from vector<2x32x8xtf32>
      //CHECK: %[[r36:.*]] = xegpu.load_nd %[[arg5]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x8xtf32>
      //CHECK: %[[r37:.*]] = vector.extract %[[r36]][0] : vector<32x8xtf32> from vector<2x32x8xtf32>
      //CHECK: %[[r38:.*]] = vector.extract %[[r36]][1] : vector<32x8xtf32> from vector<2x32x8xtf32>
      //CHECK: %[[r39:.*]] = vector.extract_strided_slice %[[r34]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r40:.*]] = vector.extract_strided_slice %[[r34]] {offsets = [8, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r41:.*]] = vector.extract_strided_slice %[[r34]] {offsets = [16, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r42:.*]] = vector.extract_strided_slice %[[r34]] {offsets = [24, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r43:.*]] = vector.extract_strided_slice %[[r35]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r44:.*]] = vector.extract_strided_slice %[[r35]] {offsets = [8, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r45:.*]] = vector.extract_strided_slice %[[r35]] {offsets = [16, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r46:.*]] = vector.extract_strided_slice %[[r35]] {offsets = [24, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r47:.*]] = vector.extract_strided_slice %[[r37]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r48:.*]] = vector.extract_strided_slice %[[r37]] {offsets = [8, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r49:.*]] = vector.extract_strided_slice %[[r37]] {offsets = [16, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r50:.*]] = vector.extract_strided_slice %[[r37]] {offsets = [24, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r51:.*]] = vector.extract_strided_slice %[[r38]] {offsets = [0, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r52:.*]] = vector.extract_strided_slice %[[r38]] {offsets = [8, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r53:.*]] = vector.extract_strided_slice %[[r38]] {offsets = [16, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      //CHECK: %[[r54:.*]] = vector.extract_strided_slice %[[r38]] {offsets = [24, 0], sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      %7 = xetile.load_tile %arg4 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xtf32> -> vector<32x32xtf32>

      //CHECK: %[[r55:.*]] = xegpu.load_nd %[[arg6]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xtf32>
      //CHECK: %[[r56:.*]] = xegpu.load_nd %[[arg7]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xtf32>
      //CHECK: %[[r57:.*]] = vector.extract_strided_slice %[[r55]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      //CHECK: %[[r58:.*]] = vector.extract_strided_slice %[[r55]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      //CHECK: %[[r59:.*]] = vector.extract_strided_slice %[[r55]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      //CHECK: %[[r60:.*]] = vector.extract_strided_slice %[[r55]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      //CHECK: %[[r61:.*]] = vector.extract_strided_slice %[[r56]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      //CHECK: %[[r62:.*]] = vector.extract_strided_slice %[[r56]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      //CHECK: %[[r63:.*]] = vector.extract_strided_slice %[[r56]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      //CHECK: %[[r64:.*]] = vector.extract_strided_slice %[[r56]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>

      %8 = xetile.load_tile %arg5 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xtf32> -> vector<32x32xtf32>

      //CHECK: %[[r73:.*]] = xegpu.dpas %[[r39]], %[[r57]], %[[r65]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r74:.*]] = xegpu.dpas %[[r43]], %[[r58]], %[[r73]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r75:.*]] = xegpu.dpas %[[r47]], %[[r59]], %[[r74]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r76:.*]] = xegpu.dpas %[[r51]], %[[r60]], %[[r75]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r77:.*]] = xegpu.dpas %[[r39]], %[[r61]], %[[r69]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r78:.*]] = xegpu.dpas %[[r43]], %[[r62]], %[[r77]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r79:.*]] = xegpu.dpas %[[r47]], %[[r63]], %[[r78]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r80:.*]] = xegpu.dpas %[[r51]], %[[r64]], %[[r79]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r81:.*]] = xegpu.dpas %[[r40]], %[[r57]], %[[r66]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r82:.*]] = xegpu.dpas %[[r44]], %[[r58]], %[[r81]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r83:.*]] = xegpu.dpas %[[r48]], %[[r59]], %[[r82]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r84:.*]] = xegpu.dpas %[[r52]], %[[r60]], %[[r83]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r85:.*]] = xegpu.dpas %[[r40]], %[[r61]], %[[r70]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r86:.*]] = xegpu.dpas %[[r44]], %[[r62]], %[[r85]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r87:.*]] = xegpu.dpas %[[r48]], %[[r63]], %[[r86]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r88:.*]] = xegpu.dpas %[[r52]], %[[r64]], %[[r87]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r89:.*]] = xegpu.dpas %[[r41]], %[[r57]], %[[r67]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r90:.*]] = xegpu.dpas %[[r45]], %[[r58]], %[[r89]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r91:.*]] = xegpu.dpas %[[r49]], %[[r59]], %[[r90]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r92:.*]] = xegpu.dpas %[[r53]], %[[r60]], %[[r91]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r93:.*]] = xegpu.dpas %[[r41]], %[[r61]], %[[r71]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r94:.*]] = xegpu.dpas %[[r45]], %[[r62]], %[[r93]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r95:.*]] = xegpu.dpas %[[r49]], %[[r63]], %[[r94]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r96:.*]] = xegpu.dpas %[[r53]], %[[r64]], %[[r95]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r97:.*]] = xegpu.dpas %[[r42]], %[[r57]], %[[r68]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r98:.*]] = xegpu.dpas %[[r46]], %[[r58]], %[[r97]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r99:.*]] = xegpu.dpas %[[r50]], %[[r59]], %[[r98]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r100:.*]] = xegpu.dpas %[[r54]], %[[r60]], %[[r99]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r101:.*]] = xegpu.dpas %[[r42]], %[[r61]], %[[r72]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r102:.*]] = xegpu.dpas %[[r46]], %[[r62]], %[[r101]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r103:.*]] = xegpu.dpas %[[r50]], %[[r63]], %[[r102]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r104:.*]] = xegpu.dpas %[[r54]], %[[r64]], %[[r103]] : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r105:.*]] = vector.shuffle %76, %84 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r106:.*]] = vector.shuffle %92, %100 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r107:.*]] = vector.shuffle %105, %106 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>
      //CHECK: %[[r108:.*]] = vector.shuffle %80, %88 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r109:.*]] = vector.shuffle %96, %104 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x16xf32>, vector<8x16xf32>
      //CHECK: %[[r110:.*]] = vector.shuffle %108, %109 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x16xf32>, vector<16x16xf32>


      %9 = xetile.tile_mma %7, %8, %arg6 : vector<32x32xtf32>, vector<32x32xtf32>, vector<32x32xf32> -> vector<32x32xf32>

      //CHECK: %[[r111:.*]] = xegpu.update_nd_offset %[[arg4]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      //CHECK: %[[r112:.*]] = xegpu.update_nd_offset %[[arg5]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      %10 = xetile.update_tile_offset %arg4, [%c0,  %c64] : !xetile.tile<32x32xtf32>, index, index -> !xetile.tile<32x32xtf32>

      //CHECK: %[[r113:.*]] = xegpu.update_nd_offset %[[arg6]], [%[[c64]], %[[c0]]] : !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r114:.*]] = xegpu.update_nd_offset %[[arg7]], [%[[c64]], %[[c0]]] : !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %11 = xetile.update_tile_offset %arg5, [%c64,  %c0] : !xetile.tile<32x32xtf32>, index, index -> !xetile.tile<32x32xtf32>

      //CHECK: scf.yield %[[r111]], %[[r112]], %[[r113]], %[[r114]], %[[r107]], %[[r110]]
      //CHECK-SAME: !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
      //CHECK-SAME: !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>,
      //CHECK-SAME: !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>,
      //CHECK-SAME: !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>, vector<32x16xf32>, vector<32x16xf32>
      scf.yield %10, %11, %9 : !xetile.tile<32x32xtf32>, !xetile.tile<32x32xtf32>, vector<32x32xf32>
    }

    //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r24]]#4 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r24]]#4 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r24]]#4 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r28:.*]] = vector.extract_strided_slice %[[r24]]#4 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r29:.*]] = vector.extract_strided_slice %[[r24]]#5 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r30:.*]] = vector.extract_strided_slice %[[r24]]#5 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r31:.*]] = vector.extract_strided_slice %[[r24]]#5 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: %[[r32:.*]] = vector.extract_strided_slice %[[r24]]#5 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    //CHECK: xegpu.store_nd %[[r25]], %[[r4]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r29]], %[[r6]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r26]], %[[r8]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r30]], %[[r9]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r27]], %[[r11]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r31]], %[[r12]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r28]], %[[r14]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: xegpu.store_nd %[[r32]], %[[r15]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
    xetile.store_tile %6#2,  %2 : vector<32x32xf32>, !xetile.tile<32x32xf32>
    //CHECK: gpu.return
    gpu.return
  }
}
