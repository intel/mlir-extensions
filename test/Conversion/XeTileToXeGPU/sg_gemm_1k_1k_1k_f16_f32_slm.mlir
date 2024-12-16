// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -o -| FileCheck %s


#tile_attr = #xetile.tile_attr<memory_space = 3>

// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {

  //CHECK: gpu.func @test_gemm(%[[arg0:.*]]: memref<128x128xf16, 3>, %[[arg1:.*]]: memref<128x128xf16, 3>, %[[arg2:.*]]: memref<128x128xf32>)
  gpu.func @test_gemm(%arg0: memref<128x128xf16, 3>, %arg1: memref<128x128xf16, 3>, %arg2: memref<128x128xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
    //CHECK: %[[block_id_y:.*]] = gpu.block_id  y
    //CHECK: %[[r0:.*]] = arith.muli %[[block_id_x]], %[[c16]] : index
    //CHECK: %[[r1:.*]] = arith.muli %[[block_id_y]], %[[c16]] : index
    //CHECK: %[[r2:.*]] = arith.addi %[[r0]], %[[c0]] : index
    //CHECK: %[[r3:.*]] = arith.addi %[[r1]], %[[c0]] : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c16 : index
    %1 = arith.muli %block_id_y, %c16 : index

    //CHECK: %[[r4:.*]] = xegpu.create_nd_tdesc %[[arg2]][%[[r2]], %[[r3]]] : memref<128x128xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[r5:.*]] = xegpu.load_nd %[[r4]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16xf32>
    %2 = xetile.init_tile %arg2[%0, %1] : memref<128x128xf32> -> !xetile.tile<8x16xf32>
    %3 = xetile.load_tile %2 {padding = 0.000000e+00 : f32}  : !xetile.tile<8x16xf32> -> vector<8x16xf32>

    //CHECK: %[[r6:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r2]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    //CHECK: %[[r7:.*]] = arith.addi %[[r0]], %[[c1]] : index
    //CHECK: %[[r8:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r7]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c2:.*]] = arith.constant 2 : index
    //CHECK: %[[r9:.*]] = arith.addi %[[r0]], %[[c2]] : index
    //CHECK: %[[r10:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r9]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c3:.*]] = arith.constant 3 : index
    //CHECK: %[[r11:.*]] = arith.addi %[[r0]], %[[c3]] : index
    //CHECK: %[[r12:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r11]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c4:.*]] = arith.constant 4 : index
    //CHECK: %[[r13:.*]] = arith.addi %[[r0]], %[[c4]] : index
    //CHECK: %[[r14:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r13]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c5:.*]] = arith.constant 5 : index
    //CHECK: %[[r15:.*]] = arith.addi %[[r0]], %[[c5]] : index
    //CHECK: %[[r16:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r15]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c6:.*]] = arith.constant 6 : index
    //CHECK: %[[r17:.*]] = arith.addi %[[r0]], %[[c6]] : index
    //CHECK: %[[r18:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r17]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    //CHECK: %[[c7:.*]] = arith.constant 7 : index
    //CHECK: %[[r19:.*]] = arith.addi %[[r0]], %[[c7]] : index
    //CHECK: %[[r20:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[r19]], %[[c0]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    %4 = xetile.init_tile %arg0[%0, %c0] : memref<128x128xf16, 3> -> !xetile.tile<8x16xf16, #tile_attr>


    //CHECK: %[[r21:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[r3]]] : memref<128x128xf16, 3> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
    %5 = xetile.init_tile %arg1[%c0, %1] : memref<128x128xf16, 3> -> !xetile.tile<16x16xf16, #tile_attr>

    //CHECK: %[[r37:.*]]:10 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c16]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[r6]], %[[arg5:.*]] = %[[r8]], %[[arg6:.*]] = %[[r10]],
    //CHECK-SAME: %[[arg7:.*]] = %[[r12]], %[[arg8:.*]] = %[[r14]], %[[arg9:.*]] = %[[r16]],
    //CHECK-SAME: %[[arg10:.*]] = %[[r18]], %[[arg11:.*]] = %[[r20]], %[[arg12:.*]] = %[[r21]],
    //CHECK-SAME: %[[arg28:.*]] = %[[r5]])
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>,
    //CHECK-SAME: !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>, vector<8x16xf32>
    %6:3 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3)
          -> (!xetile.tile<8x16xf16, #tile_attr>, !xetile.tile<16x16xf16, #tile_attr>, vector<8x16xf32>) {
      //CHECK: %[[r38:.*]] = xegpu.load_nd %[[arg4]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r39:.*]] = xegpu.load_nd %[[arg5]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r40:.*]] = xegpu.load_nd %[[arg6]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r41:.*]] = xegpu.load_nd %[[arg7]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r42:.*]] = xegpu.load_nd %[[arg8]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r43:.*]] = xegpu.load_nd %[[arg9]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r44:.*]] = xegpu.load_nd %[[arg10]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r45:.*]] = xegpu.load_nd %[[arg11]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<1x16xf16>
      //CHECK: %[[r46:.*]] = vector.shuffle %[[r38]], %[[r39]] [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      //CHECK: %[[r47:.*]] = vector.shuffle %[[r40]], %[[r41]] [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      //CHECK: %[[r48:.*]] = vector.shuffle %[[r42]], %[[r43]] [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      //CHECK: %[[r49:.*]] = vector.shuffle %[[r44]], %[[r45]] [0, 1] : vector<1x16xf16>, vector<1x16xf16>
      //CHECK: %[[r50:.*]] = vector.shuffle %[[r46]], %[[r47]] [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
      //CHECK: %[[r51:.*]] = vector.shuffle %[[r48]], %[[r49]] [0, 1, 2, 3] : vector<2x16xf16>, vector<2x16xf16>
      //CHECK: %[[r52:.*]] = vector.shuffle %[[r50]], %[[r51]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf16>, vector<4x16xf16>
      %7 = xetile.load_tile %arg4 {padding = 0.000000e+00 : f32}  : !xetile.tile<8x16xf16, #tile_attr> -> vector<8x16xf16>

      //CHECK: %[[r53:.*]] = xegpu.load_nd %[[arg12]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>> -> vector<16x16xf16>
      %8 = xetile.load_tile %arg5 {padding = 0.000000e+00 : f32}  : !xetile.tile<16x16xf16, #tile_attr> -> vector<16x16xf16>


      //CHECK: %[[r84:.*]] = xegpu.dpas %[[r52]], %[[r53]], %[[arg28]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %9 = xetile.tile_mma %7, %8, %arg6 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>

      //CHECK: %[[r85:.*]] = xegpu.update_nd_offset %[[arg4]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r86:.*]] = xegpu.update_nd_offset %[[arg5]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r87:.*]] = xegpu.update_nd_offset %[[arg6]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r88:.*]] = xegpu.update_nd_offset %[[arg7]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r89:.*]] = xegpu.update_nd_offset %[[arg8]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r90:.*]] = xegpu.update_nd_offset %[[arg9]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r91:.*]] = xegpu.update_nd_offset %[[arg10]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      //CHECK: %[[r92:.*]] = xegpu.update_nd_offset %[[arg11]], [%[[c0]], %[[c16]]] : !xegpu.tensor_desc<1x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      %10 = xetile.update_tile_offset %arg4, [%c0,  %c16] : !xetile.tile<8x16xf16, #tile_attr>
      //CHECK: %[[r108:.*]] = xegpu.update_nd_offset %[[arg12]], [%[[c16]], %[[c0]]] : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space =  slm, array_length = 1 : i64, boundary_check = true>>
      %11 = xetile.update_tile_offset %arg5, [%c16,  %c0] : !xetile.tile<16x16xf16, #tile_attr>
      scf.yield %10, %11, %9 : !xetile.tile<8x16xf16, #tile_attr>, !xetile.tile<16x16xf16, #tile_attr>, vector<8x16xf32>
    }
    //CHECK: xegpu.store_nd %[[r37]]#9, %[[r4]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xetile.store_tile %6#2,  %2 : vector<8x16xf32>, !xetile.tile<8x16xf32>
    gpu.return
  }
}
