// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  //CHECK: gpu.func @sglevel_tiled_gemm(%[[arg0:.*]]: memref<1024x1024xf16>, %[[arg1:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_tiled_gemm(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>) {
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32x32xf16>
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    %cst = arith.constant dense<0.0> : vector<32x32xf16>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index

    //CHECK: %[[r2:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>
  	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[r3:.*]]:2 = scf.for %[[arg2:.*]] = %[[c0]] to %[[c1024]] step %[[c64]] iter_args(%[[arg3:.*]] = %[[r2]], %[[arg4:.*]] = %[[cst]]) -> (!xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>, vector<32x32xf16>) {
    %nexta, %res = scf.for %k= %c0 to %c1024 step %c64 iter_args(%subA = %1, %subB = %cst) -> (!xetile.tile<32x32xf16>, vector<32x32xf16>) {

      //CHECK: %[[r12:.*]] = xegpu.load_nd %[[arg3]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>> -> vector<32x32xf16>
      %3 = xetile.load_tile %subA : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[r13:.*]] = xegpu.update_nd_offset %[[arg3]], [%[[c0]], %[[c64]]] : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>
      %5 = xetile.update_tile_offset %subA, [%c0, %c64]: !xetile.tile<32x32xf16>

      //CHECK: scf.yield %[[r13]], %[[r12]] : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<>>, vector<32x32xf16>
      scf.yield %5, %3: !xetile.tile<32x32xf16>, vector<32x32xf16>
    }

    //CHECK: %[[r8:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
    //CHECK: %[[r9:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c8]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
    //CHECK: %[[r10:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c16]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
    //CHECK: %[[r11:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c24]], %[[c64]]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
  	%5 = xetile.init_tile %b[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[r4:.*]] = vector.extract_strided_slice %[[r3]]#1 {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r3]]#1 {offsets = [8, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r3]]#1 {offsets = [16, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r3]]#1 {offsets = [24, 0], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>

    //CHECK: xegpu.store_nd %[[r4]], %[[r8]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
    //CHECK: xegpu.store_nd %[[r5]], %[[r9]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
    //CHECK: xegpu.store_nd %[[r6]], %[[r10]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
    //CHECK: xegpu.store_nd %[[r7]], %[[r11]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16, #xegpu.block_tdesc_attr<>>
    xetile.store_tile %res, %5: vector<32x32xf16>, !xetile.tile<32x32xf16>

    //CHECK: gpu.return
  	gpu.return
  }
}
