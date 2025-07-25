// RUN: imex-opt --split-input-file --xetile-canonicalization %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_module {
  gpu.func @test_static_memref(%arg0 : memref<512x128xf16, strided<[1, 512], offset:0>>, %arg1 : index, %arg2 : index) {
    %0 = xetile.init_tile %arg0 [%arg1, %arg2] : memref<512x128xf16, strided<[1, 512], offset:0>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    %3 = xetile.load_tile %0 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>> -> vector<16x32xf16>
    xetile.prefetch_tile %0 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    // With static offsets
    %1 = xetile.init_tile %arg0 [12, %arg1] : memref<512x128xf16, strided<[1, 512], offset:0>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    // Update offsets
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %2 = xetile.update_tile_offset %1, [%c32, %c16] : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    %4 = xetile.load_tile %2 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>> -> vector<16x32xf16>
    xetile.prefetch_tile %1 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    gpu.return
  }
}
// CHECK-LABEL: func @test_static_memref
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<512x128xf16, strided<[1, 512]>>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[RCAST:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [128, 512], strides: [512, 1] : memref<512x128xf16, strided<[1, 512]>> to memref<128x512xf16, strided<[512, 1]>>
// CHECK: %[[T0:.*]] = xetile.init_tile %[[RCAST]][%[[ARG2]], %[[ARG1]]] : memref<128x512xf16, strided<[512, 1]>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: %[[T1:.*]] = xetile.load_tile %[[T0]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>> -> vector<32x16xf16>
// CHECK: %[[T2:.*]] = xetile.transpose %[[T1]], [1, 0] : vector<32x16xf16> -> vector<16x32xf16>
// CHECK: xetile.prefetch_tile %[[T0]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: %[[RCAST0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [128, 512], strides: [512, 1] : memref<512x128xf16, strided<[1, 512]>> to memref<128x512xf16, strided<[512, 1]>>
// CHECK: %[[T3:.*]] = xetile.init_tile %[[RCAST0]][%[[ARG1]], 12] : memref<128x512xf16, strided<[512, 1]>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: %[[T4:.*]] = xetile.update_tile_offset %[[T3]], [%[[C16]],  %[[C32]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: %[[T5:.*]] = xetile.load_tile %[[T4]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>> -> vector<32x16xf16>
// CHECK: %[[T6:.*]] = xetile.transpose %[[T5]], [1, 0] : vector<32x16xf16> -> vector<16x32xf16>
// CHECK: xetile.prefetch_tile %[[T3]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>>

// -----
gpu.module @test_module {
  gpu.func @test_dynamic_memref(%arg0 : memref<?x?xf16>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) {
    %0 = xetile.init_tile %arg0 [%arg1, %arg2], [%arg3, %arg4], [%arg5, %arg6] : memref<?x?xf16> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    xetile.load_tile %0 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>> -> vector<16x32xf16>
    // Update offsets
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %2 = xetile.update_tile_offset %0, [%c32, %c16] : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    %3 = xetile.load_tile %2 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>> -> vector<16x32xf16>
    xetile.prefetch_tile %2 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    gpu.return
  }
}

// CHECK-LABEL: @test_dynamic_memref
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf16>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[ARG4:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[ARG5:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[ARG6:[a-zA-Z0-9]+]]: index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[T0:.*]] = xetile.init_tile %[[ARG0]][%[[ARG2]], %[[ARG1]]], [%[[ARG4]], %[[ARG3]]], [%[[ARG6]], %[[ARG5]]] : memref<?x?xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: %[[T1:.*]] = xetile.load_tile %[[T0]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>> -> vector<32x16xf16>
// CHECK: %[[T2:.*]] = xetile.transpose %[[T1]], [1, 0] : vector<32x16xf16> -> vector<16x32xf16>
// CHECK: %[[T3:.*]] = xetile.update_tile_offset %[[T0]], [%[[C16]],  %[[C32]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: %[[T4:.*]] = xetile.load_tile %[[T3]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>> -> vector<32x16xf16>
// CHECK: %[[T5:.*]] = xetile.transpose %[[T4]], [1, 0] : vector<32x16xf16> -> vector<16x32xf16>
// CHECK: xetile.prefetch_tile %[[T3]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>>

// -----
gpu.module @test_module {
  // Init tile and pass to a scf.for and update offsets.
  gpu.func @test_scf(%arg0 : memref<512x128xf16, strided<[1, 512], offset:0>>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %0 = xetile.init_tile %arg0 [%c0, %c0] : memref<512x128xf16, strided<[1, 512], offset:0>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %0) -> !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>> {
      %1 = xetile.load_tile %arg2 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>> -> vector<16x32xf16>
      %2 = xetile.update_tile_offset %arg2, [%c0, %c32] : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
      scf.yield %2 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    }
    gpu.return
  }
}

// CHECK-LABEL: @test_scf
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<512x128xf16, strided<[1, 512]>>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[RCAST:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [128, 512], strides: [512, 1] : memref<512x128xf16, strided<[1, 512]>> to memref<128x512xf16, strided<[512, 1]>>
// CHECK: %[[T0:.*]] = xetile.init_tile %[[RCAST]][%[[C0]], %[[C0]]] : memref<128x512xf16, strided<[512, 1]>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: scf.for %[[ARG1:[a-zA-Z0-9]+]] = %[[C0]] to %[[C128]] step %[[C32]] iter_args(%[[ARG3:[a-zA-Z0-9]+]] = %[[T0]]) -> (!xetile.tile<32x16xf16, #xetile.tile_attr<>>) {
// CHECK: %[[T2:.*]] = xetile.load_tile %[[ARG2]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>> -> vector<32x16xf16>
// CHECK: %[[T3:.*]] = xetile.transpose %[[T2]], [1, 0] : vector<32x16xf16> -> vector<16x32xf16>
// CHECK: %[[T4:.*]] = xetile.update_tile_offset %[[ARG2]], [%[[C32]],  %[[C0]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: scf.yield %[[T4]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>>

// -----
gpu.module @test_module {
  gpu.func @test_tranpose_after_load(%arg0 : memref<512x128xf16, strided<[1, 512], offset:0>>, %arg1 : index, %arg2 : index) -> vector<32x16xf16> {
    %0 = xetile.init_tile %arg0 [%arg1, %arg2] : memref<512x128xf16, strided<[1, 512], offset:0>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>>
    %3 = xetile.load_tile %0 : !xetile.tile<16x32xf16, #xetile.tile_attr<order=[0,1]>> -> vector<16x32xf16>
    // Transpose col-major load again.
    %1 = vector.transpose %3, [1, 0] : vector<16x32xf16> to vector<32x16xf16>
    gpu.return %1 : vector<32x16xf16>
  }
}

// CHECK-LABEL: @test_tranpose_after_load(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<512x128xf16, strided<[1, 512]>>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK: %[[RCAST:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [128, 512], strides: [512, 1] : memref<512x128xf16, strided<[1, 512]>> to memref<128x512xf16, strided<[512, 1]>>
// CHECK: %[[T0:.*]] = xetile.init_tile %[[RCAST]][%[[ARG2]], %[[ARG1]]] : memref<128x512xf16, strided<[512, 1]>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<>>
// CHECK: %[[T1:.*]] = xetile.load_tile %[[T0]] : !xetile.tile<32x16xf16, #xetile.tile_attr<>> -> vector<32x16xf16>
// CHECK: gpu.return %[[T1]] : vector<32x16xf16>

// -----
gpu.module @test_module {
  gpu.func @test_gemm_1(%A: memref<1024x512xf16>, %B: memref<1024x512xf16, strided<[1, 1024]>>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c32 : index
    %n = arith.muli %block_id_y, %c32 : index
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
    %c_init_value = xetile.load_tile %c_init_tile  : !xetile.tile<32x32xf32> -> vector<32x32xf32>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x512xf16> -> !xetile.tile<32x16xf16>
    %b_init_tile = xetile.init_tile %B[%n, %c0] : memref<1024x512xf16, strided<[1, 1024]>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
    %out:3 = scf.for %k = %c0 to %c512 step %c16
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>) {
      %a_value = xetile.load_tile %a_tile  : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %b_value = xetile.load_tile %b_tile  : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>-> vector<32x16xf16>
      %b_trans = vector.transpose %b_value, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
      %c_new_value = xetile.tile_mma %a_value, %b_trans, %c_value
        : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c16] : !xetile.tile<32x16xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c0, %c16] : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>
    }
    xetile.store_tile %out#2, %c_init_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}

// CHECK-LABEL: @test_gemm_1
// CHECK: %[[ARG0:[a-zA-Z0-9]+]]: memref<1024x512xf16>
// CHECK: %[[ARG1:[a-zA-Z0-9]+]]: memref<1024x512xf16, strided<[1, 1024]>>
// CHECK: %[[ARG2:[a-zA-Z0-9]+]]: memref<1024x1024xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
// CHECK: %[[BIDX:.*]] = gpu.block_id  x
// CHECK: %[[BIDY:.*]] = gpu.block_id  y
// CHECK: %[[T0:.*]] = arith.muli %[[BIDX]], %[[C32]] : index
// CHECK: %[[T1:.*]] = arith.muli %[[BIDY]], %[[C32]] : index
// CHECK: %[[T2:.*]] = xetile.init_tile %[[ARG2]][%[[T0]], %[[T1]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
// CHECK: %[[T3:.*]] = xetile.load_tile %[[T2]] : !xetile.tile<32x32xf32> -> vector<32x32xf32>
// CHECK: %[[T4:.*]] = xetile.init_tile %[[ARG0]][%[[T0]], %[[C0]]] : memref<1024x512xf16> -> !xetile.tile<32x16xf16>
// CHECK: %[[RCAST:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [512, 1024], strides: [1024, 1] : memref<1024x512xf16, strided<[1, 1024]>> to memref<512x1024xf16, strided<[1024, 1]>>
// CHECK: %[[T5:.*]] = xetile.init_tile %[[RCAST]][%[[C0]], %[[T1]]] : memref<512x1024xf16, strided<[1024, 1]>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<>>
// CHECK: scf.for %[[ARG3:.*]] = %[[C0]] to %[[C512]] step %[[C16]] iter_args(%[[ARG4:.*]] = %[[T4]], %[[ARG5]] = %[[T5]], %[[ARG6]] = %[[T3]]) -> (!xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<>>, vector<32x32xf32>) {
// CHECK: %[[T7:.*]] = xetile.load_tile %[[ARG4]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
// CHECK: %[[T8:.*]] = xetile.load_tile %[[ARG5]] : !xetile.tile<16x32xf16, #xetile.tile_attr<>> -> vector<16x32xf16>
// CHECK: %[[T9:.*]] = xetile.tile_mma %[[T7]], %[[T8]], %[[ARG6]] : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
// CHECK: %[[T10:.*]] = xetile.update_tile_offset %[[ARG4]], [%[[C0]],  %[[C16]]] : !xetile.tile<32x16xf16>
// CHECK: %[[T11:.*]] = xetile.update_tile_offset %[[ARG5]], [%[[C16]],  %[[C0]]] : !xetile.tile<16x32xf16, #xetile.tile_attr<>>
// CHECK: scf.yield %[[T10]], %[[T11]], %[[T9]] : !xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<>>, vector<32x32xf32>

// -----
gpu.module @test_module {
  gpu.func @test_gemm_2(%A: memref<1024x512xf16>, %B: memref<1024x512xf16, affine_map<(d0, d1) -> (d1 * 1024 + d0)>>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c32 : index
    %n = arith.muli %block_id_y, %c32 : index
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
    %c_init_value = xetile.load_tile %c_init_tile  : !xetile.tile<32x32xf32> -> vector<32x32xf32>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x512xf16> -> !xetile.tile<32x16xf16>
    %b_init_tile = xetile.init_tile %B[%n, %c0] : memref<1024x512xf16, affine_map<(d0, d1) -> (d1 * 1024 + d0)>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
    %out:3 = scf.for %k = %c0 to %c512 step %c16
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>) {
      %a_value = xetile.load_tile %a_tile  : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %b_value = xetile.load_tile %b_tile  : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>-> vector<32x16xf16>
      %b_trans = vector.transpose %b_value, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
      %c_new_value = xetile.tile_mma %a_value, %b_trans, %c_value
        : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>
    }
    xetile.store_tile %out#2, %c_init_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}
// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d1 * 1024 + d0)>
// CHECK-LABEL: @test_gemm_2
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<1024x512xf16>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<1024x512xf16, #[[MAP]]>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]+]]: memref<1024x1024xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C512:.*]] = arith.constant 512 : index
// CHECK: %[[BIDX:.*]] = gpu.block_id  x
// CHECK: %[[BIDY:.*]] = gpu.block_id  y
// CHECK: %[[T0:.*]] = arith.muli %[[BIDX]], %[[C32]] : index
// CHECK: %[[T1:.*]] = arith.muli %[[BIDY]], %[[C32]] : index
// CHECK: %[[T2:.*]] = xetile.init_tile %[[ARG2]][%[[T0]], %[[T1]]] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>
// CHECK: %[[T3:.*]] = xetile.load_tile %[[T2]] : !xetile.tile<32x32xf32> -> vector<32x32xf32>
// CHECK: %[[T4:.*]] = xetile.init_tile %[[ARG0]][%[[T0]], %[[C0]]] : memref<1024x512xf16> -> !xetile.tile<32x16xf16>
// CHECK: %[[RCAST:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [512, 1024], strides: [1024, 1] : memref<1024x512xf16, #map> to memref<512x1024xf16, strided<[1024, 1]>>
// CHECK: %[[T5:.*]] = xetile.init_tile %[[RCAST]][%[[C0]], %[[T1]]] : memref<512x1024xf16, strided<[1024, 1]>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<>>
// CHECK: scf.for %[[ARG3:.*]] = %[[C0]] to %[[C512]] step %[[C16]] iter_args(%[[ARG4:.*]] = %[[T4]], %[[ARG5:.*]] = %[[T5]], %[[ARG6:.*]] = %[[T3]]) -> (!xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<>>, vector<32x32xf32>) {
// CHECK: %[[T7:.*]] = xetile.load_tile %[[ARG4]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
// CHECK :%[[T8:.*]] = xetile.load_tile %[[ARG5]] : !xetile.tile<16x32xf16, #xetile.tile_attr<>> -> vector<16x32xf16>
// CHECK: %[[T9:.*]] = xetile.tile_mma %[[T7]], %[[T8]], %[[ARG6]] : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
// CHECK: %[[T10:.*]] = xetile.update_tile_offset %[[ARG4]], [%[[C0]],  %[[C16]]] : !xetile.tile<32x16xf16>
// CHECK: %[[T11:.*]] = xetile.update_tile_offset %[[ARG5]], [%[[C16]],  %[[C0]]] : !xetile.tile<16x32xf16, #xetile.tile_attr<>>
// CHECK :scf.yield %[[T10]], %[[T11]], %[[T9]] : !xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<>>, vector<32x32xf32>

// -----
gpu.module @test_module {
  gpu.func @test_broadcast_1(%arg0 : vector<8x1xf32>) -> vector<8x16xf32> {
    %0 = vector.broadcast %arg0 : vector<8x1xf32> to vector<8x16xf32>
    gpu.return %0 : vector<8x16xf32>
  }
}
// CHECK-LABEL: @test_broadcast_1
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<8x1xf32>) -> vector<8x16xf32>
// CHECK: %[[T0:.*]] = xetile.broadcast %[[ARG0]] [1] : vector<8x1xf32> -> vector<8x16xf32>
// CHECK: gpu.return %[[T0]] : vector<8x16xf32>

// -----
gpu.module @test_module {
  gpu.func @test_broadcast_2(%arg0 : vector<16xf32>) -> vector<8x16xf32> {
    %0 = vector.broadcast %arg0 : vector<16xf32> to vector<8x16xf32>
    gpu.return %0 : vector<8x16xf32>
  }
}

// CHECK-LABEL: @test_broadcast_2
// CHECK-SAMEL: %[[ARG0:[a-zA-Z0-9]+]]: vector<16xf32>) -> vector<8x16xf32>
// CHECK: %[[T0:.*]] = vector.shape_cast %[[ARG0]] : vector<16xf32> to vector<1x16xf32>
// CHECK: %[[T1:.*]] = xetile.broadcast %[[T0]] [0] : vector<1x16xf32> -> vector<8x16xf32>
// CHECK: gpu.return %[[T1]] : vector<8x16xf32>

// -----
gpu.module @test_module {
  gpu.func @test_broadcast_3(%arg0 : vector<1x16xf32>) -> vector<8x16xf32> {
    %0 = vector.broadcast %arg0 : vector<1x16xf32> to vector<8x16xf32>
    gpu.return %0 : vector<8x16xf32>
  }
}

// CHECK-LABEL: @test_broadcast_3
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<1x16xf32>) -> vector<8x16xf32>
// CHECK: %[[T0:.*]] = xetile.broadcast %[[ARG0]] [0] : vector<1x16xf32> -> vector<8x16xf32>
// CHECK: gpu.return %[[T0]] : vector<8x16xf32>

// -----
gpu.module @test_module {
  gpu.func @test_multireduction_1(%arg0 : vector<64x256xf32>, %arg1 : vector<256xf32>) -> vector<256xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %arg1 [0] : vector<64x256xf32> to vector<256xf32>
    gpu.return %0 : vector<256xf32>
  }
}

// CHECK-LABEL: @test_multireduction_1
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<64x256xf32>, %[[ARG1:[a-zA-Z0-9]+]]: vector<256xf32>) -> vector<256xf32>
// CHECK: %[[T0:.*]] = xetile.reduction <add>, %[[ARG0]] [0] : vector<64x256xf32> -> vector<1x256xf32>
// CHECK: %[[T1:.*]] = vector.shape_cast %[[T0]] : vector<1x256xf32> to vector<256xf32>
// CHECK: %[[T2:.*]] = arith.addf %[[T1]], %[[ARG1]] : vector<256xf32>
// CHECK: gpu.return %[[T2]] : vector<256xf32>

// -----
gpu.module @test_module {
  gpu.func @test_multireduction_2(%arg0 : vector<64x256xi8>, %arg1 : vector<256xi8>) -> vector<256xi8> {
    %0 = vector.multi_reduction <add>, %arg0, %arg1 [0] : vector<64x256xi8> to vector<256xi8>
    gpu.return %0 : vector<256xi8>
  }
}

// CHECK-LABEL: @test_multireduction_2
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<64x256xi8>, %[[ARG1:[a-zA-Z0-9]+]]: vector<256xi8>) -> vector<256xi8>
// CHECK: %[[T0:.*]] = xetile.reduction <add>, %[[ARG0]] [0] : vector<64x256xi8> -> vector<1x256xi8>
// CHECK: %[[T1:.*]] = vector.shape_cast %[[T0]] : vector<1x256xi8> to vector<256xi8>
// CHECK: %[[T2:.*]] = arith.addi %[[T1]], %[[ARG1]] : vector<256xi8>
// CHECK: gpu.return %[[T2]] : vector<256xi8>

// -----
gpu.module @test_module {
  gpu.func @test_transpose_1(%arg0 : vector<16x32xf32>) -> vector<32x16xf32> {
    %0 = vector.transpose %arg0, [1, 0] : vector<16x32xf32> to vector<32x16xf32>
    gpu.return %0 : vector<32x16xf32>
  }
}

// CHECK-LABEL: @test_transpose_1
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<16x32xf32>) -> vector<32x16xf32>
// CHECK: %[[T0:.*]] = xetile.transpose %arg0, [1, 0] : vector<16x32xf32> -> vector<32x16xf32>
// CHECK: gpu.return %[[T0]] : vector<32x16xf32>

// -----
// Intend to be unchanged for SLM
gpu.module @test_module {
  gpu.func @test_slm(%arg0: memref<512x128xf16, 3>) {
    %0 = xetile.init_tile %arg0[0, 0] : memref<512x128xf16, 3> -> !xetile.tile<16x32xf16, #xetile.tile_attr<memory_space = 3>>
    %1 = xetile.load_tile %0 : !xetile.tile<16x32xf16, #xetile.tile_attr<memory_space = 3 : i64>> -> vector<16x32xf16>
    %view = memref.transpose %arg0 (i, j) -> (j, i) : memref<512x128xf16, 3> to memref<128x512xf16, strided<[1, 128]>, 3>
    %2 = xetile.init_tile %view[16, 32] : memref<128x512xf16, strided<[1, 128]>, 3> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3>>
    xetile.store_tile %1, %2 : vector<16x32xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3>>
    gpu.return
  }
}

//CHECK-LABEL: gpu.func @test_slm
//CHECK-SAME: (%[[arg0:.*]]: memref<512x128xf16, 3>)
//CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<512x128xf16, 3> -> !xetile.tile<16x32xf16, #xetile.tile_attr<memory_space = 3 : i64>>
//CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<16x32xf16, #xetile.tile_attr<memory_space = 3 : i64>> -> vector<16x32xf16>
//CHECK: %[[transpose:.*]] = memref.transpose %[[arg0]] (d0, d1) -> (d1, d0) : memref<512x128xf16, 3> to memref<128x512xf16, strided<[1, 128]>, 3>
//CHECK: %[[r2:.*]] = xetile.init_tile %[[transpose]][16, 32] : memref<128x512xf16, strided<[1, 128]>, 3> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>
//CHECK: xetile.store_tile %[[r1]],  %[[r2]] : vector<16x32xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>

// -----
gpu.module @test_module {
  // CHECK-LABEL: gpu.func @test_reduction_size
  // CHECK-SAME: (%[[ARG0:.*]]: vector<64x256xf32>
  gpu.func @test_reduction_size(%arg0: vector<64x256xf32>, %arg1: memref<64x8xf32>) {
    %cst0 = arith.constant 0 : index
    // CHECK: %[[V0:.*]] = vector.shape_cast %[[ARG0]] : vector<64x256xf32> to vector<512x32xf32>
    // CHECK: %[[V1:.*]] = xetile.reduction <add>, %[[V0]] [1] : vector<512x32xf32> -> vector<512x1xf32>
    // CHECK: %[[V2:.*]] = vector.shape_cast %[[V1]] : vector<512x1xf32> to vector<64x8xf32>
    %reduced = xetile.reduction <add>, %arg0 [1] { reduction_size = 32 : i64 } : vector<64x256xf32> -> vector<64x8xf32>
    vector.store %reduced, %arg1[%cst0, %cst0] : memref<64x8xf32>, vector<64x8xf32>
    gpu.return
  }
}

// -----
gpu.module @test_module {
  // CHECK-LABEL: gpu.func @test_reduction_size
  // CHECK-SAME: (%[[ARG0:.*]]: vector<64x256xf32>
  gpu.func @test_reduction_size(%arg0: vector<64x256xf32>, %arg1: memref<2x256xf32>) {
    %cst0 = arith.constant 0 : index
    // CHECK: %[[V0:.*]] = vector.shape_cast %[[ARG0]] : vector<64x256xf32> to vector<32x512xf32>
    // CHECK: %[[V1:.*]] = xetile.reduction <add>, %[[V0]] [0] : vector<32x512xf32> -> vector<1x512xf32>
    // CHECK: %[[V2:.*]] = vector.shape_cast %[[V1]] : vector<1x512xf32> to vector<2x256xf32>
    %reduced = xetile.reduction <add>, %arg0 [0] { reduction_size = 32 : i64 } : vector<64x256xf32> -> vector<2x256xf32>
    vector.store %reduced, %arg1[%cst0, %cst0] : memref<2x256xf32>, vector<2x256xf32>
    gpu.return
  }
}
