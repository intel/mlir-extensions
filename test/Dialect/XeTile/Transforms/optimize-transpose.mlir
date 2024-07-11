// RUN: imex-opt --split-input-file --xetile-optimize-transpose %s -verify-diagnostics -o -| FileCheck %s

gpu.module @mod0 {
  // CHECK-LABEL: gpu.func @gemm0
  // CHECK-SAME: (%[[ARG0:.*]]: memref<1024x512xf16>, %[[ARG1:.*]]: memref<1024x512xf16>, %[[ARG2:.*]]: memref<1024x1024xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C16:.*]] = arith.constant 16 : index
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[C512:.*]] = arith.constant 512 : index
  // CHECK: %[[BLOCKIDY:.*]] = gpu.block_id  y
  // CHECK: %[[OFFSETY:.*]] = arith.muli %[[BLOCKIDY]], %[[C32]] : index
  gpu.func @gemm0(%A: memref<1024x512xf16>, %B: memref<1024x512xf16>, %C: memref<1024x1024xf32>) {
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
    // B init tile is replaced with a new tile constructed from col-major view of the original B memref and order
    // CHECK: %[[RECAST:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [512, 1024], strides: [1, 512]
    // CHECK-SAME: memref<1024x512xf16> to memref<512x1024xf16, strided<[1, 512]>>
    // CHECK-NEXT:   %[[COLMAJORTILE:.*]] = xetile.init_tile %[[RECAST]][%[[C0]], %[[OFFSETY]]]
    // CHECK-SAME: memref<512x1024xf16, strided<[1, 512]>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1]>>
    %b_init_tile = xetile.init_tile %B[%n, %c0] : memref<1024x512xf16> -> !xetile.tile<32x16xf16>
    // scf.for is modified to use the new B tile
    // CHECK-NEXT: scf.for %[[ARG3:.*]] = %[[C0]] to %[[C512]] step %[[C16]] iter_args(%[[ATILE:.*]] = %[[T0:.*]], %[[BTILE:.*]] = %[[COLMAJORTILE]], %[[ACC:.*]] = %[[T2:.*]]) -> (!xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1]>>, vector<32x32xf32>) {
    %out:3 = scf.for %k = %c0 to %c512 step %c16
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<32x32xf32>) {
      %a_value = xetile.load_tile %a_tile  : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      // load modified B tile which gives the transposed output. Original transpose operation is removed.
      // CHECK: %[[BVALUE:.*]] = xetile.load_tile %[[BTILE]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<16x32xf16>
      %b_value = xetile.load_tile %b_tile  : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %b_trans = vector.transpose %b_value, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
      // CHECK-NEXT: %[[T3:.*]] = xetile.tile_mma %[[T4:.*]], %[[BVALUE]], %[[ACC]] : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %c_new_value = xetile.tile_mma %a_value, %b_trans, %c_value
        : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16>, index, index -> !xetile.tile<32x16xf16>
      // CHECK: %[[UPDATEDTILE:.*]] = xetile.update_tile_offset %[[BTILE]], [%[[C16]],  %[[C0]]]
      // CHECK-SAME: !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1]>>, index, index -> !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1]>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16>, index, index -> !xetile.tile<32x16xf16>
      // CHECK: scf.yield %[[T5:.*]], %[[UPDATEDTILE]], %[[T3]] : !xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1]>>, vector<32x32xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<32x32xf32>
    }
    xetile.store_tile %out#2, %c_init_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}

// -----

gpu.module @mod1 {
  // CHECK-LABEL: gpu.func @gemm1
  // CHECK-SAME: (%[[ARG0:.*]]: memref<1024x512xf16>, %[[ARG1:.*]]: memref<1024x512xf16, strided<[1, 1024]>>, %[[ARG2:.*]]: memref<1024x1024xf32>) {
  gpu.func @gemm1(%A: memref<1024x512xf16>, %B: memref<1024x512xf16, strided<[1, 1024]>>, %C: memref<1024x1024xf32>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C16:.*]] = arith.constant 16 : index
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
    // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [512, 1024], strides: [1024, 1] : memref<1024x512xf16, strided<[1, 1024]>> to memref<512x1024xf16, strided<[1024, 1]>>
    // CHECK-NEXT: %[[ROWMAJORTILE:.*]] = xetile.init_tile %[[CAST]][%[[C0]], %{{.*}}] : memref<512x1024xf16, strided<[1024, 1]>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<>>
    %b_init_tile = xetile.init_tile %B[%n, %c0] : memref<1024x512xf16, strided<[1, 1024]>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
    // CHECK-NEXT: %{{.*}}:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %[[BTILE:.*]] = %[[ROWMAJORTILE]], %{{.*}} = %{{.*}}) -> (!xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<>>, vector<32x32xf32>) {
    %out:3 = scf.for %k = %c0 to %c512 step %c16
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>) {
      %a_value = xetile.load_tile %a_tile  : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %b_value = xetile.load_tile %b_tile  : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>-> vector<32x16xf16>
      %b_trans = vector.transpose %b_value, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
      // CHECK: %{{.*}} = xetile.load_tile %[[BTILE]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<16x32xf16, #xetile.tile_attr<>> -> vector<16x32xf16>
      %c_new_value = xetile.tile_mma %a_value, %b_trans, %c_value
        : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16>, index, index -> !xetile.tile<32x16xf16>
      // CHECK: %[[UPDATEDTILE:.*]] = xetile.update_tile_offset %[[BTILE]], [%[[C16]],  %[[C0]]] : !xetile.tile<16x32xf16, #xetile.tile_attr<>>, index, index -> !xetile.tile<16x32xf16, #xetile.tile_attr<>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, index, index -> !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
      // CHECK: scf.yield %{{.*}}, %[[UPDATEDTILE]], %{{.*}} : !xetile.tile<32x16xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<>>, vector<32x32xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>
    }
    xetile.store_tile %out#2, %c_init_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}

// -----
gpu.module @mod2 {
  // CHECK-LABEL: gpu.func @gemm2
  // CHECK-SAME: (%[[ARG0:.*]]: memref<1024x512xf16>, %[[ARG1:.*]]: memref<1024x512xf16, #{{.*}}>, %[[ARG2:.*]]: memref<1024x1024xf32>) {
  gpu.func @gemm2(%A: memref<1024x512xf16>, %B: memref<1024x512xf16, affine_map<(d0, d1) -> (d1, d0)>>, %C: memref<1024x1024xf32>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C16:.*]] = arith.constant 16 : index
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
    // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [512, 1024], strides: [1024, 1] : memref<1024x512xf16, #map> to memref<512x1024xf16, strided<[1024, 1]>>
    // CHECK-NEXT: %{{.*}} = xetile.init_tile %[[CAST]][%[[C0]], %{{.*}}] : memref<512x1024xf16, strided<[1024, 1]>> -> !xetile.tile<16x32xf16, #xetile.tile_attr<>>
    %b_init_tile = xetile.init_tile %B[%n, %c0] : memref<1024x512xf16, affine_map<(d0, d1) -> (d1, d0)>> -> !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
    %out:3 = scf.for %k = %c0 to %c512 step %c16
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>) {
      %a_value = xetile.load_tile %a_tile  : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      %b_value = xetile.load_tile %b_tile  : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>-> vector<32x16xf16>
      %b_trans = vector.transpose %b_value, [1, 0] : vector<32x16xf16> to vector<16x32xf16>
      %c_new_value = xetile.tile_mma %a_value, %b_trans, %c_value
        : vector<32x16xf16>, vector<16x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16>, index, index -> !xetile.tile<32x16xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c0, %c16]
        : !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, index, index -> !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16, #xetile.tile_attr<order=[0, 1]>>, vector<32x32xf32>
    }
    xetile.store_tile %out#2, %c_init_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}
