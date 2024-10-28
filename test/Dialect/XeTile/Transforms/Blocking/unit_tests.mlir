// RUN: imex-opt --split-input-file --xetile-blocking --canonicalize %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_load_tile(%[[arg0:.*]]: memref<32x32xf16>)
  //CHECK: %[[c0:.*]] = arith.constant 0 : index
  //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
  //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
  gpu.func @sg_load_tile(%a: memref<32x32xf16>) {
    %c0 = arith.constant 0 : index
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  	gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_unaligned(%[[arg0:.*]]: memref<128x128xf16>)
  //CHECK: %[[c0:.*]] = arith.constant 0 : index
  //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<85x76xf16, #xetile.tile_attr<inner_blocks = [17, 19]>>
  //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<85x76xf16, #xetile.tile_attr<inner_blocks = [17, 19]>> -> vector<5x4x17x19xf16>
  //CHECK: gpu.return
  gpu.func @sg_load_tile_unaligned(%a: memref<128x128xf16>) {
    %c0 = arith.constant 0 : index
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<85x76xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<85x76xf16> -> vector<85x76xf16>
   gpu.return
  }

  //CHECK: gpu.func @sg_store_tile(%[[arg0:.*]]: memref<32x32xf32>)
  //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4x2x8x16xf32>
  //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  //CHECK: xetile.store_tile %[[cst]],  %[[R0]] : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
	gpu.func @sg_store_tile(%a: memref<32x32xf32>) {
		%result = arith.constant dense<0.0>: vector<32x32xf32>
		%1 = xetile.init_tile %a[0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
		xetile.store_tile %result, %1: vector<32x32xf32>, !xetile.tile<32x32xf32>
		gpu.return
	}

  //CHECK: gpu.func @create_mask
  //CHECK: %[[MASK:.*]] = vector.constant_mask [32, 1, 1, 20] : vector<32x1x1x32xi1>
  gpu.func @create_mask(%a: vector<32x32xf16>, %b: vector<32x32xf16>, %c: memref<32x32xf16>) {
    %c32 = arith.constant 32 : index
    %c20 = arith.constant 20 : index
    %mask = vector.create_mask %c32, %c20 : vector<32x32xi1>
    %select = arith.select %mask, %a, %b : vector<32x32xi1>, vector<32x32xf16>
    %tile = xetile.init_tile %c[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    xetile.store_tile %select, %tile: vector<32x32xf16>, !xetile.tile<32x32xf16>
    gpu.return
  }

  //CHECK: gpu.func @create_mask_2
  //CHECK: %[[MASK:.*]] = vector.constant_mask [20, 1, 1, 32] : vector<32x1x1x32xi1>
  gpu.func @create_mask_2(%a: vector<32x32xf16>, %b: vector<32x32xf16>, %c: memref<32x32xf16>) {
    %c20 = arith.constant 20 : index
    %c32 = arith.constant 32 : index
    %mask = vector.create_mask %c20, %c32 : vector<32x32xi1>
    %select = arith.select %mask, %a, %b : vector<32x32xi1>, vector<32x32xf16>
    %tile = xetile.init_tile %c[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    xetile.store_tile %select, %tile: vector<32x32xf16>, !xetile.tile<32x32xf16>
    gpu.return
  }

  //CHECK: gpu.func @sg_store_tile_unaligned(%[[arg0:.*]]: memref<128x128xf32>)
  //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<11x19x4x2xf32>
  //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<128x128xf32> -> !xetile.tile<44x38xf32, #xetile.tile_attr<inner_blocks = [4, 2]>>
  //CHECK: xetile.store_tile %[[cst]],  %[[R0]] : vector<11x19x4x2xf32>, !xetile.tile<44x38xf32, #xetile.tile_attr<inner_blocks = [4, 2]>>
  //CHECK: gpu.return
	gpu.func @sg_store_tile_unaligned(%a: memref<128x128xf32>) {
	  %result = arith.constant dense<0.0>: vector<44x38xf32>
	  %1 = xetile.init_tile %a[0, 0] : memref<128x128xf32> -> !xetile.tile<44x38xf32>
	  xetile.store_tile %result, %1: vector<44x38xf32>, !xetile.tile<44x38xf32>
	  gpu.return
	}

  //CHECK: gpu.func @sg_tile_mma(%[[arg0:.*]]: memref<32x32xf16>, %[[arg1:.*]]: memref<32x32xf16>)
  gpu.func @sg_tile_mma(%a: memref<32x32xf16>, %b: memref<32x32xf16>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>

    //CHECK: %[[R3:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
  	%3 = xetile.init_tile %b[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R4:.*]] = xetile.load_tile %[[R3]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
    //CHECK: %[[R5:.*]] = xetile.tile_unpack %[[R4]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
    %4 = xetile.load_tile %3 : !xetile.tile<32x32xf16> -> vector<32x32xf16>

    //CHECK: %[[R6:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>
    //CHECK: %[[R7:.*]] = xetile.tile_pack %[[R5]] {inner_blocks = array<i64: 16, 16>}  : vector<32x32xf16> -> vector<2x2x16x16xf16>
    //CHECK: %[[R8:.*]] = xetile.tile_mma %[[R6]], %[[R7]] : vector<4x2x8x16xf16>, vector<2x2x16x16xf16> -> vector<4x2x8x16xf32>
    %5 = xetile.tile_mma %2, %4: vector<32x32xf16>, vector<32x32xf16> -> vector<32x32xf32>
  	gpu.return
  }

  ///-CHECK: gpu.func @tile_mma_irregular(%[[arg0:.*]]: memref<128x128xf16>, %[[arg1:.*]]: memref<128x128xf16>)
  ///-CHECK: %[[c0:.*]] = arith.constant 0 : index
  ///-CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<90x76xf16, #xetile.tile_attr<inner_blocks = [30, 19]>>
  ///-CHECK: %[[R1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1], inner_blocks = [19, 6]>>
  ///-  gpu.func @tile_mma_irregular(%a: memref<128x128xf16>, %b: memref<128x128xf16>) {
  ///-    %c0 = arith.constant 0 : index
  ///-    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<90x76xf16>
  ///-    %2 = xetile.init_tile %b[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1]>>
  ///-    %3 = xetile.load_tile %1 : !xetile.tile<90x76xf16> -> vector<90x76xf16>
  ///-    %4 = xetile.load_tile %2 : !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<76x90xf16>
  ///-    %5 = xetile.tile_mma %3, %4: vector<90x76xf16>, vector<76x90xf16> -> vector<90x90xf32>
  ///-    gpu.return
  ///-  }


  //CHECK: gpu.func @sg_gemm(%[[arg0:.*]]: memref<32x128xf16>, %[[arg1:.*]]: memref<128x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_gemm(%a: memref<32x128xf16>, %b: memref<128x32xf16>, %c: memref<32x32xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    %c32 = arith.constant 32 : index

    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    %c128 = arith.constant 128 : index

    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4x2x8x16xf32>
    %cst = arith.constant dense<0.0>: vector<32x32xf32>

    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R2:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]] iter_args(%[[arg4:.*]] = %[[R0]], %[[arg5:.*]] = %[[R1]], %[[arg6:.*]] = %[[cst]]) -> (!xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x2x8x16xf32>) {
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      //CHECK: %[[R4:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[R5:.*]] = xetile.tile_unpack %[[R4]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      //CHECK: %[[R6:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[R7:.*]] = xetile.tile_unpack %[[R6]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[R8:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c32]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      //CHECK: %[[R9:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c32]],  %[[c0]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]:  !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]:  !xetile.tile<32x32xf16>
      //CHECK: %[[R10:.*]] = xetile.tile_pack %[[R5]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>
      //CHECK: %[[R11:.*]] = xetile.tile_pack %[[R7]] {inner_blocks = array<i64: 16, 16>}  : vector<32x32xf16> -> vector<2x2x16x16xf16>
      //CHECK: %[[R12:.*]] = xetile.tile_mma %[[R10]], %[[R11]], %[[arg6]] : vector<4x2x8x16xf16>, vector<2x2x16x16xf16>, vector<4x2x8x16xf32> -> vector<4x2x8x16xf32>
      %c_new_value = xetile.tile_mma %3, %4, %c_value:
        vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      //CHECK: scf.yield %[[R8]], %[[R9]], %[[R12]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x2x8x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }

    //CHECK: %[[R3]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
    //CHECK: xetile.store_tile %[[R2]]#2,  %[[R3]] : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
  	gpu.return
  }

  // CHECK-LABEL: gpu.func @inner_reduction
  // CHECK-SAME: (%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>)
  gpu.func @inner_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>>
    %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>> -> vector<1x1x16x32xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 16, 32>}  : vector<1x1x16x32xf16> -> vector<16x32xf16>
    %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
    //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 1, 32>}  : vector<16x32xf16> -> vector<16x1x1x32xf16>
    //CHECK: %[[R4:.*]] = math.exp %[[R3]] : vector<16x1x1x32xf16>
    %e = math.exp %v: vector<16x32xf16>

    //CHECK: %[[R5:.*]] = xetile.reduction <add>, %[[R4]] [1, 3] : vector<16x1x1x32xf16> -> vector<16x1x1x1xf16>
    //CHECK: %[[R6:.*]] = xetile.tile_unpack %[[R5]] {inner_blocks = array<i64: 1, 1>} : vector<16x1x1x1xf16> -> vector<16x1xf16>
    %r = xetile.reduction <add>, %e [1] : vector<16x32xf16> -> vector<16x1xf16>
    //CHECK: %[[R7:.*]] = vector.shape_cast %[[R6]] : vector<16x1xf16> to vector<2x8xf16>
    %c = vector.shape_cast %r: vector<16x1xf16> to vector<2x8xf16>
    //CHECK: %[[R8:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<2x8xf16, #xetile.tile_attr<inner_blocks = [2, 8]>>
    %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<2x8xf16>
    //CHECK: %[[R9:.*]] = xetile.tile_pack %[[R7]] {inner_blocks = array<i64: 2, 8>}  : vector<2x8xf16> -> vector<1x1x2x8xf16>
    //CHECK: xetile.store_tile %[[R9]],  %[[R8]] : vector<1x1x2x8xf16>, !xetile.tile<2x8xf16, #xetile.tile_attr<inner_blocks = [2, 8]>>
    xetile.store_tile %c, %s : vector<2x8xf16>, !xetile.tile<2x8xf16>
    gpu.return
  }

  //CHECK: gpu.func @outter_reduction(%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>) {
  gpu.func @outter_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index

    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>>
    %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>

    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>> -> vector<1x1x16x32xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 16, 32>}  : vector<1x1x16x32xf16> -> vector<16x32xf16>
    %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>

    //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 1, 32>}  : vector<16x32xf16> -> vector<16x1x1x32xf16>
    //CHECK: %[[R4:.*]] = math.exp %[[R3]] : vector<16x1x1x32xf16>
    %e = math.exp %v: vector<16x32xf16>

    //CHECK: %[[R5:.*]] = xetile.reduction <add>, %4 [0, 2] : vector<16x1x1x32xf16> -> vector<1x1x1x32xf16>
    //CHECK: %[[R6:.*]] = xetile.tile_unpack %5 {inner_blocks = array<i64: 1, 32>}  : vector<1x1x1x32xf16> -> vector<1x32xf16>
    %r = xetile.reduction <add>, %e [0] : vector<16x32xf16> -> vector<1x32xf16>

    //CHECK: %[[R7:.*]] = vector.shape_cast %6 : vector<1x32xf16> to vector<4x8xf16>
    %c = vector.shape_cast %r: vector<1x32xf16> to vector<4x8xf16>

    //CHECK: %[[R8:.*]] = xetile.init_tile %arg1[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<4x8xf16, #xetile.tile_attr<inner_blocks = [4, 8]>>
    %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<4x8xf16>

    //CHECK: %[[R9:.*]] = xetile.tile_pack %7 {inner_blocks = array<i64: 4, 8>}  : vector<4x8xf16> -> vector<1x1x4x8xf16>
    //CHECK: xetile.store_tile %9,  %8 : vector<1x1x4x8xf16>, !xetile.tile<4x8xf16, #xetile.tile_attr<inner_blocks = [4, 8]>>
    xetile.store_tile %c, %s : vector<4x8xf16>, !xetile.tile<4x8xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_gemm_with_preops_for_c
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x128xf16>, %[[arg1:.*]]: memref<128x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_gemm_with_preops_for_c(%a: memref<32x128xf16>, %b: memref<128x32xf16>, %c: memref<32x32xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4x2x8x16xf32>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant dense<0.0>: vector<32x32xf32>

    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]] iter_args(%[[arg4:.*]] = %[[r0]], %[[arg5:.*]] = %[[r1]], %[[arg6:.*]] = %[[cst]]) -> (!xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x2x8x16xf32>) {
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      //CHECK: %[[r7:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[r8:.*]] = xetile.tile_unpack %[[r7]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[r9:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[r10:.*]] = xetile.tile_unpack %[[r9]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[r11:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c32]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      //CHECK: %[[r12:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c32]],  %[[c0]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]:  !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]:  !xetile.tile<32x32xf16>

      //CHECK: %[[r15:.*]] = arith.addf %[[arg6]], %[[arg6]] : vector<4x2x8x16xf32>
      %5 = arith.addf %c_value, %c_value: vector<32x32xf32>

      //CHECK: %[[r17:.*]] = xetile.tile_pack %[[r8]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>
      //CHECK: %[[r18:.*]] = xetile.tile_pack %[[r10]] {inner_blocks = array<i64: 16, 16>}  : vector<32x32xf16> -> vector<2x2x16x16xf16>
      //CHECK: %[[r20:.*]] = xetile.tile_mma %[[r17]], %[[r18]], %[[r15]] : vector<4x2x8x16xf16>, vector<2x2x16x16xf16>, vector<4x2x8x16xf32> -> vector<4x2x8x16xf32>
      %c_new_value = xetile.tile_mma %3, %4, %5: vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      //CHECK: scf.yield %[[r11]], %[[r12]], %[[r20]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x2x8x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }

    //CHECK: %[[r4:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>

    //CHECK: xetile.store_tile %[[r2]]#2,  %[[r4]] : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
  	gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_reduction_broadcast_dim_0
  //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_reduction_broadcast_dim_0(%a: memref<1024x1024xf16>) {
    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 1, 32>}  : vector<32x64xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[R4:.*]] = xetile.reduction <add>, %[[R3]] [0, 2] : vector<32x2x1x32xf16> -> vector<1x2x1x32xf16>
    %3 = xetile.reduction <add>, %2 [0]: vector<32x64xf16> -> vector<1x64xf16>
    //CHECK: %[[R5:.*]] = xetile.broadcast %[[R4]] [0, 2] : vector<1x2x1x32xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[R6:.*]] = xetile.tile_unpack %[[R5]] {inner_blocks = array<i64: 1, 32>}  : vector<32x2x1x32xf16> -> vector<32x64xf16>
    %4 = xetile.broadcast %3 [0]: vector<1x64xf16> -> vector<32x64xf16>
    //CHECK: %[[R7:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %5 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: %[[R8:.*]] = xetile.tile_pack %[[R6]] {inner_blocks = array<i64: 8, 32>}  : vector<32x64xf16> -> vector<4x2x8x32xf16>
    //CHECK: xetile.store_tile %[[R8]],  %[[R7]] : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    xetile.store_tile %4, %5: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }


  //CHECK-LABEL: gpu.func @sglevel_reduction_broadcast_dim_1
  //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_reduction_broadcast_dim_1(%a: memref<1024x1024xf16>) {
    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 1, 32>}  : vector<32x64xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[R4:.*]] = xetile.reduction <add>, %[[R3]] [1, 3] : vector<32x2x1x32xf16> -> vector<32x1x1x1xf16>
    %3 = xetile.reduction <add>, %2 [1]: vector<32x64xf16> -> vector<32x1xf16>
    //CHECK: %[[R5:.*]] = xetile.broadcast %[[R4]] [1, 3] : vector<32x1x1x1xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[R6:.*]] = xetile.tile_unpack %[[R5]] {inner_blocks = array<i64: 1, 32>}  : vector<32x2x1x32xf16> -> vector<32x64xf16>
    %4 = xetile.broadcast %3 [1]: vector<32x1xf16> -> vector<32x64xf16>
    //CHECK: %[[R7:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %5 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: %[[R8:.*]] = xetile.tile_pack %[[R6]] {inner_blocks = array<i64: 8, 32>}  : vector<32x64xf16> -> vector<4x2x8x32xf16>
    //CHECK: xetile.store_tile %[[R8]],  %[[R7]] : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    xetile.store_tile %4, %5: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }


  //CHECK-LABEL: gpu.func @sglevel_reduction_broadcast_transpose
  //CHECK-SAME(%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_reduction_broadcast_transpose(%a: memref<1024x1024xf16>) {
    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 1, 32>}  : vector<32x64xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[R4:.*]] = xetile.reduction <add>, %[[R3]] [1, 3] : vector<32x2x1x32xf16> -> vector<32x1x1x1xf16>
    %3 = xetile.reduction <add>, %2 [1]: vector<32x64xf16> -> vector<32x1xf16>
    //CHECK: %[[R5:.*]] = xetile.broadcast %[[R4]] [1, 3] : vector<32x1x1x1xf16> -> vector<32x8x1x8xf16>
    //CHECK: %[[R6:.*]] = xetile.tile_unpack %[[R5]] {inner_blocks = array<i64: 1, 8>} : vector<32x8x1x8xf16> -> vector<32x64xf16>
    %4 = xetile.broadcast %3 [1]: vector<32x1xf16> -> vector<32x64xf16>
    //CHECK: %[[R7:.*]] = xetile.tile_pack %[[R6]] {inner_blocks = array<i64: 32, 8>} : vector<32x64xf16> -> vector<1x8x32x8xf16>
    //CHECK: %[[R8:.*]] = xetile.transpose %[[R7]], [1, 0, 3, 2] : vector<1x8x32x8xf16> -> vector<8x1x8x32xf16>
    %5 = xetile.transpose %4, [1, 0]: vector<32x64xf16> -> vector<64x32xf16>
    //CHECK: %[[R9:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %6 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16>
    //CHECK: xetile.store_tile %[[R8]],  %[[R9]] : vector<8x1x8x32xf16>, !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    xetile.store_tile %5, %6: vector<64x32xf16>, !xetile.tile<64x32xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_softmax_dim_0
  //CHECK-SAME(%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_softmax_dim_0(%a: memref<1024x1024xf16>) {

    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 8, 32>}  : vector<32x64xf16> -> vector<4x2x8x32xf16>
    //CHECK: %[[R4:.*]] = math.exp %[[R3]] : vector<4x2x8x32xf16>
    //CHECK: %[[R5:.*]] = xetile.tile_unpack %[[R4]] {inner_blocks = array<i64: 8, 32>}  : vector<4x2x8x32xf16> -> vector<32x64xf16>
    %3 = math.exp %2: vector<32x64xf16>
    //CHECK: %[[R6:.*]] = xetile.tile_pack %[[R5]] {inner_blocks = array<i64: 1, 32>}  : vector<32x64xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[R7:.*]] = xetile.reduction <add>, %[[R6]] [0, 2] : vector<32x2x1x32xf16> -> vector<1x2x1x32xf16>
    %4 = xetile.reduction <add>, %3 [0]: vector<32x64xf16> -> vector<1x64xf16>
    //CHECK: %[[R8:.*]] = xetile.broadcast %[[R7]] [0, 2] : vector<1x2x1x32xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[R9:.*]] = xetile.tile_unpack %[[R8]] {inner_blocks = array<i64: 1, 32>}  : vector<32x2x1x32xf16> -> vector<32x64xf16>
    %5 = xetile.broadcast %4 [0]: vector<1x64xf16> -> vector<32x64xf16>
    //CHECK: %[[R10:.*]] = xetile.tile_pack %[[R9]] {inner_blocks = array<i64: 8, 32>}  : vector<32x64xf16> -> vector<4x2x8x32xf16>
    //CHECK: %[[R11:.*]] = arith.divf %[[R4]], %[[R10]] : vector<4x2x8x32xf16>
    %6 = arith.divf %3, %5: vector<32x64xf16>
    //CHECK: %[[R12:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: xetile.store_tile %[[R11]],  %[[R12]] : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }

  //CHECK: (%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_softmax_dim_1(%a: memref<1024x1024xf16>) {
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.tile_unpack %[[r1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>

    //CHECK: %[[r3:.*]] = xetile.tile_pack %[[r2]] {inner_blocks = array<i64: 8, 32>} : vector<32x64xf16> -> vector<4x2x8x32xf16>
    //CHECK: %[[r4:.*]] = math.exp %[[r3]] : vector<4x2x8x32xf16>
    %3 = math.exp %2: vector<32x64xf16>

    //CHECK: %[[r5:.*]] = xetile.tile_unpack %[[r4]] {inner_blocks = array<i64: 8, 32>} : vector<4x2x8x32xf16> -> vector<32x64xf16>
    //CHECK: %[[r6:.*]] = xetile.tile_pack %[[r5]] {inner_blocks = array<i64: 1, 32>} : vector<32x64xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[r7:.*]] = xetile.reduction <add>, %[[r6]] [1, 3] : vector<32x2x1x32xf16> -> vector<32x1x1x1xf16>
    %4 = xetile.reduction <add>, %3 [1]: vector<32x64xf16> -> vector<32x1xf16>
    //CHECK: %[[r8:.*]] = xetile.broadcast %[[r7]] [1, 3] : vector<32x1x1x1xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[r9:.*]] = xetile.tile_unpack %[[r8]] {inner_blocks = array<i64: 1, 32>} : vector<32x2x1x32xf16> -> vector<32x64xf16>
    %5 = xetile.broadcast %4 [1]: vector<32x1xf16> -> vector<32x64xf16>

    //CHECK: %[[r10:.*]] = xetile.tile_pack %[[r9]] {inner_blocks = array<i64: 8, 32>} : vector<32x64xf16> -> vector<4x2x8x32xf16>
    //CHECK: %[[r11:.*]] = arith.divf %[[r4]], %[[r10]] : vector<4x2x8x32xf16>
    %6 = arith.divf %3, %5: vector<32x64xf16>
    //CHECK: %[[r12:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK: xetile.store_tile %[[r11]],  %[[r12]] : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_softmax_transpose
  //CHECK-SAME(%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_softmax_transpose(%a: memref<1024x1024xf16>) {
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 8]>>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 8]>> -> vector<1x8x32x8xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>

    //CHECK: %[[r2:.*]] = math.exp %[[r1]] : vector<1x8x32x8xf16>
    //CHECK: %[[r3:.*]] = xetile.tile_unpack %[[r2]] {inner_blocks = array<i64: 32, 8>} : vector<1x8x32x8xf16> -> vector<32x64xf16>
    %3 = math.exp %2: vector<32x64xf16>

    //CHECK: %[[r4:.*]] = xetile.tile_pack %[[r3]] {inner_blocks = array<i64: 1, 32>} : vector<32x64xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[r5:.*]] = xetile.reduction <add>, %[[r4]] [1, 3] : vector<32x2x1x32xf16> -> vector<32x1x1x1xf16>
    %4 = xetile.reduction <add>, %3 [1]: vector<32x64xf16> -> vector<32x1xf16>

    //CHECK: %[[r6:.*]] = xetile.broadcast %[[r5]] [1, 3] : vector<32x1x1x1xf16> -> vector<32x8x1x8xf16>
    //CHECK: %[[r7:.*]] = xetile.tile_unpack %[[r6]] {inner_blocks = array<i64: 1, 8>} : vector<32x8x1x8xf16> -> vector<32x64xf16>
    %5 = xetile.broadcast %4 [1]: vector<32x1xf16> -> vector<32x64xf16>

    //CHECK: %[[r8:.*]] = xetile.tile_pack %[[r7]] {inner_blocks = array<i64: 32, 8>} : vector<32x64xf16> -> vector<1x8x32x8xf16>
    //CHECK: %[[r9:.*]] = arith.divf %[[r2]], %[[r8]] : vector<1x8x32x8xf16>
    %6 = arith.divf %3, %5: vector<32x64xf16>

    //CHECK: %[[r10:.*]] = xetile.transpose %[[r9]], [1, 0, 3, 2] : vector<1x8x32x8xf16> -> vector<8x1x8x32xf16>
    %7 = xetile.transpose %6, [1, 0]: vector<32x64xf16> -> vector<64x32xf16>

    //CHECK: %[[r11:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    %8 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16>

    //CHECK: xetile.store_tile %[[r10]],  %[[r11]] : vector<8x1x8x32xf16>, !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    xetile.store_tile %7, %8: vector<64x32xf16>, !xetile.tile<64x32xf16>
    gpu.return
  }

  gpu.func @sglevel_unregular_gemm(%arg0: memref<16384x12288xf16>, %arg1: memref<1536x12288xf16>, %arg2: memref<16384x1536xf32>) attributes {gemm_tiles_b = 1 : i64, gemm_tiles_x = dense<[8, 2, 4, 8]> : vector<4xi64>, gemm_tiles_y = dense<[1, 1, 8, 4]> : vector<4xi64>, physical_nd_range = dense<[8, 32]> : vector<2xi64>, region_partition = 0 : i64, region_size = 32 : i64, syn.fusion_successful, syn.tensor_signature = (tensor<16384x12288xf16>, tensor<1536x12288xf16>) -> tensor<16384x1536xf32>, synFusionGenOps = 6 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1007562515.4 : f64} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant dense<0.000000e+00> : vector<32x64xf32>
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c256 = arith.constant 256 : index
    %c2048 = arith.constant 2048 : index
    %c2 = arith.constant 2 : index
    %c12288 = arith.constant 12288 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.divsi %block_id_y, %c8 : index
    %1 = arith.remsi %block_id_y, %c8 : index
    %2 = arith.muli %1, %c256 : index
    %3 = arith.muli %block_id_x, %c2048 : index
    %4 = arith.muli %0, %c256 : index
    %5 = arith.addi %3, %4 : index
    %6 = gpu.subgroup_id : index
    %7 = index.floordivs %6, %c4
    %8 = index.remu %6, %c4
    %9 = index.remu %7, %c8
    %10 = index.mul %9, %c32
    %11 = index.add %5, %10
    %12 = index.remu %8, %c4
    %13 = index.mul %12, %c64
    %14 = index.add %2, %13
    %15 = xetile.init_tile %arg2[%11, %14] : memref<16384x1536xf32> -> !xetile.tile<32x64xf32>
    %16 = index.remu %8, %c1
    %17 = index.mul %16, %c32
    %18 = xetile.init_tile %arg0[%11, %17] : memref<16384x12288xf16> -> !xetile.tile<32x32xf16>
    %19 = index.floordivs %6, %c8
    %20 = index.remu %6, %c8
    %21 = index.remu %19, %c4
    %22 = index.mul %21, %c64
    %23 = index.add %2, %22
    %24 = index.remu %20, %c1
    %25 = index.mul %24, %c32

    // CHECK: xetile.init_tile %{{.*}} : memref<1536x12288xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    %26 = xetile.init_tile %arg1[%23, %25] : memref<1536x12288xf16> -> !xetile.tile<64x32xf16>
    %27:2 = scf.for %arg15 = %c0 to %c2 step %c1 iter_args(%arg16 = %15, %arg17 = %18) -> (!xetile.tile<32x64xf32>, !xetile.tile<32x32xf16>) {
      //CHECK: xetile.update_tile_offset %{{.*}}, [%c1024,  %c0] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      //CHECK: xetile.update_tile_offset %{{.*}}, [%c1024,  %c0] : !xetile.tile<32x64xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
      %28 = xetile.update_tile_offset %arg17, [%c1024,  %c0] :  !xetile.tile<32x32xf16>
      %29 = xetile.update_tile_offset %arg16, [%c1024,  %c0] : !xetile.tile<32x64xf32>
      %30:3 = scf.for %arg18 = %c0 to %c12288 step %c32 iter_args(%arg19 = %cst, %arg20 = %arg17, %arg21 = %26) -> (vector<32x64xf32>, !xetile.tile<32x32xf16>, !xetile.tile<64x32xf16>) {
        //CHECK: xetile.update_tile_offset %{{.*}}, [%c0,  %c32] : !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
        //CHECK: xetile.update_tile_offset %{{.*}}, [%c0,  %c32] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
        %32 = xetile.update_tile_offset %arg21, [%c0,  %c32] : !xetile.tile<64x32xf16>
        %33 = xetile.update_tile_offset %arg20, [%c0,  %c32] :  !xetile.tile<32x32xf16>
        %34 = xetile.load_tile %arg20 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %35 = math.exp %34 : vector<32x32xf16>
        %36 = xetile.load_tile %arg21 {padding = 0.000000e+00 : f32}  : !xetile.tile<64x32xf16> -> vector<64x32xf16>
        %37 = xetile.transpose %36, [1, 0] : vector<64x32xf16> -> vector<32x64xf16>
        %38 = math.exp %37 : vector<32x64xf16>
        xegpu.compile_hint
        %39 = xetile.tile_mma %35, %38, %cst : vector<32x32xf16>, vector<32x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
        xegpu.compile_hint
        %40 = arith.addf %arg19, %39 : vector<32x64xf32>
        scf.yield %40, %33, %32 : vector<32x64xf32>, !xetile.tile<32x32xf16>, !xetile.tile<64x32xf16>
      }
      %31 = math.exp %30#0 : vector<32x64xf32>
      xetile.store_tile %31,  %arg16 : vector<32x64xf32>, !xetile.tile<32x64xf32>
      scf.yield %29, %28 : !xetile.tile<32x64xf32>, !xetile.tile<32x32xf16>
    }
    gpu.return
  }

  gpu.func @sglevel_transpose_broadcast_dim_0(%arg0: memref<384x1xf32>, %arg1: memref<256x384xf32>) {
    %1 = xetile.init_tile %arg0[0, 0] : memref<384x1xf32> -> !xetile.tile<32x1xf32>
    %2 = xetile.load_tile %1 {padding = 0.000000e+00 : f32} : !xetile.tile<32x1xf32> -> vector<32x1xf32>
    %3 = xetile.transpose %2, [1, 0] : vector<32x1xf32> -> vector<1x32xf32>
    %4 = xetile.broadcast %3 [0] : vector<1x32xf32> -> vector<64x32xf32>
    %5 = xetile.init_tile %arg1[0, 0] : memref<256x384xf32> -> !xetile.tile<64x32xf32>
    xetile.store_tile %4, %5 : vector<64x32xf32>, !xetile.tile<64x32xf32>

    //CHECK: %[[r0:.*]] = xetile.init_tile %{{.*}}[0, 0] : memref<384x1xf32> -> !xetile.tile<32x1xf32, #xetile.tile_attr<inner_blocks = [32, 1]>>
    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] {padding = 0.000000e+00 : f32}  : !xetile.tile<32x1xf32, #xetile.tile_attr<inner_blocks = [32, 1]>> -> vector<1x1x32x1xf32>
    //CHECK: %[[r2:.*]] = xetile.tile_unpack %[[r1]] {inner_blocks = array<i64: 32, 1>}  : vector<1x1x32x1xf32> -> vector<32x1xf32>
    //CHECK: %[[r3:.*]] = xetile.tile_pack %[[r2]] {inner_blocks = array<i64: 16, 1>}  : vector<32x1xf32> -> vector<2x1x16x1xf32>
    //CHECK: %[[r4:.*]] = xetile.transpose %[[r3]], [1, 0, 3, 2] : vector<2x1x16x1xf32> -> vector<1x2x1x16xf32>
    //CHECK: %[[r5:.*]] = xetile.broadcast %[[r4]] [0, 2] : vector<1x2x1x16xf32> -> vector<64x2x1x16xf32>
    //CHECK: %[[r6:.*]] = xetile.tile_unpack %[[r5]] {inner_blocks = array<i64: 1, 16>}  : vector<64x2x1x16xf32> -> vector<64x32xf32>
    //CHECK: %[[r7:.*]] = xetile.init_tile %{{.*}}[0, 0] : memref<256x384xf32> -> !xetile.tile<64x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    //CHECK: %[[r8:.*]] = xetile.tile_pack %[[r6]] {inner_blocks = array<i64: 8, 16>}  : vector<64x32xf32> -> vector<8x2x8x16xf32>
    //CHECK: xetile.store_tile %[[r8]],  %[[r7]] : vector<8x2x8x16xf32>, !xetile.tile<64x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    gpu.return
  }

  gpu.func @sglevel_transpose_broadcast_dim_1(%arg0: memref<1x384xf16>, %arg1: memref<384x256xf16>) {
    %1 = xetile.init_tile %arg0[0, 0] : memref<1x384xf16> -> !xetile.tile<1x32xf16>
    %2 = xetile.load_tile %1 {padding = 0.000000e+00 : f32} : !xetile.tile<1x32xf16> -> vector<1x32xf16>
    %3 = xetile.transpose %2, [1, 0] : vector<1x32xf16> -> vector<32x1xf16>
    %4 = xetile.broadcast %3 [1] : vector<32x1xf16> -> vector<32x64xf16>
    %5 = xetile.init_tile %arg1[0, 0] : memref<384x256xf16> -> !xetile.tile<32x64xf16>
    xetile.store_tile %4, %5 : vector<32x64xf16>, !xetile.tile<32x64xf16>

    //CHECK: %[[r0:.*]] = xetile.init_tile %{{.*}}[0, 0] : memref<1x384xf16> -> !xetile.tile<1x32xf16, #xetile.tile_attr<inner_blocks = [1, 32]>>
    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] {padding = 0.000000e+00 : f32}  : !xetile.tile<1x32xf16, #xetile.tile_attr<inner_blocks = [1, 32]>> -> vector<1x1x1x32xf16>
    //CHECK: %[[r2:.*]] = xetile.transpose %[[r1]], [1, 0, 3, 2] : vector<1x1x1x32xf16> -> vector<1x1x32x1xf16>
    //CHECK: %[[r3:.*]] = xetile.tile_unpack %[[r2]] {inner_blocks = array<i64: 32, 1>}  : vector<1x1x32x1xf16> -> vector<32x1xf16>
    //CHECK: %[[r4:.*]] = xetile.tile_pack %[[r3]] {inner_blocks = array<i64: 1, 1>}  : vector<32x1xf16> -> vector<32x1x1x1xf16>
    //CHECK: %[[r5:.*]] = xetile.broadcast %[[r4]] [1, 3] : vector<32x1x1x1xf16> -> vector<32x2x1x32xf16>
    //CHECK: %[[r6:.*]] = xetile.tile_unpack %[[r5]] {inner_blocks = array<i64: 1, 32>}  : vector<32x2x1x32xf16> -> vector<32x64xf16>
    //CHECK: %[[r7:.*]] = xetile.init_tile %{{.*}}[0, 0] : memref<384x256xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    //CHECK: %[[r8:.*]] = xetile.tile_pack %[[r6]] {inner_blocks = array<i64: 8, 32>}  : vector<32x64xf16> -> vector<4x2x8x32xf16>
    //CHECK: xetile.store_tile %[[r8]],  %[[r7]] : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_loadgather
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>, %[[arg1:.*]]: vector<4x32xindex>
  gpu.func @sg_loadgather(%a: memref<1024xf16>, %indices: vector<4x32xindex>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<4x1x1x32xi1>
    //CHECK: %[[r0:.*]] = xetile.tile_pack %[[arg1]] {inner_blocks = array<i64: 1, 32>} : vector<4x32xindex> -> vector<4x1x1x32xindex>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]], %[[r0]] : memref<1024xf16>, vector<4x1x1x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 32], scattered = true>>
    //CHECK: %[[r2:.*]] = xetile.load %[[r1]], %[[cst]] : !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 32], scattered = true>>, vector<4x1x1x32xi1> -> vector<4x1x1x32xf16>
    %mask = arith.constant dense<1> : vector<4x32xi1>
    %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
    %2 = xetile.load %1, %mask : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1> -> vector<4x32xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_storescatter
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>, %[[arg1:.*]]: vector<4x32xindex>
  gpu.func @sg_storescatter(%a: memref<1024xf16>, %indices: vector<4x32xindex>) {

    //CHECK: %[[cst:.*]] = arith.constant dense<4.200000e+01> : vector<4x1x1x32xf16>
    //CHECK: %[[cst_0:.*]] = arith.constant dense<true> : vector<4x1x1x32xi1>
    //CHECK: %[[cst_1:.*]] = arith.constant dense<1> : vector<4x1x1x32xindex>
    //CHECK: %[[r0:.*]] = xetile.tile_pack %[[arg1]] {inner_blocks = array<i64: 1, 32>} : vector<4x32xindex> -> vector<4x1x1x32xindex>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]], %[[r0]] : memref<1024xf16>, vector<4x1x1x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 32], scattered = true>>
    //CHECK: xetile.store %[[cst]], %[[r1]], %[[cst_0]] : vector<4x1x1x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 32], scattered = true>>, vector<4x1x1x32xi1>
    //CHECK: %[[r2]] = xetile.update_tile_offset %[[r1]], %[[cst_1]] : !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 32], scattered = true>>, vector<4x1x1x32xindex>
    //CHECK: xetile.store %[[cst]], %[[r2]], %[[cst_0]] : vector<4x1x1x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<inner_blocks = [1, 32], scattered = true>>, vector<4x1x1x32xi1>
    %offsets = arith.constant dense<1> : vector<4x32xindex>
    %mask = arith.constant dense<1> : vector<4x32xi1>
    %data = arith.constant dense<42.0> : vector<4x32xf16>
    %tile = xetile.init_tile %a, %indices : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
    xetile.store %data, %tile, %mask : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
    %next = xetile.update_tile_offset %tile, %offsets : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xindex>
    xetile.store %data, %next, %mask : vector<4x32xf16>, !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @add_kernel
  //CHECK-SAME: %[[arg0:.*]]: memref<*xf32>, %[[arg1:.*]]: memref<*xf32>, %[[arg2:.*]]: memref<*xf32>
  gpu.func @add_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>) {
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<1x2x1x16xi1>
    //CHECK: %[[cast:.*]] = memref.cast %[[arg0:.*]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_0:.*]] = memref.cast %[[arg1:.*]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_1:.*]] = memref.cast %[[arg2:.*]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
    //CHECK: %[[r0:.*]] = arith.muli %[[block_id_x]], %c1024 : index
    //CHECK: %[[r1:.*]] = vector.splat %[[r0]] : vector<1x2x1x16xindex>
    //CHECK: %[[r2:.*]] = xetile.init_tile %[[cast]], %[[r1]] : memref<?xf32>, vector<1x2x1x16xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>
    //CHECK: %[[r3:.*]] = xetile.load %[[r2]], %[[cst]] : !xetile.tile<1x32xf32, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>, vector<1x2x1x16xi1> -> vector<1x2x1x16xf32>
    //CHECK: %[[r4:.*]] = xetile.init_tile %[[cast_0]], %[[r1]] : memref<?xf32>, vector<1x2x1x16xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>
    //CHECK: %[[r5:.*]] = xetile.load %[[r4]], %[[cst]] : !xetile.tile<1x32xf32, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>, vector<1x2x1x16xi1> -> vector<1x2x1x16xf32>
    //CHECK: %[[r6:.*]] = arith.addf %[[r3]], %[[r5]] : vector<1x2x1x16xf32>
    //CHECK: %[[r7:.*]] = xetile.init_tile %[[cast_1]], %[[r1]] : memref<?xf32>, vector<1x2x1x16xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>
    //CHECK: xetile.store %[[r6]], %[[r7]], %[[cst]] : vector<1x2x1x16xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<inner_blocks = [1, 16], scattered = true>>, vector<1x2x1x16xi1>

    %cst = arith.constant dense<true> : vector<1x32xi1>
    %c1024 = arith.constant 1024 : index
    %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %block_id_x = gpu.block_id  x
    %0 = arith.muli %block_id_x, %c1024 : index
    %1 = vector.splat %0 : vector<1x32xindex>
    %2 = xetile.init_tile %cast, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    %3 = xetile.load %2, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
    %4 = xetile.init_tile %cast_0, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    %5 = xetile.load %4, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
    %6 = arith.addf %3, %5 : vector<1x32xf32>
    %7 = xetile.init_tile %cast_1, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    xetile.store %6, %7, %cst : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    gpu.return
  }
}
