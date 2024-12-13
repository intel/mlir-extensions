// RUN: imex-opt --split-input-file --xetile-blocking="enable-2d-transform=true" %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_load_tile(%[[arg0:.*]]: memref<32x32xf16>)
  gpu.func @sg_load_tile(%a: memref<32x32xf16>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    %c0 = arith.constant 0 : index
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  	gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_unaligned(%[[arg0:.*]]: memref<128x128xf16>)
  gpu.func @sg_load_tile_unaligned(%a: memref<128x128xf16>) {
    %c0 = arith.constant 0 : index
    //CHECK-COUNT-20: xetile.init_tile %arg0[{{.*}}] : memref<128x128xf16> -> !xetile.tile<17x19xf16>
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<85x76xf16>
    //CHECK-COUNT-20: xetile.load_tile {{.*}} : !xetile.tile<17x19xf16> -> vector<17x19xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<85x76xf16> -> vector<85x76xf16>
    //CHECK: gpu.return
    gpu.return
  }

  //CHECK: gpu.func @sg_store_tile(%[[arg0:.*]]: memref<32x32xf32>)
	gpu.func @sg_store_tile(%a: memref<32x32xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
		%result = arith.constant dense<0.0>: vector<32x32xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %arg0[%{{.*}}, %{{.*}}] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
		%1 = xetile.init_tile %a[0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
    //CHECK-COUNT-8: xetile.store_tile %[[cst]],  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
		xetile.store_tile %result, %1: vector<32x32xf32>, !xetile.tile<32x32xf32>
    //CHECK: gpu.return
		gpu.return
	}

  //CHECK: gpu.func @create_mask
  //CHECK-SAME: %[[arg0:.*]]: vector<32x32xf16>, %[[arg1:.*]]: vector<32x32xf16>, %[[arg2:.*]]: memref<32x32xf16>
  gpu.func @create_mask(%a: vector<32x32xf16>, %b: vector<32x32xf16>, %c: memref<32x32xf16>) {
    %c32 = arith.constant 32 : index
    %c20 = arith.constant 20 : index

    //CHECK: %[[r0:.*]] = vector.constant_mask [8, 20] : vector<8x32xi1>
    %mask = vector.create_mask %c32, %c20 : vector<32x32xi1>

    //CHECK-COUNT-4: vector.extract_strided_slice %[[arg0]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-COUNT-4: vector.extract_strided_slice %[[arg1]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-COUNT-4: arith.select %[[r0]], %{{.*}}, %{{.*}} : vector<8x32xi1>, vector<8x32xf16>
    %select = arith.select %mask, %a, %b : vector<32x32xi1>, vector<32x32xf16>

    //CHECK-COUNT-4: xetile.init_tile %[[arg2]][{{.*}}] : memref<32x32xf16> -> !xetile.tile<8x32xf16>
    %tile = xetile.init_tile %c[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>

    //CHECK-COUNT-4: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    xetile.store_tile %select, %tile: vector<32x32xf16>, !xetile.tile<32x32xf16>

    //CHECK: gpu.return
    gpu.return
  }

  //CHECK: gpu.func @create_mask_2
  //CHECK-SAME: %[[arg0:.*]]: vector<32x32xf16>, %[[arg1:.*]]: vector<32x32xf16>, %[[arg2:.*]]: memref<32x32xf16>
  gpu.func @create_mask_2(%a: vector<32x32xf16>, %b: vector<32x32xf16>, %c: memref<32x32xf16>) {
    %c20 = arith.constant 20 : index
    %c32 = arith.constant 32 : index
    //CHECK: %[[r0:.*]] = vector.constant_mask [8, 32] : vector<8x32xi1>
    //CHECK: %[[r1:.*]] = vector.constant_mask [4, 32] : vector<8x32xi1>
    //CHECK: %[[r2:.*]] = vector.constant_mask [0, 0] : vector<8x32xi1>
    %mask = vector.create_mask %c20, %c32 : vector<32x32xi1>

    //CHECK-COUNT-4: vector.extract_strided_slice %[[arg0]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-COUNT-4: vector.extract_strided_slice %[[arg1]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-COUNT-4: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<8x32xi1>, vector<8x32xf16>
    %select = arith.select %mask, %a, %b : vector<32x32xi1>, vector<32x32xf16>

    //CHECK-COUNT-4: xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<32x32xf16> -> !xetile.tile<8x32xf16>
    %tile = xetile.init_tile %c[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK-COUNT-4: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    xetile.store_tile %select, %tile: vector<32x32xf16>, !xetile.tile<32x32xf16>
    gpu.return
  }

  //CHECK: gpu.func @sg_store_tile_unaligned(%[[arg0:.*]]: memref<128x128xf32>)
	gpu.func @sg_store_tile_unaligned(%a: memref<128x128xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4x2xf32>
	  %result = arith.constant dense<0.0>: vector<44x38xf32>
    //CHECK-COUNT-209: xetile.init_tile %[[arg0]][{{.*}}] : memref<128x128xf32> -> !xetile.tile<4x2xf32>
	  %1 = xetile.init_tile %a[0, 0] : memref<128x128xf32> -> !xetile.tile<44x38xf32>
    //CHECK-COUNT-209: xetile.store_tile %[[cst]],  %{{.*}} : vector<4x2xf32>, !xetile.tile<4x2xf32>
	  xetile.store_tile %result, %1: vector<44x38xf32>, !xetile.tile<44x38xf32>
    //CHECK: gpu.return
	  gpu.return
	}

  //CHECK: gpu.func @sg_tile_mma(%[[arg0:.*]]: memref<32x32xf16>, %[[arg1:.*]]: memref<32x32xf16>)
  //CHECK-COUNT-2: xetile.init_tile %[[arg0]][{{.*}}] : memref<32x32xf16> -> !xetile.tile<32x16xf16>
  //CHECK-COUNT-2: xetile.load_tile %{{.*}} : !xetile.tile<32x16xf16> -> vector<32x16xf16>
  //CHECK-COUNT-8: vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
  //CHECK-COUNT-2: xetile.init_tile %[[arg1]][{{.*}}] : memref<32x32xf16> -> !xetile.tile<32x16xf16>
  //CHECK-COUNT-2: xetile.load_tile %{{.*}} : !xetile.tile<32x16xf16> -> vector<32x16xf16>
  //CHECK-COUNT-4: vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
  //CHECK-COUNT-16: xetile.tile_mma %{{.*}}, %{{.*}}{{.*}} : vector<8x16xf16>, vector<16x16xf16>{{.*}} -> vector<8x16xf32>
  gpu.func @sg_tile_mma(%a: memref<32x32xf16>, %b: memref<32x32xf16>) {
    %c0 = arith.constant 0 : index
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  	%3 = xetile.init_tile %b[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    %4 = xetile.load_tile %3 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    %5 = xetile.tile_mma %2, %4: vector<32x32xf16>, vector<32x32xf16> -> vector<32x32xf32>
  	gpu.return
  }

  //CHECK: gpu.func @sg_tile_mma_b_transpose(%[[arg0:.*]]: memref<64x32xf16>, %[[arg1:.*]]: memref<64x32xf16>, %[[arg2:.*]]: memref<64x64xf32>)
  gpu.func @sg_tile_mma_b_transpose(%a: memref<64x32xf16>, %b: memref<64x32xf16>, %c: memref<64x64xf32>) {
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}] : memref<64x32xf16> -> !xetile.tile<32x16xf16>
    %0 = xetile.init_tile %a[0, 0] : memref<64x32xf16> -> !xetile.tile<64x32xf16>
    //CHECK-COUNT-4: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x16xf16> -> vector<32x16xf16>
    //CHECK-COUNT-16: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    %1 = xetile.load_tile %0 : !xetile.tile<64x32xf16> -> vector<64x32xf16>

    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg1]][{{.*}}] : memref<64x32xf16> -> !xetile.tile<16x16xf16>
    %2 = xetile.init_tile %b[0, 0] : memref<64x32xf16> -> !xetile.tile<64x32xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<16x16xf16> -> vector<16x16xf16>
    %3 = xetile.load_tile %2 : !xetile.tile<64x32xf16> -> vector<64x32xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.transpose %{{.*}}, [1, 0] : vector<16x16xf16> -> vector<16x16xf16>
    %4 = xetile.transpose %3, [1, 0] : vector<64x32xf16> -> vector<32x64xf16>
    //CHECK-COUNT-64: %{{.*}} = xetile.tile_mma {{.*}} : vector<8x16xf16>, vector<16x16xf16>{{.*}}-> vector<8x16xf32>
    %5 = xetile.tile_mma %1, %4: vector<64x32xf16>, vector<32x64xf16> -> vector<64x64xf32>
    //CHECK-COUNT-32: xetile.init_tile %[[arg2]][{{.*}}] : memref<64x64xf32> -> !xetile.tile<8x16xf32>
    %6 = xetile.init_tile %c[0, 0] : memref<64x64xf32> -> !xetile.tile<64x64xf32>
    //CHECK-COUNT-32: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %5, %6: vector<64x64xf32>, !xetile.tile<64x64xf32>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @inner_reduction
  // CHECK-SAME: (%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>)
  gpu.func @inner_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
    //CHECK: %[[c15_i32:.*]] = arith.constant 15 : i32
    //CHECK: %[[c14_i32:.*]] = arith.constant 14 : i32
    //CHECK: %[[c13_i32:.*]] = arith.constant 13 : i32
    //CHECK: %[[c12_i32:.*]] = arith.constant 12 : i32
    //CHECK: %[[c11_i32:.*]] = arith.constant 11 : i32
    //CHECK: %[[c10_i32:.*]] = arith.constant 10 : i32
    //CHECK: %[[c9_i32:.*]] = arith.constant 9 : i32
    //CHECK: %[[c8_i32:.*]] = arith.constant 8 : i32
    //CHECK: %[[c7_i32:.*]] = arith.constant 7 : i32
    //CHECK: %[[c6_i32:.*]] = arith.constant 6 : i32
    //CHECK: %[[c5_i32:.*]] = arith.constant 5 : i32
    //CHECK: %[[c4_i32:.*]] = arith.constant 4 : i32
    //CHECK: %[[c3_i32:.*]] = arith.constant 3 : i32
    //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
    //CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
    //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<16x32xf16> -> vector<16x32xf16>
    //CHECK: %[[r2:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r4:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r10:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r11:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r12:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r13:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r14:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r18:.*]] = math.exp %[[r2]] : vector<1x32xf16>
    //CHECK: %[[r19:.*]] = math.exp %[[r3]] : vector<1x32xf16>
    //CHECK: %[[r20:.*]] = math.exp %[[r4]] : vector<1x32xf16>
    //CHECK: %[[r21:.*]] = math.exp %[[r5]] : vector<1x32xf16>
    //CHECK: %[[r22:.*]] = math.exp %[[r6]] : vector<1x32xf16>
    //CHECK: %[[r23:.*]] = math.exp %[[r7]] : vector<1x32xf16>
    //CHECK: %[[r24:.*]] = math.exp %[[r8]] : vector<1x32xf16>
    //CHECK: %[[r25:.*]] = math.exp %[[r9]] : vector<1x32xf16>
    //CHECK: %[[r26:.*]] = math.exp %[[r10]] : vector<1x32xf16>
    //CHECK: %[[r27:.*]] = math.exp %[[r11]] : vector<1x32xf16>
    //CHECK: %[[r28:.*]] = math.exp %[[r12]] : vector<1x32xf16>
    //CHECK: %[[r29:.*]] = math.exp %[[r13]] : vector<1x32xf16>
    //CHECK: %[[r30:.*]] = math.exp %[[r14]] : vector<1x32xf16>
    //CHECK: %[[r31:.*]] = math.exp %[[r15]] : vector<1x32xf16>
    //CHECK: %[[r32:.*]] = math.exp %[[r16]] : vector<1x32xf16>
    //CHECK: %[[r33:.*]] = math.exp %[[r17]] : vector<1x32xf16>
    //CHECK: %[[r34:.*]] = vector.shape_cast %[[r18]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r35:.*]] = vector.shape_cast %[[r19]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r36:.*]] = vector.shape_cast %[[r20]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r37:.*]] = vector.shape_cast %[[r21]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r38:.*]] = vector.shape_cast %[[r22]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r39:.*]] = vector.shape_cast %[[r23]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r40:.*]] = vector.shape_cast %[[r24]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r41:.*]] = vector.shape_cast %[[r25]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r42:.*]] = vector.shape_cast %[[r26]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r43:.*]] = vector.shape_cast %[[r27]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r44:.*]] = vector.shape_cast %[[r28]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r45:.*]] = vector.shape_cast %[[r29]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r46:.*]] = vector.shape_cast %[[r30]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r47:.*]] = vector.shape_cast %[[r31]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r48:.*]] = vector.shape_cast %[[r32]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r49:.*]] = vector.shape_cast %[[r33]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r50:.*]] = vector.shuffle %[[r34]], %[[r35]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r51:.*]] = vector.shuffle %[[r34]], %[[r35]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r52:.*]] = arith.addf %[[r50]], %[[r51]] : vector<32xf16>
    //CHECK: %[[r53:.*]] = vector.shuffle %[[r36]], %[[r37]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r54:.*]] = vector.shuffle %[[r36]], %[[r37]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r55:.*]] = arith.addf %[[r53]], %[[r54]] : vector<32xf16>
    //CHECK: %[[r56:.*]] = vector.shuffle %[[r38]], %[[r39]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r57:.*]] = vector.shuffle %[[r38]], %[[r39]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r58:.*]] = arith.addf %[[r56]], %[[r57]] : vector<32xf16>
    //CHECK: %[[r59:.*]] = vector.shuffle %[[r40]], %[[r41]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r60:.*]] = vector.shuffle %[[r40]], %[[r41]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r61:.*]] = arith.addf %[[r59]], %[[r60]] : vector<32xf16>
    //CHECK: %[[r62:.*]] = vector.shuffle %[[r42]], %[[r43]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r63:.*]] = vector.shuffle %[[r42]], %[[r43]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r64:.*]] = arith.addf %[[r62]], %[[r63]] : vector<32xf16>
    //CHECK: %[[r65:.*]] = vector.shuffle %[[r44]], %[[r45]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r66:.*]] = vector.shuffle %[[r44]], %[[r45]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r67:.*]] = arith.addf %[[r65]], %[[r66]] : vector<32xf16>
    //CHECK: %[[r68:.*]] = vector.shuffle %[[r46]], %[[r47]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r69:.*]] = vector.shuffle %[[r46]], %[[r47]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r70:.*]] = arith.addf %[[r68]], %[[r69]] : vector<32xf16>
    //CHECK: %[[r71:.*]] = vector.shuffle %[[r48]], %[[r49]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r72:.*]] = vector.shuffle %[[r48]], %[[r49]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r73:.*]] = arith.addf %[[r71]], %[[r72]] : vector<32xf16>
    //CHECK: %[[r74:.*]] = vector.shuffle %[[r52]], %[[r55]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r75:.*]] = vector.shuffle %[[r52]], %[[r55]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r76:.*]] = arith.addf %[[r74]], %[[r75]] : vector<32xf16>
    //CHECK: %[[r77:.*]] = vector.shuffle %[[r58]], %[[r61]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r78:.*]] = vector.shuffle %[[r58]], %[[r61]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r79:.*]] = arith.addf %[[r77]], %[[r78]] : vector<32xf16>
    //CHECK: %[[r80:.*]] = vector.shuffle %[[r64]], %[[r67]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r81:.*]] = vector.shuffle %[[r64]], %[[r67]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r82:.*]] = arith.addf %[[r80]], %[[r81]] : vector<32xf16>
    //CHECK: %[[r83:.*]] = vector.shuffle %[[r70]], %[[r73]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r84:.*]] = vector.shuffle %[[r70]], %[[r73]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r85:.*]] = arith.addf %[[r83]], %[[r84]] : vector<32xf16>
    //CHECK: %[[r86:.*]] = vector.shuffle %[[r76]], %[[r79]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r87:.*]] = vector.shuffle %[[r76]], %[[r79]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r88:.*]] = arith.addf %[[r86]], %[[r87]] : vector<32xf16>
    //CHECK: %[[r89:.*]] = vector.shuffle %[[r82]], %[[r85]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r90:.*]] = vector.shuffle %[[r82]], %[[r85]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r91:.*]] = arith.addf %[[r89]], %[[r90]] : vector<32xf16>
    //CHECK: %[[r92:.*]] = vector.shuffle %[[r88]], %[[r91]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r93:.*]] = vector.shuffle %[[r88]], %[[r91]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r94:.*]] = arith.addf %[[r92]], %[[r93]] : vector<32xf16>
    //CHECK: %[[r95:.*]] = vector.shuffle %[[r94]], %[[r94]] [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r96:.*]] = vector.shuffle %[[r94]], %[[r94]] [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r97:.*]] = arith.addf %[[r95]], %[[r96]] : vector<16xf16>
    //CHECK: %[[r98:.*]] = vector.extractelement %[[r97]][%[[c0_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r99:.*]] = vector.splat %[[r98]] : vector<1x1xf16>
    //CHECK: %[[r100:.*]] = vector.extractelement %[[r97]][%[[c1_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r101:.*]] = vector.splat %[[r100]] : vector<1x1xf16>
    //CHECK: %[[r102:.*]] = vector.extractelement %[[r97]][%[[c2_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r103:.*]] = vector.splat %[[r102]] : vector<1x1xf16>
    //CHECK: %[[r104:.*]] = vector.extractelement %[[r97]][%[[c3_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r105:.*]] = vector.splat %[[r104]] : vector<1x1xf16>
    //CHECK: %[[r106:.*]] = vector.extractelement %[[r97]][%[[c4_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r107:.*]] = vector.splat %[[r106]] : vector<1x1xf16>
    //CHECK: %[[r108:.*]] = vector.extractelement %[[r97]][%[[c5_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r109:.*]] = vector.splat %[[r108]] : vector<1x1xf16>
    //CHECK: %[[r110:.*]] = vector.extractelement %[[r97]][%[[c6_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r111:.*]] = vector.splat %[[r110]] : vector<1x1xf16>
    //CHECK: %[[r112:.*]] = vector.extractelement %[[r97]][%[[c7_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r113:.*]] = vector.splat %[[r112]] : vector<1x1xf16>
    //CHECK: %[[r114:.*]] = vector.extractelement %[[r97]][%[[c8_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r115:.*]] = vector.splat %[[r114]] : vector<1x1xf16>
    //CHECK: %[[r116:.*]] = vector.extractelement %[[r97]][%[[c9_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r117:.*]] = vector.splat %[[r116]] : vector<1x1xf16>
    //CHECK: %[[r118:.*]] = vector.extractelement %[[r97]][%[[c10_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r119:.*]] = vector.splat %[[r118]] : vector<1x1xf16>
    //CHECK: %[[r120:.*]] = vector.extractelement %[[r97]][%[[c11_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r121:.*]] = vector.splat %[[r120]] : vector<1x1xf16>
    //CHECK: %[[r122:.*]] = vector.extractelement %[[r97]][%[[c12_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r123:.*]] = vector.splat %[[r122]] : vector<1x1xf16>
    //CHECK: %[[r124:.*]] = vector.extractelement %[[r97]][%[[c13_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r125:.*]] = vector.splat %[[r124]] : vector<1x1xf16>
    //CHECK: %[[r126:.*]] = vector.extractelement %[[r97]][%[[c14_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r127:.*]] = vector.splat %[[r126]] : vector<1x1xf16>
    //CHECK: %[[r128:.*]] = vector.extractelement %[[r97]][%[[c15_i32]] : i32] : vector<16xf16>
    //CHECK: %[[r129:.*]] = vector.splat %[[r128]] : vector<1x1xf16>
    //CHECK: %[[r130:.*]] = vector.shuffle %[[r99]], %[[r101]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r131:.*]] = vector.shuffle %[[r103]], %[[r105]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r132:.*]] = vector.shuffle %[[r107]], %[[r109]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r133:.*]] = vector.shuffle %[[r111]], %[[r113]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r134:.*]] = vector.shuffle %[[r115]], %[[r117]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r135:.*]] = vector.shuffle %[[r119]], %[[r121]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r136:.*]] = vector.shuffle %[[r123]], %[[r125]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r137:.*]] = vector.shuffle %[[r127]], %[[r129]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r138:.*]] = vector.shuffle %[[r130]], %[[r131]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r139:.*]] = vector.shuffle %[[r132]], %[[r133]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r140:.*]] = vector.shuffle %[[r134]], %[[r135]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r141:.*]] = vector.shuffle %[[r136]], %[[r137]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r142:.*]] = vector.shuffle %[[r138]], %[[r139]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r143:.*]] = vector.shuffle %[[r140]], %[[r141]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r144:.*]] = vector.shuffle %[[r142]], %[[r143]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
    //CHECK: %[[r145:.*]] = vector.shape_cast %[[r144]] : vector<16x1xf16> to vector<2x8xf16>
    //CHECK: %[[r146:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<2x8xf16>
    //CHECK: xetile.store_tile %[[r145]],  %[[r146]] : vector<2x8xf16>, !xetile.tile<2x8xf16>
    %c0 = arith.constant 0 : index
    %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
    %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
    %e = math.exp %v: vector<16x32xf16>
    %r = xetile.reduction <add>, %e [1] : vector<16x32xf16> -> vector<16x1xf16>
    %c = vector.shape_cast %r: vector<16x1xf16> to vector<2x8xf16>
    %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<2x8xf16>
    xetile.store_tile %c, %s : vector<2x8xf16>, !xetile.tile<2x8xf16>
    gpu.return
  }

  //CHECK: gpu.func @outter_reduction(%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>) {
  gpu.func @outter_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<16x32xf16> -> vector<16x32xf16>
    //CHECK: %[[r2:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r4:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r10:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r11:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r12:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r13:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r14:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
    //CHECK: %[[r18:.*]] = math.exp %[[r2:.*]] : vector<1x32xf16>
    //CHECK: %[[r19:.*]] = math.exp %[[r3:.*]] : vector<1x32xf16>
    //CHECK: %[[r20:.*]] = math.exp %[[r4:.*]] : vector<1x32xf16>
    //CHECK: %[[r21:.*]] = math.exp %[[r5:.*]] : vector<1x32xf16>
    //CHECK: %[[r22:.*]] = math.exp %[[r6:.*]] : vector<1x32xf16>
    //CHECK: %[[r23:.*]] = math.exp %[[r7:.*]] : vector<1x32xf16>
    //CHECK: %[[r24:.*]] = math.exp %[[r8:.*]] : vector<1x32xf16>
    //CHECK: %[[r25:.*]] = math.exp %[[r9:.*]] : vector<1x32xf16>
    //CHECK: %[[r26:.*]] = math.exp %[[r10:.*]] : vector<1x32xf16>
    //CHECK: %[[r27:.*]] = math.exp %[[r11:.*]] : vector<1x32xf16>
    //CHECK: %[[r28:.*]] = math.exp %[[r12:.*]] : vector<1x32xf16>
    //CHECK: %[[r29:.*]] = math.exp %[[r13:.*]] : vector<1x32xf16>
    //CHECK: %[[r30:.*]] = math.exp %[[r14:.*]] : vector<1x32xf16>
    //CHECK: %[[r31:.*]] = math.exp %[[r15:.*]] : vector<1x32xf16>
    //CHECK: %[[r32:.*]] = math.exp %[[r16:.*]] : vector<1x32xf16>
    //CHECK: %[[r33:.*]] = math.exp %[[r17:.*]] : vector<1x32xf16>
    //CHECK: %[[r34:.*]] = arith.addf %[[r18]], %[[r19]] : vector<1x32xf16>
    //CHECK: %[[r35:.*]] = arith.addf %[[r34]], %[[r20]] : vector<1x32xf16>
    //CHECK: %[[r36:.*]] = arith.addf %[[r35]], %[[r21]] : vector<1x32xf16>
    //CHECK: %[[r37:.*]] = arith.addf %[[r36]], %[[r22]] : vector<1x32xf16>
    //CHECK: %[[r38:.*]] = arith.addf %[[r37]], %[[r23]] : vector<1x32xf16>
    //CHECK: %[[r39:.*]] = arith.addf %[[r38]], %[[r24]] : vector<1x32xf16>
    //CHECK: %[[r40:.*]] = arith.addf %[[r39]], %[[r25]] : vector<1x32xf16>
    //CHECK: %[[r41:.*]] = arith.addf %[[r40]], %[[r26]] : vector<1x32xf16>
    //CHECK: %[[r42:.*]] = arith.addf %[[r41]], %[[r27]] : vector<1x32xf16>
    //CHECK: %[[r43:.*]] = arith.addf %[[r42]], %[[r28]] : vector<1x32xf16>
    //CHECK: %[[r44:.*]] = arith.addf %[[r43]], %[[r29]] : vector<1x32xf16>
    //CHECK: %[[r45:.*]] = arith.addf %[[r44]], %[[r30]] : vector<1x32xf16>
    //CHECK: %[[r46:.*]] = arith.addf %[[r45]], %[[r31]] : vector<1x32xf16>
    //CHECK: %[[r47:.*]] = arith.addf %[[r46]], %[[r32]] : vector<1x32xf16>
    //CHECK: %[[r48:.*]] = arith.addf %[[r47]], %[[r33]] : vector<1x32xf16>
    //CHECK: %[[r49:.*]] = vector.shape_cast %[[r48]] : vector<1x32xf16> to vector<4x8xf16>
    //CHECK: %[[r50:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<4x8xf16>
    //CHECK: xetile.store_tile %[[r49]],  %[[r50]] : vector<4x8xf16>, !xetile.tile<4x8xf16>
    %c0 = arith.constant 0 : index
    %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
    %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
    %e = math.exp %v: vector<16x32xf16>
    %r = xetile.reduction <add>, %e [0] : vector<16x32xf16> -> vector<1x32xf16>
    %c = vector.shape_cast %r: vector<1x32xf16> to vector<4x8xf16>
    %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<4x8xf16>
    xetile.store_tile %c, %s : vector<4x8xf16>, !xetile.tile<4x8xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_gemm
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x128xf16>, %[[arg1:.*]]: memref<128x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_gemm(%a: memref<32x128xf16>, %b: memref<128x32xf16>, %c: memref<32x32xf32>) {

    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c16]]] : memref<32x128xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %[[r2:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %[[r3:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c16]]] : memref<128x32xf16> -> !xetile.tile<32x16xf16>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant dense<0.0>: vector<32x32xf32>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r4:.*]]:12 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]] iter_args(%[[arg4:.*]] = %[[r0]], %[[arg5:.*]] = %[[r1]], %[[arg6:.*]] = %[[r2]], %[[arg7:.*]] = %[[r3]], %[[arg8:.*]] = %[[cst]], %[[arg9:.*]] = %[[cst]], %[[arg10:.*]] = %[[cst]], %[[arg11:.*]] = %[[cst]], %[[arg12:.*]] = %[[cst]], %[[arg13:.*]] = %[[cst]], %[[arg14:.*]] = %[[cst]], %[[arg15:.*]] = %[[cst]]) -> (!xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      //CHECK: %[[r13:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      //CHECK: %[[r14:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r18:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r19:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r20:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r21:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r22:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK: %[[r23:.*]] = xetile.load_tile %[[arg6]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      //CHECK: %[[r24:.*]] = xetile.load_tile %[[arg7]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
      //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r23]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r23]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r24]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r28:.*]] = vector.extract_strided_slice %[[r24]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK: %[[r29:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]], %[[c32]]] : !xetile.tile<32x16xf16>
      //CHECK: %[[r30:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c0]], %[[c32]]] : !xetile.tile<32x16xf16>
      //CHECK: %[[r31:.*]] = xetile.update_tile_offset %[[arg6]], [%[[c32]], %[[c0]]] : !xetile.tile<32x16xf16>
      //CHECK: %[[r32:.*]] = xetile.update_tile_offset %[[arg7]], [%[[c32]], %[[c0]]] : !xetile.tile<32x16xf16>
      //CHECK: %[[r33:.*]] = xetile.tile_mma %[[r15]], %[[r25]], %[[arg8]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r34:.*]] = xetile.tile_mma %[[r19]], %[[r26]], %[[r33]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r35:.*]] = xetile.tile_mma %[[r15]], %[[r27]], %[[arg9]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r36:.*]] = xetile.tile_mma %[[r19]], %[[r28]], %[[r35]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r37:.*]] = xetile.tile_mma %[[r16]], %[[r25]], %[[arg10]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r38:.*]] = xetile.tile_mma %[[r20]], %[[r26]], %[[r37]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r39:.*]] = xetile.tile_mma %[[r16]], %[[r27]], %[[arg11]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r40:.*]] = xetile.tile_mma %[[r20]], %[[r28]], %[[r39]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r41:.*]] = xetile.tile_mma %[[r17]], %[[r25]], %[[arg12]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r42:.*]] = xetile.tile_mma %[[r21]], %[[r26]], %[[r41]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r43:.*]] = xetile.tile_mma %[[r17]], %[[r27]], %[[arg13]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r44:.*]] = xetile.tile_mma %[[r21]], %[[r28]], %[[r43]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r45:.*]] = xetile.tile_mma %[[r18]], %[[r25]], %[[arg14]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r46:.*]] = xetile.tile_mma %[[r22]], %[[r26]], %[[r45]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r47:.*]] = xetile.tile_mma %[[r18]], %[[r27]], %[[arg15]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r48:.*]] = xetile.tile_mma %[[r22]], %[[r28]], %[[r47]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]:  !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]:  !xetile.tile<32x32xf16>
      %c_new_value = xetile.tile_mma %3, %4, %c_value:
        vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      //CHECK: scf.yield %[[r29]], %[[r30]], %[[r31]], %[[r32]], %[[r34]], %[[r36]], %[[r38]], %[[r40]], %[[r42]], %[[r44]], %[[r46]], %[[r48]] : !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }

    //CHECK: %[[r5:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r7:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r8:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r9:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r10:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r11:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r12:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>

    //CHECK: xetile.store_tile %[[r4]]#4,  %[[r5]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#5,  %[[r6]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#6,  %[[r7]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#7,  %[[r8]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#8,  %[[r9]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#9,  %[[r10]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#10,  %[[r11]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#11,  %[[r12]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
  	gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_gemm_with_preops_for_c
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x128xf16>, %[[arg1:.*]]: memref<128x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_gemm_with_preops_for_c(%a: memref<32x128xf16>, %b: memref<128x32xf16>, %c: memref<32x32xf32>) {
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK: %[[r0]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %[[r1]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c16]]] : memref<32x128xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %[[r2]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %[[r3]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c16]]] : memref<128x32xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %[[r4]]:12 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]] iter_args(%[[arg4:.*]] = %[[r0]], %[[arg5:.*]] = %[[r1]], %[[arg6:.*]] = %[[r2]], %[[arg7:.*]] = %[[r3]], %[[arg8:.*]] = %[[cst]], %[[arg9:.*]] = %[[cst]], %[[arg10:.*]] = %[[cst]], %[[arg11:.*]] = %[[cst]], %[[arg12:.*]] = %[[cst]], %[[arg13:.*]] = %[[cst]], %[[arg14:.*]] = %[[cst]], %[[arg15:.*]] = %[[cst]]) -> (!xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
    //CHECK:   %[[r13:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
    //CHECK:   %[[r14:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
    //CHECK:   %[[r15:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r16:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r17:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r18:.*]] = vector.extract_strided_slice %[[r13]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r19:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r20:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r21:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r22:.*]] = vector.extract_strided_slice %[[r14]] {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r23:.*]] = xetile.load_tile %[[arg6]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
    //CHECK:   %[[r24:.*]] = xetile.load_tile %[[arg7]] : !xetile.tile<32x16xf16> -> vector<32x16xf16>
    //CHECK:   %[[r25:.*]] = vector.extract_strided_slice %[[r23]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK:   %[[r26:.*]] = vector.extract_strided_slice %[[r23]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK:   %[[r27:.*]] = vector.extract_strided_slice %[[r24]] {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK:   %[[r28:.*]] = vector.extract_strided_slice %[[r24]] {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK:   %[[r29:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]], %[[c32]]] : !xetile.tile<32x16xf16>
    //CHECK:   %[[r30:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c0]], %[[c32]]] : !xetile.tile<32x16xf16>
    //CHECK:   %[[r31:.*]] = xetile.update_tile_offset %[[arg6]], [%[[c32]], %[[c0]]] : !xetile.tile<32x16xf16>
    //CHECK:   %[[r32:.*]] = xetile.update_tile_offset %[[arg7]], [%[[c32]], %[[c0]]] : !xetile.tile<32x16xf16>
    //CHECK:   %[[r33:.*]] = arith.addf %[[arg8]], %[[arg8]] : vector<8x16xf32>
    //CHECK:   %[[r34:.*]] = arith.addf %[[arg9]], %[[arg9]] : vector<8x16xf32>
    //CHECK:   %[[r35:.*]] = arith.addf %[[arg10]], %[[arg10]] : vector<8x16xf32>
    //CHECK:   %[[r36:.*]] = arith.addf %[[arg11]], %[[arg11]] : vector<8x16xf32>
    //CHECK:   %[[r37:.*]] = arith.addf %[[arg12]], %[[arg12]] : vector<8x16xf32>
    //CHECK:   %[[r38:.*]] = arith.addf %[[arg13]], %[[arg13]] : vector<8x16xf32>
    //CHECK:   %[[r39:.*]] = arith.addf %[[arg14]], %[[arg14]] : vector<8x16xf32>
    //CHECK:   %[[r40:.*]] = arith.addf %[[arg15]], %[[arg15]] : vector<8x16xf32>
    //CHECK:   %[[r41:.*]] = xetile.tile_mma %[[r15]], %[[r25]], %[[r33]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r42:.*]] = xetile.tile_mma %[[r19]], %[[r26]], %[[r41]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r43:.*]] = xetile.tile_mma %[[r15]], %[[r27]], %[[r34]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r44:.*]] = xetile.tile_mma %[[r19]], %[[r28]], %[[r43]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r45:.*]] = xetile.tile_mma %[[r16]], %[[r25]], %[[r35]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r46:.*]] = xetile.tile_mma %[[r20]], %[[r26]], %[[r45]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r47:.*]] = xetile.tile_mma %[[r16]], %[[r27]], %[[r36]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r48:.*]] = xetile.tile_mma %[[r20]], %[[r28]], %[[r47]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r49:.*]] = xetile.tile_mma %[[r17]], %[[r25]], %[[r37]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r50:.*]] = xetile.tile_mma %[[r21]], %[[r26]], %[[r49]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r51:.*]] = xetile.tile_mma %[[r17]], %[[r27]], %[[r38]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r52:.*]] = xetile.tile_mma %[[r21]], %[[r28]], %[[r51]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r53:.*]] = xetile.tile_mma %[[r18]], %[[r25]], %[[r39]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r54:.*]] = xetile.tile_mma %[[r22]], %[[r26]], %[[r53]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r55:.*]] = xetile.tile_mma %[[r18]], %[[r27]], %[[r40]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r56:.*]] = xetile.tile_mma %[[r22]], %[[r28]], %[[r55]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   scf.yield %[[r29]], %[[r30]], %[[r31]], %[[r32]], %[[r42]], %[[r44]], %[[r46]], %[[r48]], %[[r50]], %[[r52]], %[[r54]], %[[r56]] : !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    //CHECK: }
    //CHECK: %[[r5:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r7:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r8:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r9:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r10:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r11:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r12:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#4,  %[[r5]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#5,  %[[r6]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#6,  %[[r7]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#7,  %[[r8]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#8,  %[[r9]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#9,  %[[r10]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#10,  %[[r11]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r4]]#11,  %[[r12]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant dense<0.0>: vector<32x32xf32>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]:  !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]:  !xetile.tile<32x32xf16>
      %5 = arith.addf %c_value, %c_value: vector<32x32xf32>
      %c_new_value = xetile.tile_mma %3, %4, %5: vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
  	gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_reduction_broadcast_dim_0
  //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_reduction_broadcast_dim_0(%a: memref<1024x1024xf16>) {

    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK-COUNT-32: %{{.*}} = vector.extract_strided_slice %[[r2]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-32: %{{.*}} = vector.extract_strided_slice %[[r3]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-62: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>

    //CHECK-COUNT-2: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<64xf16> to vector<1x64xf16>
    //CHECK: %{{.*}} = xetile.broadcast %{{.*}} [0] : vector<1x64xf16> -> vector<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    %3 = xetile.reduction <add>, %2 [0]: vector<32x64xf16> -> vector<1x64xf16>
    %4 = xetile.broadcast %3 [0]: vector<1x64xf16> -> vector<32x64xf16>
    %5 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    xetile.store_tile %4, %5: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }


  //CHECK-LABEL: gpu.func @sglevel_reduction_broadcast_dim_1
  //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_reduction_broadcast_dim_1(%a: memref<1024x1024xf16>) {

    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
    //CHECK: %[[c30_i32:.*]] = arith.constant 30 : i32
    //CHECK: %[[c29_i32:.*]] = arith.constant 29 : i32
    //CHECK: %[[c28_i32:.*]] = arith.constant 28 : i32
    //CHECK: %[[c27_i32:.*]] = arith.constant 27 : i32
    //CHECK: %[[c26_i32:.*]] = arith.constant 26 : i32
    //CHECK: %[[c25_i32:.*]] = arith.constant 25 : i32
    //CHECK: %[[c24_i32:.*]] = arith.constant 24 : i32
    //CHECK: %[[c23_i32:.*]] = arith.constant 23 : i32
    //CHECK: %[[c22_i32:.*]] = arith.constant 22 : i32
    //CHECK: %[[c21_i32:.*]] = arith.constant 21 : i32
    //CHECK: %[[c20_i32:.*]] = arith.constant 20 : i32
    //CHECK: %[[c19_i32:.*]] = arith.constant 19 : i32
    //CHECK: %[[c18_i32:.*]] = arith.constant 18 : i32
    //CHECK: %[[c17_i32:.*]] = arith.constant 17 : i32
    //CHECK: %[[c16_i32:.*]] = arith.constant 16 : i32
    //CHECK: %[[c15_i32:.*]] = arith.constant 15 : i32
    //CHECK: %[[c14_i32:.*]] = arith.constant 14 : i32
    //CHECK: %[[c13_i32:.*]] = arith.constant 13 : i32
    //CHECK: %[[c12_i32:.*]] = arith.constant 12 : i32
    //CHECK: %[[c11_i32:.*]] = arith.constant 11 : i32
    //CHECK: %[[c10_i32:.*]] = arith.constant 10 : i32
    //CHECK: %[[c9_i32:.*]] = arith.constant 9 : i32
    //CHECK: %[[c8_i32:.*]] = arith.constant 8 : i32
    //CHECK: %[[c7_i32:.*]] = arith.constant 7 : i32
    //CHECK: %[[c6_i32:.*]] = arith.constant 6 : i32
    //CHECK: %[[c5_i32:.*]] = arith.constant 5 : i32
    //CHECK: %[[c4_i32:.*]] = arith.constant 4 : i32
    //CHECK: %[[c3_i32:.*]] = arith.constant 3 : i32
    //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
    //CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
    //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r4:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r10:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r11:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r12:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r13:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r14:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r18:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r19:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r20:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [16, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r21:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [17, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r22:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [18, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r23:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [19, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r24:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [20, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [21, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [22, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [23, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r28:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [24, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r29:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [25, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r30:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [26, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r31:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [27, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r32:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [28, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r33:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [29, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r34:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [30, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r35:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [31, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r36:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r37:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r38:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r39:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r40:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r41:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r42:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r43:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r44:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r45:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r46:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r47:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r48:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r49:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r50:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r51:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r52:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [16, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r53:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [17, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r54:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [18, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r55:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [19, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r56:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [20, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r57:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [21, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r58:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [22, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r59:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [23, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r60:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [24, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r61:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [25, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r62:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [26, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r63:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [27, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r64:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [28, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r65:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [29, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r66:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [30, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r67:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [31, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r68:.*]] = arith.addf %[[r4]], %[[r36]] : vector<1x32xf16>
    //CHECK: %[[r69:.*]] = vector.shape_cast %[[r68]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r70:.*]] = arith.addf %[[r5]], %[[r37]] : vector<1x32xf16>
    //CHECK: %[[r71:.*]] = vector.shape_cast %[[r70]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r72:.*]] = arith.addf %[[r6]], %[[r38]] : vector<1x32xf16>
    //CHECK: %[[r73:.*]] = vector.shape_cast %[[r72]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r74:.*]] = arith.addf %[[r7]], %[[r39]] : vector<1x32xf16>
    //CHECK: %[[r75:.*]] = vector.shape_cast %[[r74]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r76:.*]] = arith.addf %[[r8]], %[[r40]] : vector<1x32xf16>
    //CHECK: %[[r77:.*]] = vector.shape_cast %[[r76]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r78:.*]] = arith.addf %[[r9]], %[[r41]] : vector<1x32xf16>
    //CHECK: %[[r79:.*]] = vector.shape_cast %[[r78]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r80:.*]] = arith.addf %[[r10]], %[[r42]] : vector<1x32xf16>
    //CHECK: %[[r81:.*]] = vector.shape_cast %[[r80]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r82:.*]] = arith.addf %[[r11]], %[[r43]] : vector<1x32xf16>
    //CHECK: %[[r83:.*]] = vector.shape_cast %[[r82]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r84:.*]] = arith.addf %[[r12]], %[[r44]] : vector<1x32xf16>
    //CHECK: %[[r85:.*]] = vector.shape_cast %[[r84]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r86:.*]] = arith.addf %[[r13]], %[[r45]] : vector<1x32xf16>
    //CHECK: %[[r87:.*]] = vector.shape_cast %[[r86]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r88:.*]] = arith.addf %[[r14]], %[[r46]] : vector<1x32xf16>
    //CHECK: %[[r89:.*]] = vector.shape_cast %[[r88]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r90:.*]] = arith.addf %[[r15]], %[[r47]] : vector<1x32xf16>
    //CHECK: %[[r91:.*]] = vector.shape_cast %[[r90]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r92:.*]] = arith.addf %[[r16]], %[[r48]] : vector<1x32xf16>
    //CHECK: %[[r93:.*]] = vector.shape_cast %[[r92]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r94:.*]] = arith.addf %[[r17]], %[[r49]] : vector<1x32xf16>
    //CHECK: %[[r95:.*]] = vector.shape_cast %[[r94]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r96:.*]] = arith.addf %[[r18]], %[[r50]] : vector<1x32xf16>
    //CHECK: %[[r97:.*]] = vector.shape_cast %[[r96]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r98:.*]] = arith.addf %[[r19]], %[[r51]] : vector<1x32xf16>
    //CHECK: %[[r99:.*]] = vector.shape_cast %[[r98]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r100:.*]] = arith.addf %[[r20]], %[[r52]] : vector<1x32xf16>
    //CHECK: %[[r101:.*]] = vector.shape_cast %[[r100]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r102:.*]] = arith.addf %[[r21]], %[[r53]] : vector<1x32xf16>
    //CHECK: %[[r103:.*]] = vector.shape_cast %[[r102]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r104:.*]] = arith.addf %[[r22]], %[[r54]] : vector<1x32xf16>
    //CHECK: %[[r105:.*]] = vector.shape_cast %[[r104]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r106:.*]] = arith.addf %[[r23]], %[[r55]] : vector<1x32xf16>
    //CHECK: %[[r107:.*]] = vector.shape_cast %[[r106]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r108:.*]] = arith.addf %[[r24]], %[[r56]] : vector<1x32xf16>
    //CHECK: %[[r109:.*]] = vector.shape_cast %[[r108]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r110:.*]] = arith.addf %[[r25]], %[[r57]] : vector<1x32xf16>
    //CHECK: %[[r111:.*]] = vector.shape_cast %[[r110]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r112:.*]] = arith.addf %[[r26]], %[[r58]] : vector<1x32xf16>
    //CHECK: %[[r113:.*]] = vector.shape_cast %[[r112]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r114:.*]] = arith.addf %[[r27]], %[[r59]] : vector<1x32xf16>
    //CHECK: %[[r115:.*]] = vector.shape_cast %[[r114]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r116:.*]] = arith.addf %[[r28]], %[[r60]] : vector<1x32xf16>
    //CHECK: %[[r117:.*]] = vector.shape_cast %[[r116]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r118:.*]] = arith.addf %[[r29]], %[[r61]] : vector<1x32xf16>
    //CHECK: %[[r119:.*]] = vector.shape_cast %[[r118]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r120:.*]] = arith.addf %[[r30]], %[[r62]] : vector<1x32xf16>
    //CHECK: %[[r121:.*]] = vector.shape_cast %[[r120]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r122:.*]] = arith.addf %[[r31]], %[[r63]] : vector<1x32xf16>
    //CHECK: %[[r123:.*]] = vector.shape_cast %[[r122]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r124:.*]] = arith.addf %[[r32]], %[[r64]] : vector<1x32xf16>
    //CHECK: %[[r125:.*]] = vector.shape_cast %[[r124]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r126:.*]] = arith.addf %[[r33]], %[[r65]] : vector<1x32xf16>
    //CHECK: %[[r127:.*]] = vector.shape_cast %[[r126]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r128:.*]] = arith.addf %[[r34]], %[[r66]] : vector<1x32xf16>
    //CHECK: %[[r129:.*]] = vector.shape_cast %[[r128]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r130:.*]] = arith.addf %[[r35]], %[[r67]] : vector<1x32xf16>
    //CHECK: %[[r131:.*]] = vector.shape_cast %[[r130]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r132:.*]] = vector.shuffle %[[r69]], %[[r71]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r133:.*]] = vector.shuffle %[[r69]], %[[r71]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r134:.*]] = arith.addf %[[r132]], %[[r133]] : vector<32xf16>
    //CHECK: %[[r135:.*]] = vector.shuffle %[[r73]], %[[r75]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r136:.*]] = vector.shuffle %[[r73]], %[[r75]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r137:.*]] = arith.addf %[[r135]], %[[r136]] : vector<32xf16>
    //CHECK: %[[r138:.*]] = vector.shuffle %[[r77]], %[[r79]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r139:.*]] = vector.shuffle %[[r77]], %[[r79]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r140:.*]] = arith.addf %[[r138]], %[[r139]] : vector<32xf16>
    //CHECK: %[[r141:.*]] = vector.shuffle %[[r81]], %[[r83]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r142:.*]] = vector.shuffle %[[r81]], %[[r83]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r143:.*]] = arith.addf %[[r141]], %[[r142]] : vector<32xf16>
    //CHECK: %[[r144:.*]] = vector.shuffle %[[r85]], %[[r87]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r145:.*]] = vector.shuffle %[[r85]], %[[r87]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r146:.*]] = arith.addf %[[r144]], %[[r145]] : vector<32xf16>
    //CHECK: %[[r147:.*]] = vector.shuffle %[[r89]], %[[r91]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r148:.*]] = vector.shuffle %[[r89]], %[[r91]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r149:.*]] = arith.addf %[[r147]], %[[r148]] : vector<32xf16>
    //CHECK: %[[r150:.*]] = vector.shuffle %[[r93]], %[[r95]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r151:.*]] = vector.shuffle %[[r93]], %[[r95]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r152:.*]] = arith.addf %[[r150]], %[[r151]] : vector<32xf16>
    //CHECK: %[[r153:.*]] = vector.shuffle %[[r97]], %[[r99]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r154:.*]] = vector.shuffle %[[r97]], %[[r99]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r155:.*]] = arith.addf %[[r153]], %[[r154]] : vector<32xf16>
    //CHECK: %[[r156:.*]] = vector.shuffle %[[r101]], %[[r103]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r157:.*]] = vector.shuffle %[[r101]], %[[r103]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r158:.*]] = arith.addf %[[r156]], %[[r157]] : vector<32xf16>
    //CHECK: %[[r159:.*]] = vector.shuffle %[[r105]], %[[r107]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r160:.*]] = vector.shuffle %[[r105]], %[[r107]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r161:.*]] = arith.addf %[[r159]], %[[r160]] : vector<32xf16>
    //CHECK: %[[r162:.*]] = vector.shuffle %[[r109]], %[[r111]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r163:.*]] = vector.shuffle %[[r109]], %[[r111]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r164:.*]] = arith.addf %[[r162]], %[[r163]] : vector<32xf16>
    //CHECK: %[[r165:.*]] = vector.shuffle %[[r113]], %[[r115]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r166:.*]] = vector.shuffle %[[r113]], %[[r115]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r167:.*]] = arith.addf %[[r165]], %[[r166]] : vector<32xf16>
    //CHECK: %[[r168:.*]] = vector.shuffle %[[r117]], %[[r119]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r169:.*]] = vector.shuffle %[[r117]], %[[r119]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r170:.*]] = arith.addf %[[r168]], %[[r169]] : vector<32xf16>
    //CHECK: %[[r171:.*]] = vector.shuffle %[[r121]], %[[r123]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r172:.*]] = vector.shuffle %[[r121]], %[[r123]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r173:.*]] = arith.addf %[[r171]], %[[r172]] : vector<32xf16>
    //CHECK: %[[r174:.*]] = vector.shuffle %[[r125]], %[[r127]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r175:.*]] = vector.shuffle %[[r125]], %[[r127]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r176:.*]] = arith.addf %[[r174]], %[[r175]] : vector<32xf16>
    //CHECK: %[[r177:.*]] = vector.shuffle %[[r129]], %[[r131]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r178:.*]] = vector.shuffle %[[r129]], %[[r131]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r179:.*]] = arith.addf %[[r177]], %[[r178]] : vector<32xf16>
    //CHECK: %[[r180:.*]] = vector.shuffle %[[r134]], %[[r137]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r181:.*]] = vector.shuffle %[[r134]], %[[r137]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r182:.*]] = arith.addf %[[r180]], %[[r181]] : vector<32xf16>
    //CHECK: %[[r183:.*]] = vector.shuffle %[[r140]], %[[r143]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r184:.*]] = vector.shuffle %[[r140]], %[[r143]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r185:.*]] = arith.addf %[[r183]], %[[r184]] : vector<32xf16>
    //CHECK: %[[r186:.*]] = vector.shuffle %[[r146]], %[[r149]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r187:.*]] = vector.shuffle %[[r146]], %[[r149]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r188:.*]] = arith.addf %[[r186]], %[[r187]] : vector<32xf16>
    //CHECK: %[[r189:.*]] = vector.shuffle %[[r152]], %[[r155]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r190:.*]] = vector.shuffle %[[r152]], %[[r155]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r191:.*]] = arith.addf %[[r189]], %[[r190]] : vector<32xf16>
    //CHECK: %[[r192:.*]] = vector.shuffle %[[r158]], %[[r161]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r193:.*]] = vector.shuffle %[[r158]], %[[r161]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r194:.*]] = arith.addf %[[r192]], %[[r193]] : vector<32xf16>
    //CHECK: %[[r195:.*]] = vector.shuffle %[[r164]], %[[r167]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r196:.*]] = vector.shuffle %[[r164]], %[[r167]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r197:.*]] = arith.addf %[[r195]], %[[r196]] : vector<32xf16>
    //CHECK: %[[r198:.*]] = vector.shuffle %[[r170]], %[[r173]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r199:.*]] = vector.shuffle %[[r170]], %[[r173]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r200:.*]] = arith.addf %[[r198]], %[[r199]] : vector<32xf16>
    //CHECK: %[[r201:.*]] = vector.shuffle %[[r176]], %[[r179]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r202:.*]] = vector.shuffle %[[r176]], %[[r179]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r203:.*]] = arith.addf %[[r201]], %[[r202]] : vector<32xf16>
    //CHECK: %[[r204:.*]] = vector.shuffle %[[r182]], %[[r185]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r205:.*]] = vector.shuffle %[[r182]], %[[r185]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r206:.*]] = arith.addf %[[r204]], %[[r205]] : vector<32xf16>
    //CHECK: %[[r207:.*]] = vector.shuffle %[[r188]], %[[r191]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r208:.*]] = vector.shuffle %[[r188]], %[[r191]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r209:.*]] = arith.addf %[[r207]], %[[r208]] : vector<32xf16>
    //CHECK: %[[r210:.*]] = vector.shuffle %[[r194]], %[[r197]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r211:.*]] = vector.shuffle %[[r194]], %[[r197]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r212:.*]] = arith.addf %[[r210]], %[[r211]] : vector<32xf16>
    //CHECK: %[[r213:.*]] = vector.shuffle %[[r200]], %[[r203]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r214:.*]] = vector.shuffle %[[r200]], %[[r203]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r215:.*]] = arith.addf %[[r213]], %[[r214]] : vector<32xf16>
    //CHECK: %[[r216:.*]] = vector.shuffle %[[r206]], %[[r209]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r217:.*]] = vector.shuffle %[[r206]], %[[r209]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r218:.*]] = arith.addf %[[r216]], %[[r217]] : vector<32xf16>
    //CHECK: %[[r219:.*]] = vector.shuffle %[[r212]], %[[r215]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r220:.*]] = vector.shuffle %[[r212]], %[[r215]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r221:.*]] = arith.addf %[[r219]], %[[r220]] : vector<32xf16>
    //CHECK: %[[r222:.*]] = vector.shuffle %[[r218]], %[[r221]] [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r223:.*]] = vector.shuffle %[[r218]], %[[r221]] [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r224:.*]] = arith.addf %[[r222]], %[[r223]] : vector<32xf16>
    //CHECK: %[[r225:.*]] = vector.extractelement %[[r224]][%[[c0_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r226:.*]] = vector.splat %[[r225]] : vector<1x1xf16>
    //CHECK: %[[r227:.*]] = vector.extractelement %[[r224]][%[[c1_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r228:.*]] = vector.splat %[[r227]] : vector<1x1xf16>
    //CHECK: %[[r229:.*]] = vector.extractelement %224[%c2_i32 : i32] : vector<32xf16>
    //CHECK: %[[r230:.*]] = vector.splat %[[r229]] : vector<1x1xf16>
    //CHECK: %[[r231:.*]] = vector.extractelement %[[r224]][%[[c3_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r232:.*]] = vector.splat %[[r231]] : vector<1x1xf16>
    //CHECK: %[[r233:.*]] = vector.extractelement %[[r224]][%[[c4_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r234:.*]] = vector.splat %[[r233]] : vector<1x1xf16>
    //CHECK: %[[r235:.*]] = vector.extractelement %[[r224]][%[[c5_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r236:.*]] = vector.splat %[[r235]] : vector<1x1xf16>
    //CHECK: %[[r237:.*]] = vector.extractelement %[[r224]][%[[c6_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r238:.*]] = vector.splat %[[r237]] : vector<1x1xf16>
    //CHECK: %[[r239:.*]] = vector.extractelement %[[r224]][%[[c7_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r240:.*]] = vector.splat %[[r239]] : vector<1x1xf16>
    //CHECK: %[[r241:.*]] = vector.extractelement %[[r224]][%[[c8_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r242:.*]] = vector.splat %[[r241]] : vector<1x1xf16>
    //CHECK: %[[r243:.*]] = vector.extractelement %[[r224]][%[[c9_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r244:.*]] = vector.splat %[[r243]] : vector<1x1xf16>
    //CHECK: %[[r245:.*]] = vector.extractelement %[[r224]][%[[c10_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r246:.*]] = vector.splat %[[r245]] : vector<1x1xf16>
    //CHECK: %[[r247:.*]] = vector.extractelement %[[r224]][%[[c11_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r248:.*]] = vector.splat %[[r247]] : vector<1x1xf16>
    //CHECK: %[[r249:.*]] = vector.extractelement %[[r224]][%[[c12_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r250:.*]] = vector.splat %[[r249]] : vector<1x1xf16>
    //CHECK: %[[r251:.*]] = vector.extractelement %[[r224]][%[[c13_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r252:.*]] = vector.splat %[[r251]] : vector<1x1xf16>
    //CHECK: %[[r253:.*]] = vector.extractelement %[[r224]][%[[c14_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r254:.*]] = vector.splat %[[r253]] : vector<1x1xf16>
    //CHECK: %[[r255:.*]] = vector.extractelement %[[r224]][%[[c15_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r256:.*]] = vector.splat %[[r255]] : vector<1x1xf16>
    //CHECK: %[[r257:.*]] = vector.extractelement %[[r224]][%[[c16_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r258:.*]] = vector.splat %[[r257]] : vector<1x1xf16>
    //CHECK: %[[r259:.*]] = vector.extractelement %[[r224]][%[[c17_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r260:.*]] = vector.splat %[[r259]] : vector<1x1xf16>
    //CHECK: %[[r261:.*]] = vector.extractelement %[[r224]][%[[c18_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r262:.*]] = vector.splat %[[r261]] : vector<1x1xf16>
    //CHECK: %[[r263:.*]] = vector.extractelement %[[r224]][%[[c19_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r264:.*]] = vector.splat %[[r263]] : vector<1x1xf16>
    //CHECK: %[[r265:.*]] = vector.extractelement %[[r224]][%[[c20_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r266:.*]] = vector.splat %[[r265]] : vector<1x1xf16>
    //CHECK: %[[r267:.*]] = vector.extractelement %[[r224]][%[[c21_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r268:.*]] = vector.splat %[[r267]] : vector<1x1xf16>
    //CHECK: %[[r269:.*]] = vector.extractelement %[[r224]][%[[c22_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r270:.*]] = vector.splat %[[r269]] : vector<1x1xf16>
    //CHECK: %[[r271:.*]] = vector.extractelement %[[r224]][%[[c23_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r272:.*]] = vector.splat %[[r271]] : vector<1x1xf16>
    //CHECK: %[[r273:.*]] = vector.extractelement %[[r224]][%[[c24_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r274:.*]] = vector.splat %[[r273]] : vector<1x1xf16>
    //CHECK: %[[r275:.*]] = vector.extractelement %[[r224]][%[[c25_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r276:.*]] = vector.splat %[[r275]] : vector<1x1xf16>
    //CHECK: %[[r277:.*]] = vector.extractelement %[[r224]][%[[c26_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r278:.*]] = vector.splat %[[r277]] : vector<1x1xf16>
    //CHECK: %[[r279:.*]] = vector.extractelement %[[r224]][%[[c27_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r280:.*]] = vector.splat %[[r279]] : vector<1x1xf16>
    //CHECK: %[[r281:.*]] = vector.extractelement %[[r224]][%c28_i32 : i32] : vector<32xf16>
    //CHECK: %[[r282:.*]] = vector.splat %[[r281]] : vector<1x1xf16>
    //CHECK: %[[r283:.*]] = vector.extractelement %[[r224]][%[[c29_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r284:.*]] = vector.splat %[[r283]] : vector<1x1xf16>
    //CHECK: %[[r285:.*]] = vector.extractelement %[[r224]][%[[c30_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r286:.*]] = vector.splat %[[r285]] : vector<1x1xf16>
    //CHECK: %[[r287:.*]] = vector.extractelement %[[r224]][%[[c31_i32]] : i32] : vector<32xf16>
    //CHECK: %[[r288:.*]] = vector.splat %[[r287]] : vector<1x1xf16>
    //CHECK: %[[r289:.*]] = vector.shuffle %[[r226]], %[[r228]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r290:.*]] = vector.shuffle %[[r230]], %[[r232]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r291:.*]] = vector.shuffle %[[r234]], %[[r236]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r292:.*]] = vector.shuffle %[[r238]], %[[r240]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r293:.*]] = vector.shuffle %[[r242]], %[[r244]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r294:.*]] = vector.shuffle %[[r246]], %[[r248]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r295:.*]] = vector.shuffle %[[r250]], %[[r252]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r296:.*]] = vector.shuffle %[[r254]], %[[r256]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r297:.*]] = vector.shuffle %[[r258]], %[[r260]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r298:.*]] = vector.shuffle %[[r262]], %[[r264]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r299:.*]] = vector.shuffle %[[r266]], %[[r268]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r300:.*]] = vector.shuffle %[[r270]], %[[r272]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r301:.*]] = vector.shuffle %[[r274]], %[[r276]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r302:.*]] = vector.shuffle %[[r278]], %[[r280]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r303:.*]] = vector.shuffle %[[r282]], %[[r284]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r304:.*]] = vector.shuffle %[[r286]], %[[r288]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r305:.*]] = vector.shuffle %[[r289]], %[[r290]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r306:.*]] = vector.shuffle %[[r291]], %[[r292]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r307:.*]] = vector.shuffle %[[r293]], %[[r294]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r308:.*]] = vector.shuffle %[[r295]], %[[r296]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r309:.*]] = vector.shuffle %[[r297]], %[[r298]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r310:.*]] = vector.shuffle %[[r299]], %[[r300]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r311:.*]] = vector.shuffle %[[r301]], %[[r302]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r312:.*]] = vector.shuffle %[[r303]], %[[r304]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r313:.*]] = vector.shuffle %[[r305]], %[[r306]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r314:.*]] = vector.shuffle %[[r307]], %[[r308]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r315:.*]] = vector.shuffle %[[r309]], %[[r310]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r316:.*]] = vector.shuffle %[[r311]], %[[r312]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r317:.*]] = vector.shuffle %[[r313]], %[[r314]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
    //CHECK: %[[r318:.*]] = vector.shuffle %[[r315]], %[[r316]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
    //CHECK: %[[r319:.*]] = vector.shuffle %[[r317]], %[[r318]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x1xf16>, vector<16x1xf16>
    //CHECK: %[[r320:.*]] = xetile.broadcast %[[r319]] [1] : vector<32x1xf16> -> vector<32x64xf16>
    //CHECK: %[[r321:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r322:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r323:.*]] = xetile.init_tile %[[arg0]][%[[c8]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r324:.*]] = xetile.init_tile %[[arg0]][%[[c8]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r325:.*]] = xetile.init_tile %[[arg0]][%[[c16]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r326:.*]] = xetile.init_tile %[[arg0]][%[[c16]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r327:.*]] = xetile.init_tile %[[arg0]][%[[c24]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r328:.*]] = xetile.init_tile %[[arg0]][%[[c24]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r329:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 0], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK: %[[r330:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [8, 0], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK: %[[r331:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [16, 0], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK: %[[r332:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [24, 0], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK: %[[r333:.*]] = vector.extract_strided_slice %[[r329]] {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: %[[r334:.*]] = vector.extract_strided_slice %[[r329]] {offsets = [0, 32], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: %[[r335:.*]] = vector.extract_strided_slice %[[r330]] {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: %[[r336:.*]] = vector.extract_strided_slice %[[r330]] {offsets = [0, 32], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: %[[r337:.*]] = vector.extract_strided_slice %[[r331]] {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: %[[r338:.*]] = vector.extract_strided_slice %[[r331]] {offsets = [0, 32], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: %[[r339:.*]] = vector.extract_strided_slice %[[r332]] {offsets = [0, 0], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: %[[r340:.*]] = vector.extract_strided_slice %[[r332]] {offsets = [0, 32], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK: xetile.store_tile %[[r333]],  %[[r321]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r334]],  %[[r322]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r335]],  %[[r323]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r336]],  %[[r324]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r337]],  %[[r325]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r338]],  %[[r326]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r339]],  %[[r327]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r340]],  %[[r328]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    %3 = xetile.reduction <add>, %2 [1]: vector<32x64xf16> -> vector<32x1xf16>
    %4 = xetile.broadcast %3 [1]: vector<32x1xf16> -> vector<32x64xf16>
    %5 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    xetile.store_tile %4, %5: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }


  //CHECK-LABEL: gpu.func @sglevel_reduction_broadcast_transpose
  //CHECK-SAME(%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_reduction_broadcast_transpose(%a: memref<1024x1024xf16>) {
    //CHECK: %[[c56:.*]] = arith.constant 56 : index
    //CHECK: %[[c48:.*]] = arith.constant 48 : index
    //CHECK: %[[c40:.*]] = arith.constant 40 : index
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
    //CHECK: %[[c30_i32:.*]] = arith.constant 30 : i32
    //CHECK: %[[c29_i32:.*]] = arith.constant 29 : i32
    //CHECK: %[[c28_i32:.*]] = arith.constant 28 : i32
    //CHECK: %[[c27_i32:.*]] = arith.constant 27 : i32
    //CHECK: %[[c26_i32:.*]] = arith.constant 26 : i32
    //CHECK: %[[c25_i32:.*]] = arith.constant 25 : i32
    //CHECK: %[[c24_i32:.*]] = arith.constant 24 : i32
    //CHECK: %[[c23_i32:.*]] = arith.constant 23 : i32
    //CHECK: %[[c22_i32:.*]] = arith.constant 22 : i32
    //CHECK: %[[c21_i32:.*]] = arith.constant 21 : i32
    //CHECK: %[[c20_i32:.*]] = arith.constant 20 : i32
    //CHECK: %[[c19_i32:.*]] = arith.constant 19 : i32
    //CHECK: %[[c18_i32:.*]] = arith.constant 18 : i32
    //CHECK: %[[c17_i32:.*]] = arith.constant 17 : i32
    //CHECK: %[[c16_i32:.*]] = arith.constant 16 : i32
    //CHECK: %[[c15_i32:.*]] = arith.constant 15 : i32
    //CHECK: %[[c14_i32:.*]] = arith.constant 14 : i32
    //CHECK: %[[c13_i32:.*]] = arith.constant 13 : i32
    //CHECK: %[[c12_i32:.*]] = arith.constant 12 : i32
    //CHECK: %[[c11_i32:.*]] = arith.constant 11 : i32
    //CHECK: %[[c10_i32:.*]] = arith.constant 10 : i32
    //CHECK: %[[c9_i32:.*]] = arith.constant 9 : i32
    //CHECK: %[[c8_i32:.*]] = arith.constant 8 : i32
    //CHECK: %[[c7_i32:.*]] = arith.constant 7 : i32
    //CHECK: %[[c6_i32:.*]] = arith.constant 6 : i32
    //CHECK: %[[c5_i32:.*]] = arith.constant 5 : i32
    //CHECK: %[[c4_i32:.*]] = arith.constant 4 : i32
    //CHECK: %[[c3_i32:.*]] = arith.constant 3 : i32
    //CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
    //CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
    //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r4:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r10:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r11:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r12:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r13:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r14:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r18:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r19:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r20:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [16, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r21:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [17, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r22:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [18, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r23:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [19, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r24:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [20, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [21, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [22, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [23, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r28:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [24, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r29:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [25, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r30:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [26, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r31:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [27, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r32:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [28, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r33:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [29, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r34:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [30, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r35:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [31, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r36:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r37:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r38:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r39:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r40:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r41:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r42:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r43:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r44:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r45:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r46:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r47:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r48:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r49:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r50:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r51:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r52:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [16, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r53:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [17, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r54:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [18, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r55:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [19, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r56:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [20, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r57:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [21, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r58:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [22, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r59:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [23, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r60:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [24, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r61:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [25, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r62:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [26, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r63:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [27, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r64:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [28, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r65:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [29, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r66:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [30, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r67:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [31, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r68:.*]] = arith.addf %4, %36 : vector<1x32xf16>
    //CHECK: %[[r69:.*]] = vector.shape_cast %68 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r70:.*]] = arith.addf %5, %37 : vector<1x32xf16>
    //CHECK: %[[r71:.*]] = vector.shape_cast %70 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r72:.*]] = arith.addf %6, %38 : vector<1x32xf16>
    //CHECK: %[[r73:.*]] = vector.shape_cast %72 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r74:.*]] = arith.addf %7, %39 : vector<1x32xf16>
    //CHECK: %[[r75:.*]] = vector.shape_cast %74 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r76:.*]] = arith.addf %8, %40 : vector<1x32xf16>
    //CHECK: %[[r77:.*]] = vector.shape_cast %76 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r78:.*]] = arith.addf %9, %41 : vector<1x32xf16>
    //CHECK: %[[r79:.*]] = vector.shape_cast %78 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r80:.*]] = arith.addf %10, %42 : vector<1x32xf16>
    //CHECK: %[[r81:.*]] = vector.shape_cast %80 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r82:.*]] = arith.addf %11, %43 : vector<1x32xf16>
    //CHECK: %[[r83:.*]] = vector.shape_cast %82 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r84:.*]] = arith.addf %12, %44 : vector<1x32xf16>
    //CHECK: %[[r85:.*]] = vector.shape_cast %84 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r86:.*]] = arith.addf %13, %45 : vector<1x32xf16>
    //CHECK: %[[r87:.*]] = vector.shape_cast %86 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r88:.*]] = arith.addf %14, %46 : vector<1x32xf16>
    //CHECK: %[[r89:.*]] = vector.shape_cast %88 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r90:.*]] = arith.addf %15, %47 : vector<1x32xf16>
    //CHECK: %[[r91:.*]] = vector.shape_cast %90 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r92:.*]] = arith.addf %16, %48 : vector<1x32xf16>
    //CHECK: %[[r93:.*]] = vector.shape_cast %92 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r94:.*]] = arith.addf %17, %49 : vector<1x32xf16>
    //CHECK: %[[r95:.*]] = vector.shape_cast %94 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r96:.*]] = arith.addf %18, %50 : vector<1x32xf16>
    //CHECK: %[[r97:.*]] = vector.shape_cast %96 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r98:.*]] = arith.addf %19, %51 : vector<1x32xf16>
    //CHECK: %[[r99:.*]] = vector.shape_cast %98 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r100:.*]] = arith.addf %20, %52 : vector<1x32xf16>
    //CHECK: %[[r101:.*]] = vector.shape_cast %100 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r102:.*]] = arith.addf %21, %53 : vector<1x32xf16>
    //CHECK: %[[r103:.*]] = vector.shape_cast %102 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r104:.*]] = arith.addf %22, %54 : vector<1x32xf16>
    //CHECK: %[[r105:.*]] = vector.shape_cast %104 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r106:.*]] = arith.addf %23, %55 : vector<1x32xf16>
    //CHECK: %[[r107:.*]] = vector.shape_cast %106 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r108:.*]] = arith.addf %24, %56 : vector<1x32xf16>
    //CHECK: %[[r109:.*]] = vector.shape_cast %108 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r110:.*]] = arith.addf %25, %57 : vector<1x32xf16>
    //CHECK: %[[r111:.*]] = vector.shape_cast %110 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r112:.*]] = arith.addf %26, %58 : vector<1x32xf16>
    //CHECK: %[[r113:.*]] = vector.shape_cast %112 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r114:.*]] = arith.addf %27, %59 : vector<1x32xf16>
    //CHECK: %[[r115:.*]] = vector.shape_cast %114 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r116:.*]] = arith.addf %28, %60 : vector<1x32xf16>
    //CHECK: %[[r117:.*]] = vector.shape_cast %116 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r118:.*]] = arith.addf %29, %61 : vector<1x32xf16>
    //CHECK: %[[r119:.*]] = vector.shape_cast %118 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r120:.*]] = arith.addf %30, %62 : vector<1x32xf16>
    //CHECK: %[[r121:.*]] = vector.shape_cast %120 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r122:.*]] = arith.addf %31, %63 : vector<1x32xf16>
    //CHECK: %[[r123:.*]] = vector.shape_cast %122 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r124:.*]] = arith.addf %32, %64 : vector<1x32xf16>
    //CHECK: %[[r125:.*]] = vector.shape_cast %124 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r126:.*]] = arith.addf %33, %65 : vector<1x32xf16>
    //CHECK: %[[r127:.*]] = vector.shape_cast %126 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r128:.*]] = arith.addf %34, %66 : vector<1x32xf16>
    //CHECK: %[[r129:.*]] = vector.shape_cast %128 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r130:.*]] = arith.addf %35, %67 : vector<1x32xf16>
    //CHECK: %[[r131:.*]] = vector.shape_cast %130 : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r132:.*]] = vector.shuffle %[[r69]], %[[r71]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r133:.*]] = vector.shuffle %[[r69]], %[[r71]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r134:.*]] = arith.addf %132, %133 : vector<32xf16>
    //CHECK: %[[r135:.*]] = vector.shuffle %[[r73]], %[[r75]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r136:.*]] = vector.shuffle %[[r73]], %[[r75]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r137:.*]] = arith.addf %135, %136 : vector<32xf16>
    //CHECK: %[[r138:.*]] = vector.shuffle %[[r77]], %[[r79]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r139:.*]] = vector.shuffle %[[r77]], %[[r79]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r140:.*]] = arith.addf %138, %139 : vector<32xf16>
    //CHECK: %[[r141:.*]] = vector.shuffle %[[r81]], %[[r83]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r142:.*]] = vector.shuffle %[[r81]], %[[r83]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r143:.*]] = arith.addf %141, %142 : vector<32xf16>
    //CHECK: %[[r144:.*]] = vector.shuffle %[[r85]], %[[r87]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r145:.*]] = vector.shuffle %[[r85]], %[[r87]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r146:.*]] = arith.addf %144, %145 : vector<32xf16>
    //CHECK: %[[r147:.*]] = vector.shuffle %[[r89]], %[[r91]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r148:.*]] = vector.shuffle %[[r89]], %[[r91]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r149:.*]] = arith.addf %147, %148 : vector<32xf16>
    //CHECK: %[[r150:.*]] = vector.shuffle %[[r93]], %[[r95]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r151:.*]] = vector.shuffle %[[r93]], %[[r95]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r152:.*]] = arith.addf %150, %151 : vector<32xf16>
    //CHECK: %[[r153:.*]] = vector.shuffle %[[r97]], %[[r99]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r154:.*]] = vector.shuffle %[[r97]], %[[r99]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r155:.*]] = arith.addf %153, %154 : vector<32xf16>
    //CHECK: %[[r156:.*]] = vector.shuffle %[[r101]], %[[r103]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r157:.*]] = vector.shuffle %[[r101]], %[[r103]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r158:.*]] = arith.addf %156, %157 : vector<32xf16>
    //CHECK: %[[r159:.*]] = vector.shuffle %[[r105]], %[[r107]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r160:.*]] = vector.shuffle %[[r105]], %[[r107]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r161:.*]] = arith.addf %159, %160 : vector<32xf16>
    //CHECK: %[[r162:.*]] = vector.shuffle %[[r109]], %[[r111]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r163:.*]] = vector.shuffle %[[r109]], %[[r111]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r164:.*]] = arith.addf %162, %163 : vector<32xf16>
    //CHECK: %[[r165:.*]] = vector.shuffle %[[r113]], %[[r115]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r166:.*]] = vector.shuffle %[[r113]], %[[r115]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r167:.*]] = arith.addf %165, %166 : vector<32xf16>
    //CHECK: %[[r168:.*]] = vector.shuffle %[[r117]], %[[r119]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r169:.*]] = vector.shuffle %[[r117]], %[[r119]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r170:.*]] = arith.addf %168, %169 : vector<32xf16>
    //CHECK: %[[r171:.*]] = vector.shuffle %[[r121]], %[[r123]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r172:.*]] = vector.shuffle %[[r121]], %[[r123]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r173:.*]] = arith.addf %171, %172 : vector<32xf16>
    //CHECK: %[[r174:.*]] = vector.shuffle %[[r125]], %[[r127]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r175:.*]] = vector.shuffle %[[r125]], %[[r127]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r176:.*]] = arith.addf %174, %175 : vector<32xf16>
    //CHECK: %[[r177:.*]] = vector.shuffle %[[r129]], %[[r131]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r178:.*]] = vector.shuffle %[[r129]], %[[r131]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r179:.*]] = arith.addf %177, %178 : vector<32xf16>
    //CHECK: %[[r180:.*]] = vector.shuffle %[[r134]], %[[r137]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r181:.*]] = vector.shuffle %[[r134]], %[[r137]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r182:.*]] = arith.addf %180, %181 : vector<32xf16>
    //CHECK: %[[r183:.*]] = vector.shuffle %[[r140]], %[[r143]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r184:.*]] = vector.shuffle %[[r140]], %[[r143]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r185:.*]] = arith.addf %183, %184 : vector<32xf16>
    //CHECK: %[[r186:.*]] = vector.shuffle %[[r146]], %[[r149]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r187:.*]] = vector.shuffle %[[r146]], %[[r149]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r188:.*]] = arith.addf %186, %187 : vector<32xf16>
    //CHECK: %[[r189:.*]] = vector.shuffle %[[r152]], %[[r155]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r190:.*]] = vector.shuffle %[[r152]], %[[r155]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r191:.*]] = arith.addf %189, %190 : vector<32xf16>
    //CHECK: %[[r192:.*]] = vector.shuffle %[[r158]], %[[r161]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r193:.*]] = vector.shuffle %[[r158]], %[[r161]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r194:.*]] = arith.addf %192, %193 : vector<32xf16>
    //CHECK: %[[r195:.*]] = vector.shuffle %[[r164]], %[[r167]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r196:.*]] = vector.shuffle %[[r164]], %[[r167]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r197:.*]] = arith.addf %195, %196 : vector<32xf16>
    //CHECK: %[[r198:.*]] = vector.shuffle %[[r170]], %[[r173]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r199:.*]] = vector.shuffle %[[r170]], %[[r173]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r200:.*]] = arith.addf %198, %199 : vector<32xf16>
    //CHECK: %[[r201:.*]] = vector.shuffle %[[r176]], %[[r179]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r202:.*]] = vector.shuffle %[[r176]], %[[r179]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r203:.*]] = arith.addf %201, %202 : vector<32xf16>
    //CHECK: %[[r204:.*]] = vector.shuffle %[[r182]], %[[r185]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r205:.*]] = vector.shuffle %[[r182]], %[[r185]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r206:.*]] = arith.addf %204, %205 : vector<32xf16>
    //CHECK: %[[r207:.*]] = vector.shuffle %[[r188]], %[[r191]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r208:.*]] = vector.shuffle %[[r188]], %[[r191]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r209:.*]] = arith.addf %207, %208 : vector<32xf16>
    //CHECK: %[[r210:.*]] = vector.shuffle %[[r194]], %[[r197]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r211:.*]] = vector.shuffle %[[r194]], %[[r197]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r212:.*]] = arith.addf %210, %211 : vector<32xf16>
    //CHECK: %[[r213:.*]] = vector.shuffle %[[r200]], %[[r203]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r214:.*]] = vector.shuffle %[[r200]], %[[r203]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r215:.*]] = arith.addf %213, %214 : vector<32xf16>
    //CHECK: %[[r216:.*]] = vector.shuffle %[[r206]], %[[r209]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r217:.*]] = vector.shuffle %[[r206]], %[[r209]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r218:.*]] = arith.addf %216, %217 : vector<32xf16>
    //CHECK: %[[r219:.*]] = vector.shuffle %[[r212]], %[[r215]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r220:.*]] = vector.shuffle %[[r212]], %[[r215]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r221:.*]] = arith.addf %219, %220 : vector<32xf16>
    //CHECK: %[[r222:.*]] = vector.shuffle %[[r218]], %[[r221]] [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r223:.*]] = vector.shuffle %[[r218]], %[[r221]] [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %[[r224:.*]] = arith.addf %222, %223 : vector<32xf16>
    //CHECK: %[[r225:.*]] = vector.extractelement %224[%c0_i32 : i32] : vector<32xf16>
    //CHECK: %[[r226:.*]] = vector.splat %225 : vector<1x1xf16>
    //CHECK: %[[r227:.*]] = vector.extractelement %224[%c1_i32 : i32] : vector<32xf16>
    //CHECK: %[[r228:.*]] = vector.splat %227 : vector<1x1xf16>
    //CHECK: %[[r229:.*]] = vector.extractelement %224[%c2_i32 : i32] : vector<32xf16>
    //CHECK: %[[r230:.*]] = vector.splat %229 : vector<1x1xf16>
    //CHECK: %[[r231:.*]] = vector.extractelement %224[%c3_i32 : i32] : vector<32xf16>
    //CHECK: %[[r232:.*]] = vector.splat %231 : vector<1x1xf16>
    //CHECK: %[[r233:.*]] = vector.extractelement %224[%c4_i32 : i32] : vector<32xf16>
    //CHECK: %[[r234:.*]] = vector.splat %233 : vector<1x1xf16>
    //CHECK: %[[r235:.*]] = vector.extractelement %224[%c5_i32 : i32] : vector<32xf16>
    //CHECK: %[[r236:.*]] = vector.splat %235 : vector<1x1xf16>
    //CHECK: %[[r237:.*]] = vector.extractelement %224[%c6_i32 : i32] : vector<32xf16>
    //CHECK: %[[r238:.*]] = vector.splat %237 : vector<1x1xf16>
    //CHECK: %[[r239:.*]] = vector.extractelement %224[%c7_i32 : i32] : vector<32xf16>
    //CHECK: %[[r240:.*]] = vector.splat %239 : vector<1x1xf16>
    //CHECK: %[[r241:.*]] = vector.extractelement %224[%c8_i32 : i32] : vector<32xf16>
    //CHECK: %[[r242:.*]] = vector.splat %241 : vector<1x1xf16>
    //CHECK: %[[r243:.*]] = vector.extractelement %224[%c9_i32 : i32] : vector<32xf16>
    //CHECK: %[[r244:.*]] = vector.splat %243 : vector<1x1xf16>
    //CHECK: %[[r245:.*]] = vector.extractelement %224[%c10_i32 : i32] : vector<32xf16>
    //CHECK: %[[r246:.*]] = vector.splat %245 : vector<1x1xf16>
    //CHECK: %[[r247:.*]] = vector.extractelement %224[%c11_i32 : i32] : vector<32xf16>
    //CHECK: %[[r248:.*]] = vector.splat %247 : vector<1x1xf16>
    //CHECK: %[[r249:.*]] = vector.extractelement %224[%c12_i32 : i32] : vector<32xf16>
    //CHECK: %[[r250:.*]] = vector.splat %249 : vector<1x1xf16>
    //CHECK: %[[r251:.*]] = vector.extractelement %224[%c13_i32 : i32] : vector<32xf16>
    //CHECK: %[[r252:.*]] = vector.splat %251 : vector<1x1xf16>
    //CHECK: %[[r253:.*]] = vector.extractelement %224[%c14_i32 : i32] : vector<32xf16>
    //CHECK: %[[r254:.*]] = vector.splat %253 : vector<1x1xf16>
    //CHECK: %[[r255:.*]] = vector.extractelement %224[%c15_i32 : i32] : vector<32xf16>
    //CHECK: %[[r256:.*]] = vector.splat %255 : vector<1x1xf16>
    //CHECK: %[[r257:.*]] = vector.extractelement %224[%c16_i32 : i32] : vector<32xf16>
    //CHECK: %[[r258:.*]] = vector.splat %257 : vector<1x1xf16>
    //CHECK: %[[r259:.*]] = vector.extractelement %224[%c17_i32 : i32] : vector<32xf16>
    //CHECK: %[[r260:.*]] = vector.splat %259 : vector<1x1xf16>
    //CHECK: %[[r261:.*]] = vector.extractelement %224[%c18_i32 : i32] : vector<32xf16>
    //CHECK: %[[r262:.*]] = vector.splat %261 : vector<1x1xf16>
    //CHECK: %[[r263:.*]] = vector.extractelement %224[%c19_i32 : i32] : vector<32xf16>
    //CHECK: %[[r264:.*]] = vector.splat %263 : vector<1x1xf16>
    //CHECK: %[[r265:.*]] = vector.extractelement %224[%c20_i32 : i32] : vector<32xf16>
    //CHECK: %[[r266:.*]] = vector.splat %265 : vector<1x1xf16>
    //CHECK: %[[r267:.*]] = vector.extractelement %224[%c21_i32 : i32] : vector<32xf16>
    //CHECK: %[[r268:.*]] = vector.splat %267 : vector<1x1xf16>
    //CHECK: %[[r269:.*]] = vector.extractelement %224[%c22_i32 : i32] : vector<32xf16>
    //CHECK: %[[r270:.*]] = vector.splat %269 : vector<1x1xf16>
    //CHECK: %[[r271:.*]] = vector.extractelement %224[%c23_i32 : i32] : vector<32xf16>
    //CHECK: %[[r272:.*]] = vector.splat %271 : vector<1x1xf16>
    //CHECK: %[[r273:.*]] = vector.extractelement %224[%c24_i32 : i32] : vector<32xf16>
    //CHECK: %[[r274:.*]] = vector.splat %273 : vector<1x1xf16>
    //CHECK: %[[r275:.*]] = vector.extractelement %224[%c25_i32 : i32] : vector<32xf16>
    //CHECK: %[[r276:.*]] = vector.splat %275 : vector<1x1xf16>
    //CHECK: %[[r277:.*]] = vector.extractelement %224[%c26_i32 : i32] : vector<32xf16>
    //CHECK: %[[r278:.*]] = vector.splat %277 : vector<1x1xf16>
    //CHECK: %[[r279:.*]] = vector.extractelement %224[%c27_i32 : i32] : vector<32xf16>
    //CHECK: %[[r280:.*]] = vector.splat %279 : vector<1x1xf16>
    //CHECK: %[[r281:.*]] = vector.extractelement %224[%c28_i32 : i32] : vector<32xf16>
    //CHECK: %[[r282:.*]] = vector.splat %281 : vector<1x1xf16>
    //CHECK: %[[r283:.*]] = vector.extractelement %224[%c29_i32 : i32] : vector<32xf16>
    //CHECK: %[[r284:.*]] = vector.splat %283 : vector<1x1xf16>
    //CHECK: %[[r285:.*]] = vector.extractelement %224[%c30_i32 : i32] : vector<32xf16>
    //CHECK: %[[r286:.*]] = vector.splat %285 : vector<1x1xf16>
    //CHECK: %[[r287:.*]] = vector.extractelement %224[%c31_i32 : i32] : vector<32xf16>
    //CHECK: %[[r288:.*]] = vector.splat %287 : vector<1x1xf16>
    //CHECK: %[[r289:.*]] = vector.shuffle %[[r226]], %[[r228]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r290:.*]] = vector.shuffle %[[r230]], %[[r232]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r291:.*]] = vector.shuffle %[[r234]], %[[r236]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r292:.*]] = vector.shuffle %[[r238]], %[[r240]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r293:.*]] = vector.shuffle %[[r242]], %[[r244]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r294:.*]] = vector.shuffle %[[r246]], %[[r248]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r295:.*]] = vector.shuffle %[[r250]], %[[r252]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r296:.*]] = vector.shuffle %[[r254]], %[[r256]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r297:.*]] = vector.shuffle %[[r258]], %[[r260]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r298:.*]] = vector.shuffle %[[r262]], %[[r264]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r299:.*]] = vector.shuffle %[[r266]], %[[r268]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r300:.*]] = vector.shuffle %[[r270]], %[[r272]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r301:.*]] = vector.shuffle %[[r274]], %[[r276]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r302:.*]] = vector.shuffle %[[r278]], %[[r280]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r303:.*]] = vector.shuffle %[[r282]], %[[r284]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r304:.*]] = vector.shuffle %[[r286]], %[[r288]] [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK: %[[r305:.*]] = vector.shuffle %[[r289]], %[[r290]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r306:.*]] = vector.shuffle %[[r291]], %[[r292]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r307:.*]] = vector.shuffle %[[r293]], %[[r294]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r308:.*]] = vector.shuffle %[[r295]], %[[r296]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r309:.*]] = vector.shuffle %[[r297]], %[[r298]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r310:.*]] = vector.shuffle %[[r299]], %[[r300]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r311:.*]] = vector.shuffle %[[r301]], %[[r302]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r312:.*]] = vector.shuffle %[[r303]], %[[r304]] [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK: %[[r313:.*]] = vector.shuffle %[[r305]], %[[r306]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r314:.*]] = vector.shuffle %[[r307]], %[[r308]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r315:.*]] = vector.shuffle %[[r309]], %[[r310]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r316:.*]] = vector.shuffle %[[r311]], %[[r312]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK: %[[r317:.*]] = vector.shuffle %[[r313]], %[[r314]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
    //CHECK: %[[r318:.*]] = vector.shuffle %[[r315]], %[[r316]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
    //CHECK: %[[r319:.*]] = vector.shuffle %[[r317]], %[[r318]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x1xf16>, vector<16x1xf16>
    //CHECK: %[[r320:.*]] = xetile.broadcast %[[r319]] [1] : vector<32x1xf16> -> vector<32x64xf16>
    //CHECK: %[[r321:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 0], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r322:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 8], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r323:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 16], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r324:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 24], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r325:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 32], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r326:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 40], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r327:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 48], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r328:.*]] = vector.extract_strided_slice %[[r320]] {offsets = [0, 56], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK: %[[r329:.*]] = xetile.transpose %[[r321]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r330:.*]] = xetile.transpose %[[r322]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r331:.*]] = xetile.transpose %[[r323]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r332:.*]] = xetile.transpose %[[r324]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r333:.*]] = xetile.transpose %[[r325]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r334:.*]] = xetile.transpose %[[r326]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r335:.*]] = xetile.transpose %[[r327]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r336:.*]] = xetile.transpose %[[r328]], [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK: %[[r337:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r338:.*]] = xetile.init_tile %[[arg0]][%[[c8]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r339:.*]] = xetile.init_tile %[[arg0]][%[[c16]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r340:.*]] = xetile.init_tile %[[arg0]][%[[c24]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r341:.*]] = xetile.init_tile %[[arg0]][%[[c32]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r342:.*]] = xetile.init_tile %[[arg0]][%[[c40]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r343:.*]] = xetile.init_tile %[[arg0]][%[[c48]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: %[[r344:.*]] = xetile.init_tile %[[arg0]][%[[c56]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r329]],  %[[r337]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r330]],  %[[r338]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r331]],  %[[r339]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r332]],  %[[r340]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r333]],  %[[r341]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r334]],  %[[r342]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r335]],  %[[r343]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    //CHECK: xetile.store_tile %[[r336]],  %[[r344]] : vector<8x32xf16>, !xetile.tile<8x32xf16>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    %3 = xetile.reduction <add>, %2 [1]: vector<32x64xf16> -> vector<32x1xf16>
    %4 = xetile.broadcast %3 [1]: vector<32x1xf16> -> vector<32x64xf16>
    %5 = xetile.transpose %4, [1, 0]: vector<32x64xf16> -> vector<64x32xf16>
    %6 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16>
    xetile.store_tile %5, %6: vector<64x32xf16>, !xetile.tile<64x32xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_softmax_dim_0
  //CHECK-SAME(%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_softmax_dim_0(%a: memref<1024x1024xf16>) {

    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

    //CHECK-COUNT-2: %{{.*}} = xetile.load_tile %{{.*}} : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>

    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = math.exp %{{.*}} : vector<8x32xf16>
    %3 = math.exp %2: vector<32x64xf16>

    //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<8x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-62: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    %4 = xetile.reduction <add>, %3 [0]: vector<32x64xf16> -> vector<1x64xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<64xf16> to vector<1x64xf16>
    //CHECK: %{{.*}} = xetile.broadcast %{{.*}} [0] : vector<1x64xf16> -> vector<32x64xf16>
    %5 = xetile.broadcast %4 [0]: vector<1x64xf16> -> vector<32x64xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = arith.divf %{{.*}}, %{{.*}} : vector<8x32xf16>
    %6 = arith.divf %3, %5: vector<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }

  //CHECK: (%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_softmax_dim_1(%a: memref<1024x1024xf16>) {
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.extract_strided_slice %[[r2]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.extract_strided_slice %[[r3]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = math.exp %{{.*}} : vector<8x32xf16>
    //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<8x32xf16> to vector<1x32xf16>

    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>

    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>

    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>

    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>

    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>

    //CHECK-COUNT-16: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK-COUNT-2: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x1xf16>, vector<16x1xf16>
    //CHECK: %{{.*}} = xetile.broadcast %{{.*}} [1] : vector<32x1xf16> -> vector<32x64xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = arith.divf %{{.*}}, %{{.*}} : vector<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    %3 = math.exp %2: vector<32x64xf16>
    %4 = xetile.reduction <add>, %3 [1]: vector<32x64xf16> -> vector<32x1xf16>
    %5 = xetile.broadcast %4 [1]: vector<32x1xf16> -> vector<32x64xf16>
    %6 = arith.divf %3, %5: vector<32x64xf16>
    %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_softmax_transpose
  //CHECK-SAME(%[[arg0:.*]]: memref<1024x1024xf16>)
  gpu.func @sglevel_softmax_transpose(%a: memref<1024x1024xf16>) {

    //CHECK-COUNT-16: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1] : vector<1x1xf16>, vector<1x1xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3] : vector<2x1xf16>, vector<2x1xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf16>, vector<4x1xf16>
    //CHECK-COUNT-2: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8x1xf16>, vector<8x1xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16x1xf16>, vector<16x1xf16>
    //CHECK: %{{.*}} = xetile.broadcast %{{.*}} [1] : vector<32x1xf16> -> vector<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>
    //CHECK-COUNT-8: %{{.*}} = arith.divf %{{.*}}, %{{.*}} : vector<32x8xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.transpose %{{.*}}, [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>
    %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>
    %3 = math.exp %2: vector<32x64xf16>
    %4 = xetile.reduction <add>, %3 [1]: vector<32x64xf16> -> vector<32x1xf16>
    %5 = xetile.broadcast %4 [1]: vector<32x1xf16> -> vector<32x64xf16>
    %6 = arith.divf %3, %5: vector<32x64xf16>
    %7 = xetile.transpose %6, [1, 0]: vector<32x64xf16> -> vector<64x32xf16>
    %8 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16>
    xetile.store_tile %7, %8: vector<64x32xf16>, !xetile.tile<64x32xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_unregular_gemm
  //CHECK-SAME: %[[arg0:.*]]: memref<16384x12288xf16>, %[[arg1:.*]]: memref<1536x12288xf16>, %[[arg2:.*]]: memref<16384x1536xf32>
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

    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<16384x1536xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<16384x1536xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<16384x1536xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<16384x1536xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<16384x1536xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<16384x1536xf32> -> !xetile.tile<8x16xf32>
    //CHECK-COUNT-4: %{{.*}} = xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<16384x1536xf32> -> !xetile.tile<8x16xf32>

    %15 = xetile.init_tile %arg2[%11, %14] : memref<16384x1536xf32> -> !xetile.tile<32x64xf32>
    %16 = index.remu %8, %c1
    %17 = index.mul %16, %c32
    //CHECK: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<16384x12288xf16> -> !xetile.tile<32x16xf16>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<16384x12288xf16> -> !xetile.tile<32x16xf16>
    %18 = xetile.init_tile %arg0[%11, %17] : memref<16384x12288xf16> -> !xetile.tile<32x32xf16>
    %19 = index.floordivs %6, %c8
    %20 = index.remu %6, %c8
    %21 = index.remu %19, %c4
    %22 = index.mul %21, %c64
    %23 = index.add %2, %22
    %24 = index.remu %20, %c1
    %25 = index.mul %24, %c32
    //CHECK: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16>
    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16>
    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16>
    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16>
    %26 = xetile.init_tile %arg1[%23, %25] : memref<1536x12288xf16> -> !xetile.tile<64x32xf16>
    %27:2 = scf.for %arg15 = %c0 to %c2 step %c1 iter_args(%arg16 = %15, %arg17 = %18) -> (!xetile.tile<32x64xf32>, !xetile.tile<32x32xf16>) {
      //CHECK-COUNT-2: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<32x16xf16>
      %28 = xetile.update_tile_offset %arg17, [%c1024,  %c0] :  !xetile.tile<32x32xf16>
      //CHECK-COUNT-16: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<8x16xf32>
      %29 = xetile.update_tile_offset %arg16, [%c1024,  %c0] : !xetile.tile<32x64xf32>
      //CHECK: %{{.*}}:26 = scf.for %[[arg22:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args({{.*}}) -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xetile.tile<32x16xf16>, !xetile.tile<32x16xf16>, !xetile.tile<16x16xf16>, !xetile.tile<16x16xf16>, !xetile.tile<16x16xf16>, !xetile.tile<16x16xf16>, !xetile.tile<16x16xf16>, !xetile.tile<16x16xf16>, !xetile.tile<16x16xf16>, !xetile.tile<16x16xf16>) {
      %30:3 = scf.for %arg18 = %c0 to %c12288 step %c32 iter_args(%arg19 = %cst, %arg20 = %arg17, %arg21 = %26) -> (vector<32x64xf32>, !xetile.tile<32x32xf16>, !xetile.tile<64x32xf16>) {
        //CHECK-COUNT-8: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<16x16xf16>
        %32 = xetile.update_tile_offset %arg21, [%c0,  %c32] : !xetile.tile<64x32xf16>
        //CHECK-COUNT-2: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<32x16xf16>
        %33 = xetile.update_tile_offset %arg20, [%c0,  %c32] :  !xetile.tile<32x32xf16>
        //CHECK-COUNT-2: %{{.*}} = xetile.load_tile %{{.*}} {padding = 0.000000e+00 : f32} : !xetile.tile<32x16xf16> -> vector<32x16xf16>
        //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
        %34 = xetile.load_tile %arg20 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        //CHECK-COUNT-8: %{{.*}} = math.exp %{{.*}} : vector<8x16xf16>
        %35 = math.exp %34 : vector<32x32xf16>
        //CHECK-COUNT-8: %{{.*}} = xetile.load_tile %{{.*}} {padding = 0.000000e+00 : f32} : !xetile.tile<16x16xf16> -> vector<16x16xf16>
        %36 = xetile.load_tile %arg21 {padding = 0.000000e+00 : f32}  : !xetile.tile<64x32xf16> -> vector<64x32xf16>
        //CHECK-COUNT-8: %{{.*}} = xetile.transpose %{{.*}}, [1, 0] : vector<16x16xf16> -> vector<16x16xf16>
        %37 = xetile.transpose %36, [1, 0] : vector<64x32xf16> -> vector<32x64xf16>
        //CHECK-COUNT-8: %{{.*}} = math.exp %{{.*}} : vector<16x16xf16>
        %38 = math.exp %37 : vector<32x64xf16>
        //CHECK: xegpu.compile_hint
        xegpu.compile_hint
        //CHECK-COUNT-32: %{{.*}} = xetile.tile_mma %{{.*}}, %{{.*}}, %{{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %39 = xetile.tile_mma %35, %38, %cst : vector<32x32xf16>, vector<32x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
        //CHECK: xegpu.compile_hint
        xegpu.compile_hint
        //CHECK-COUNT-16: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<8x16xf32>
        %40 = arith.addf %arg19, %39 : vector<32x64xf32>
        scf.yield %40, %33, %32 : vector<32x64xf32>, !xetile.tile<32x32xf16>, !xetile.tile<64x32xf16>
      }
      //CHECK-COUNT-16: %{{.*}} = math.exp %{{.*}} : vector<8x16xf32>
      %31 = math.exp %30#0 : vector<32x64xf32>
      //CHECK-COUNT-16: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
      xetile.store_tile %31,  %arg16 : vector<32x64xf32>, !xetile.tile<32x64xf32>
      scf.yield %29, %28 : !xetile.tile<32x64xf32>, !xetile.tile<32x32xf16>
    }
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sglevel_transpose_broadcast_dim_0
  //CHECK-SAME(%[[arg0:.*]]: memref<384x1xf32>, %[[arg1:.*]]: memref<256x384xf32>)
  gpu.func @sglevel_transpose_broadcast_dim_0(%arg0: memref<384x1xf32>, %arg1: memref<256x384xf32>) {
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][{{.*}}] : memref<384x1xf32> -> !xetile.tile<16x1xf32>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][{{.*}}] : memref<384x1xf32> -> !xetile.tile<16x1xf32>
    %1 = xetile.init_tile %arg0[0, 0] : memref<384x1xf32> -> !xetile.tile<32x1xf32>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] {padding = 0.000000e+00 : f32} : !xetile.tile<16x1xf32> -> vector<16x1xf32>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] {padding = 0.000000e+00 : f32} : !xetile.tile<16x1xf32> -> vector<16x1xf32>
    %2 = xetile.load_tile %1 {padding = 0.000000e+00 : f32} : !xetile.tile<32x1xf32> -> vector<32x1xf32>
    //CHECK: %[[r4:.*]] = xetile.transpose %[[r2]], [1, 0] : vector<16x1xf32> -> vector<1x16xf32>
    //CHECK: %[[r5:.*]] = xetile.transpose %[[r3]], [1, 0] : vector<16x1xf32> -> vector<1x16xf32>
    //CHECK: %[[r6:.*]] = vector.shape_cast %[[r4]] : vector<1x16xf32> to vector<16xf32>
    //CHECK: %[[r7:.*]] = vector.shape_cast %[[r5]] : vector<1x16xf32> to vector<16xf32>
    //CHECK: %[[r8:.*]] = vector.shuffle %[[r6]], %[[r7]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
    //CHECK: %[[r9:.*]] = vector.shape_cast %[[r8]] : vector<32xf32> to vector<1x32xf32>
    %3 = xetile.transpose %2, [1, 0] : vector<32x1xf32> -> vector<1x32xf32>
    //CHECK: %{{.*}} = xetile.broadcast %{{.*}} [0] : vector<1x32xf32> -> vector<64x32xf32>
    %4 = xetile.broadcast %3 [0] : vector<1x32xf32> -> vector<64x32xf32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<256x384xf32> -> !xetile.tile<8x16xf32>
    %5 = xetile.init_tile %arg1[0, 0] : memref<256x384xf32> -> !xetile.tile<64x32xf32>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<64x32xf32> to vector<8x32xf32>
    //CHECK-COUNT-16: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<8x32xf32> to vector<8x16xf32>
    //CHECK-COUNT-16: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %4, %5 : vector<64x32xf32>, !xetile.tile<64x32xf32>
    gpu.return
  }

  gpu.func @sglevel_transpose_broadcast_dim_1(%arg0: memref<1x384xf16>, %arg1: memref<384x256xf16>) {

    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1x384xf16> -> !xetile.tile<1x32xf16>
    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] {padding = 0.000000e+00 : f32} : !xetile.tile<1x32xf16> -> vector<1x32xf16>
    //CHECK: %[[r2:.*]] = xetile.transpose %[[r1]], [1, 0] : vector<1x32xf16> -> vector<32x1xf16>
    //CHECK: %[[r3:.*]] = xetile.broadcast %[[r2]] [1] : vector<32x1xf16> -> vector<32x64xf16>
    %1 = xetile.init_tile %arg0[0, 0] : memref<1x384xf16> -> !xetile.tile<1x32xf16>
    %2 = xetile.load_tile %1 {padding = 0.000000e+00 : f32} : !xetile.tile<1x32xf16> -> vector<1x32xf16>
    %3 = xetile.transpose %2, [1, 0] : vector<1x32xf16> -> vector<32x1xf16>
    %4 = xetile.broadcast %3 [1] : vector<32x1xf16> -> vector<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<384x256xf16> -> !xetile.tile<8x32xf16>
    %5 = xetile.init_tile %arg1[0, 0] : memref<384x256xf16> -> !xetile.tile<32x64xf16>
    //CHECK-COUNT-4: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 64], strides = [1, 1]} : vector<32x64xf16> to vector<8x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<8x64xf16> to vector<8x32xf16>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    xetile.store_tile %4, %5 : vector<32x64xf16>, !xetile.tile<32x64xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_loadgather
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>, %[[arg1:.*]]: vector<4x32xindex>
  gpu.func @sg_loadgather(%a: memref<1024xf16>, %indices: vector<4x32xindex>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<1x32xi1>
    //CHECK: %[[r0:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r1:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r2:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r4:.*]] = xetile.init_tile %[[arg0]], %[[r0]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r5:.*]] = xetile.init_tile %[[arg0]], %[[r1]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg0]], %[[r2]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r7:.*]] = xetile.init_tile %[[arg0]], %[[r3]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r8:.*]] = xetile.load %[[r4]], %[[cst]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    //CHECK: %[[r9:.*]] = xetile.load %[[r5]], %[[cst]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    //CHECK: %[[r10:.*]] = xetile.load %[[r6]], %[[cst]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    //CHECK: %[[r11:.*]] = xetile.load %[[r7]], %[[cst]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    %mask = arith.constant dense<1> : vector<4x32xi1>
    %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
    %2 = xetile.load %1, %mask : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1> -> vector<4x32xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_storescatter
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>, %[[arg1:.*]]: vector<4x32xindex>
  gpu.func @sg_storescatter(%a: memref<1024xf16>, %indices: vector<4x32xindex>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<4.200000e+01> : vector<1x32xf16>
    //CHECK: %[[cst_0:.*]] = arith.constant dense<true> : vector<1x32xi1>
    //CHECK: %[[cst_1:.*]] = arith.constant dense<1> : vector<1x32xindex>
    //CHECK: %[[r0:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r1:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r2:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r4:.*]] = xetile.init_tile %[[arg0]], %[[r0]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r5:.*]] = xetile.init_tile %[[arg0]], %[[r1]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg0]], %[[r2]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r7:.*]] = xetile.init_tile %[[arg0]], %[[r3]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: xetile.store %[[cst]], %[[r4]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst]], %[[r5]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst]], %[[r6]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst]], %[[r7]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: %[[r8:.*]] = xetile.update_tile_offset %[[r4]], %[[cst_1]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: %[[r9:.*]] = xetile.update_tile_offset %[[r5]], %[[cst_1]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: %[[r10:.*]] = xetile.update_tile_offset %[[r6]], %[[cst_1]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: %[[r11:.*]] = xetile.update_tile_offset %[[r7]], %[[cst_1]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: xetile.store %[[cst]], %[[r8]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst]], %[[r9]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst]], %[[r10]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst]], %[[r11]], %[[cst_0]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
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
    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<1x16xi1>
    //CHECK: %[[cast:.*]] = memref.cast %[[arg0]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_0:.*]] = memref.cast %[[arg1]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_1:.*]] = memref.cast %[[arg2]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
    //CHECK: %[[r0:.*]] = arith.muli %[[block_id_x]], %[[c1024]] : index
    //CHECK: %[[r1:.*]] = vector.splat %[[r0]] : vector<1x32xindex>
    //CHECK: %[[r2:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r4:.*]] = xetile.init_tile %[[cast]], %[[r2]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r5:.*]] = xetile.init_tile %[[cast]], %[[r3]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r6:.*]] = xetile.load %[[r4]], %[[cst]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    //CHECK: %[[r7:.*]] = xetile.load %[[r5]], %[[cst]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    //CHECK: %[[r8:.*]] = xetile.init_tile %[[cast_0]], %[[r2]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r9:.*]] = xetile.init_tile %[[cast_0]], %[[r3]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r10:.*]] = xetile.load %[[r8]], %[[cst]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    //CHECK: %[[r11:.*]] = xetile.load %[[r9]], %[[cst]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    //CHECK: %[[r12:.*]] = arith.addf %[[r6]], %[[r10]] : vector<1x16xf32>
    //CHECK: %[[r13:.*]] = arith.addf %[[r7]], %[[r11]] : vector<1x16xf32>
    //CHECK: %[[r14:.*]] = xetile.init_tile %[[cast_1]], %[[r2]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r15:.*]] = xetile.init_tile %[[cast_1]], %[[r3]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: xetile.store %[[r12]], %[[r14]], %[[cst]] : vector<1x16xf32>, !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1>
    //CHECK: xetile.store %[[r13]], %[[r15]], %[[cst]] : vector<1x16xf32>, !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1>

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
