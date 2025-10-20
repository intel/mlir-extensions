// RUN: imex-opt --split-input-file --xetile-blocking %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {

  // CHECK-LABEL: test_blocking_elementwise
  //  CHECK-SAME: (%[[A:.*]]: vector<64x64xf16>, %[[B:.*]]: vector<64x64xf16>)
  func.func @test_blocking_elementwise(%a: vector<64x64xf16>, %b: vector<64x64xf16>) -> (vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>) {
    // Elementwise arith ops are handled in unified way, check some
    //CHECK-COUNT-2: %{{.*}} = vector.extract_strided_slice %[[A]] {offsets = [0, {{.*}}], sizes = [64, 32], strides = [1, 1]} : vector<64x64xf16> to vector<64x32xf16>
    //CHECK-COUNT-2: %{{.*}} = vector.extract_strided_slice %[[B]] {offsets = [0, {{.*}}], sizes = [64, 32], strides = [1, 1]} : vector<64x64xf16> to vector<64x32xf16>
    %0 = arith.addf %a, %b: vector<64x64xf16>

    //CHECK-COUNT-2: %{{.*}} = arith.negf %{{.*}} : vector<64x32xf16>
    %1 = arith.negf %a: vector<64x64xf16>
    //CHECK-COUNT-2: %{{.*}} = math.exp %{{.*}} : vector<64x32xf16>
    %2 = math.exp %a: vector<64x64xf16>
    return %0, %1, %2 : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf16>
  }

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
    //CHECK-COUNT-8: xetile.init_tile %arg0[{{.*}}] : memref<128x128xf16> -> !xetile.tile<20x32xf16>
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<80x64xf16>
    //CHECK-COUNT-8: xetile.load_tile {{.*}} : !xetile.tile<20x32xf16> -> vector<20x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<80x64xf16> -> vector<80x64xf16>
    //CHECK: gpu.return
    gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_unaligned_2(%[[arg0:.*]]: memref<128x128xf16>)
  gpu.func @sg_load_tile_unaligned_2(%a: memref<128x128xf16>) {
    %c0 = arith.constant 0 : index
    //CHECK: xetile.init_tile %arg0[{{.*}}] : memref<128x128xf16> -> !xetile.tile<24x32xf16>
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<24x32xf16>
    //CHECK: xetile.load_tile {{.*}} : !xetile.tile<24x32xf16> -> vector<24x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<24x32xf16> -> vector<24x32xf16>
    //CHECK: gpu.return
    gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_unaligned_3(%[[arg0:.*]]: memref<128x128xf16>)
  gpu.func @sg_load_tile_unaligned_3(%a: memref<128x128xf16>) {
    %c0 = arith.constant 0 : index
    //CHECK-COUNT-2: xetile.init_tile %arg0[{{.*}}] : memref<128x128xf16> -> !xetile.tile<28x32xf16>
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<56x32xf16>
    //CHECK-COUNT-2: xetile.load_tile {{.*}} : !xetile.tile<28x32xf16> -> vector<28x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<56x32xf16> -> vector<56x32xf16>
    //CHECK: gpu.return
    gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_unaligned_4(%[[arg0:.*]]: memref<128x128xf16>)
  gpu.func @sg_load_tile_unaligned_4(%a: memref<128x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.0> : vector<64x32xf16>
    //CHECK-COUNT-10: xetile.init_tile %arg0[{{.*}}] : memref<128x128xf16> -> !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<80x64xf16>
    //CHECK-COUNT-10: xetile.load_tile {{.*}} : !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<16x16xf16>, vector<16x16xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<80x64xf16> -> vector<80x64xf16>
    %3 = xetile.tile_mma %2, %cst: vector<80x64xf16>, vector<64x32xf16> -> vector<80x32xf32>
    //CHECK: gpu.return
    gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_unaligned_5(%[[arg0:.*]]: memref<128x128xf16>)
  gpu.func @sg_load_tile_unaligned_5(%a: memref<128x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.0> : vector<32x32xf16>
    //CHECK-COUNT-7: xetile.init_tile %arg0[{{.*}}] : memref<128x128xf16> -> !xetile.tile<8x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<56x32xf16>
    //CHECK-COUNT-7: xetile.load_tile {{.*}} : !xetile.tile<8x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<8x16xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<56x32xf16> -> vector<56x32xf16>
    %3 = xetile.tile_mma %2, %cst: vector<56x32xf16>, vector<32x32xf16> -> vector<56x32xf32>
    //CHECK: gpu.return
    gpu.return
  }

  //CHECK: gpu.func @sg_load_tile_unaligned_6(%[[arg0:.*]]: memref<128x128xf16>)
  gpu.func @sg_load_tile_unaligned_6(%a: memref<128x128xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<24x48xf16>
    %2 = xetile.init_tile %a[%c64, %c64] : memref<128x128xf16> -> !xetile.tile<48x32xf16>
    //CHECK-COUNT-3: xetile.load_tile {{.*}} : !xetile.tile<24x16xf16> -> vector<24x16xf16>
    %3 = xetile.load_tile %1 : !xetile.tile<24x48xf16> -> vector<24x48xf16>

    //CHECK-COUNT-3: xetile.load_tile {{.*}} : !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<16x16xf16>, vector<16x16xf16>
    %4 = xetile.load_tile %2 : !xetile.tile<48x32xf16> -> vector<48x32xf16>
    %5 = xetile.tile_mma %3, %4: vector<24x48xf16>, vector<48x32xf16> -> vector<24x32xf32>
    gpu.return
  }

  //CHECK: gpu.func @sg_store_tile(%[[arg0:.*]]: memref<32x32xf32>)
	gpu.func @sg_store_tile(%a: memref<32x32xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
		%result = arith.constant dense<0.0>: vector<32x32xf32>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %arg0[%{{.*}}, %{{.*}}] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
		%1 = xetile.init_tile %a[0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
		xetile.store_tile %result, %1: vector<32x32xf32>, !xetile.tile<32x32xf32>
    //CHECK: gpu.return
		gpu.return
	}

  //CHECK: gpu.func @create_mask
  //CHECK-SAME: %[[arg0:.*]]: vector<32x32xf16>, %[[arg1:.*]]: vector<32x32xf16>, %[[arg2:.*]]: memref<32x32xf16>
  gpu.func @create_mask(%a: vector<32x32xf16>, %b: vector<32x32xf16>, %c: memref<32x32xf16>) {
    %c32 = arith.constant 32 : index
    //CHECK: %[[c20:.*]] = arith.constant 20 : index
    %c20 = arith.constant 20 : index

    //CHECK-COUNT-32: %{{.*}} = vector.create_mask %{{.*}}, %[[c20]] : vector<1x32xi1>
    %mask = vector.create_mask %c32, %c20 : vector<32x32xi1>

    //CHECK-2D-FOR-LATER-COUNT-4: vector.extract_strided_slice %[[arg0]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-2D-FOR-LATER-COUNT-4: vector.extract_strided_slice %[[arg1]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-2D-FOR-LATER-COUNT-4: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<8x32xi1>, vector<8x32xf16>
    //CHECK-COUNT-32: vector.extract_strided_slice %[[arg0]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-32: vector.extract_strided_slice %[[arg1]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-32: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<1x32xi1>, vector<1x32xf16>
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
    // CHECK: %[[c32:.*]] = arith.constant 32 : index
    %c32 = arith.constant 32 : index
    //CHECK-2D-FOR-LATER: %[[r0:.*]] = vector.constant_mask [8, 32] : vector<8x32xi1>
    //CHECK-2D-FOR-LATER: %[[r1:.*]] = vector.constant_mask [4, 32] : vector<8x32xi1>
    //CHECK-2D-FOR-LATER: %[[r2:.*]] = vector.constant_mask [0, 0] : vector<8x32xi1>
    //CHECK: %{{.*}} = vector.create_mask %{{.*}}, %[[c32]] : vector<1x32xi1>
    %mask = vector.create_mask %c20, %c32 : vector<32x32xi1>

    //CHECK-2D-FOR-LATER-COUNT: vector.extract_strided_slice %[[arg0]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-2D-FOR-LATER-COUNT: vector.extract_strided_slice %[[arg1]] {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
    //CHECK-2D-FOR-LATER-COUNT: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<8x32xi1>, vector<8x32xf16>
    //CHECK-COUNT-32: vector.extract_strided_slice %[[arg0]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-32: vector.extract_strided_slice %[[arg1]] {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-32: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<1x32xi1>, vector<1x32xf16>
    %select = arith.select %mask, %a, %b : vector<32x32xi1>, vector<32x32xf16>

    //CHECK-COUNT-4: xetile.init_tile %[[arg2]][%{{.*}}, %{{.*}}] : memref<32x32xf16> -> !xetile.tile<8x32xf16>
    %tile = xetile.init_tile %c[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK-COUNT-4: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
    xetile.store_tile %select, %tile: vector<32x32xf16>, !xetile.tile<32x32xf16>
    gpu.return
  }

  //CHECK: gpu.func @sg_tile_mma(%[[arg0:.*]]: memref<32x32xf16>, %[[arg1:.*]]: memref<32x32xf16>)
  //CHECK: xetile.init_tile %[[arg0]][{{.*}}] : memref<32x32xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
  //CHECK: xetile.load_tile %{{.*}} : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
  //CHECK: xetile.init_tile %[[arg1]][{{.*}}] : memref<32x32xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
  //CHECK: xetile.load_tile %{{.*}} : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
  //CHECK-COUNT-8: vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
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

  // CHECK-LABEL: gpu.func @inner_reduction
  // CHECK-SAME: (%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>)
  gpu.func @inner_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<16x1xf16>
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
    //CHECK: %[[r98:.*]] = vector.extract %[[r97]][0] : f16 from vector<16xf16>
    //CHECK: %[[r99:.*]] = vector.broadcast %[[r98]] : f16 to vector<1x1xf16>
    //CHECK: %[[r100:.*]] = vector.extract %[[r97]][1] : f16 from vector<16xf16>
    //CHECK: %[[r101:.*]] = vector.broadcast %[[r100]] : f16 to vector<1x1xf16>
    //CHECK: %[[r102:.*]] = vector.extract %[[r97]][2] : f16 from vector<16xf16>
    //CHECK: %[[r103:.*]] = vector.broadcast %[[r102]] : f16 to vector<1x1xf16>
    //CHECK: %[[r104:.*]] = vector.extract %[[r97]][3] : f16 from vector<16xf16>
    //CHECK: %[[r105:.*]] = vector.broadcast %[[r104]] : f16 to vector<1x1xf16>
    //CHECK: %[[r106:.*]] = vector.extract %[[r97]][4] : f16 from vector<16xf16>
    //CHECK: %[[r107:.*]] = vector.broadcast %[[r106]] : f16 to vector<1x1xf16>
    //CHECK: %[[r108:.*]] = vector.extract %[[r97]][5] : f16 from vector<16xf16>
    //CHECK: %[[r109:.*]] = vector.broadcast %[[r108]] : f16 to vector<1x1xf16>
    //CHECK: %[[r110:.*]] = vector.extract %[[r97]][6] : f16 from vector<16xf16>
    //CHECK: %[[r111:.*]] = vector.broadcast %[[r110]] : f16 to vector<1x1xf16>
    //CHECK: %[[r112:.*]] = vector.extract %[[r97]][7] : f16 from vector<16xf16>
    //CHECK: %[[r113:.*]] = vector.broadcast %[[r112]] : f16 to vector<1x1xf16>
    //CHECK: %[[r114:.*]] = vector.extract %[[r97]][8] : f16 from vector<16xf16>
    //CHECK: %[[r115:.*]] = vector.broadcast %[[r114]] : f16 to vector<1x1xf16>
    //CHECK: %[[r116:.*]] = vector.extract %[[r97]][9] : f16 from vector<16xf16>
    //CHECK: %[[r117:.*]] = vector.broadcast %[[r116]] : f16 to vector<1x1xf16>
    //CHECK: %[[r118:.*]] = vector.extract %[[r97]][10] : f16 from vector<16xf16>
    //CHECK: %[[r119:.*]] = vector.broadcast %[[r118]] : f16 to vector<1x1xf16>
    //CHECK: %[[r120:.*]] = vector.extract %[[r97]][11] : f16 from vector<16xf16>
    //CHECK: %[[r121:.*]] = vector.broadcast %[[r120]] : f16 to vector<1x1xf16>
    //CHECK: %[[r122:.*]] = vector.extract %[[r97]][12] : f16 from vector<16xf16>
    //CHECK: %[[r123:.*]] = vector.broadcast %[[r122]] : f16 to vector<1x1xf16>
    //CHECK: %[[r124:.*]] = vector.extract %[[r97]][13] : f16 from vector<16xf16>
    //CHECK: %[[r125:.*]] = vector.broadcast %[[r124]] : f16 to vector<1x1xf16>
    //CHECK: %[[r126:.*]] = vector.extract %[[r97]][14] : f16 from vector<16xf16>
    //CHECK: %[[r127:.*]] = vector.broadcast %[[r126]] : f16 to vector<1x1xf16>
    //CHECK: %[[r128:.*]] = vector.extract %[[r97]][15] : f16 from vector<16xf16>
    //CHECK: %[[r129:.*]] = vector.broadcast %[[r128]] : f16 to vector<1x1xf16>

    //CHECK: %[[r130:.*]] = vector.insert_strided_slice %[[r99]], %[[cst]] {offsets = [0, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r131:.*]] = vector.insert_strided_slice %[[r101]], %[[r130]] {offsets = [1, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r132:.*]] = vector.insert_strided_slice %[[r103]], %[[r131]] {offsets = [2, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r133:.*]] = vector.insert_strided_slice %[[r105]], %[[r132]] {offsets = [3, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r134:.*]] = vector.insert_strided_slice %[[r107]], %[[r133]] {offsets = [4, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r135:.*]] = vector.insert_strided_slice %[[r109]], %[[r134]] {offsets = [5, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r136:.*]] = vector.insert_strided_slice %[[r111]], %[[r135]] {offsets = [6, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r137:.*]] = vector.insert_strided_slice %[[r113]], %[[r136]] {offsets = [7, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r138:.*]] = vector.insert_strided_slice %[[r115]], %[[r137]] {offsets = [8, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r139:.*]] = vector.insert_strided_slice %[[r117]], %[[r138]] {offsets = [9, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r140:.*]] = vector.insert_strided_slice %[[r119]], %[[r139]] {offsets = [10, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r141:.*]] = vector.insert_strided_slice %[[r121]], %[[r140]] {offsets = [11, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r142:.*]] = vector.insert_strided_slice %[[r123]], %[[r141]] {offsets = [12, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r143:.*]] = vector.insert_strided_slice %[[r125]], %[[r142]] {offsets = [13, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r144:.*]] = vector.insert_strided_slice %[[r127]], %[[r143]] {offsets = [14, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r145:.*]] = vector.insert_strided_slice %[[r129]], %[[r144]] {offsets = [15, 0], strides = [1, 1]} : vector<1x1xf16> into vector<16x1xf16>
    //CHECK: %[[r146:.*]] = vector.shape_cast %[[r145]] : vector<16x1xf16> to vector<2x8xf16>
    //CHECK: %[[r147:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<2x8xf16>
    //CHECK: xetile.store_tile %[[r146]],  %[[r147]] : vector<2x8xf16>, !xetile.tile<2x8xf16>
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

    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index

    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant dense<0.0>: vector<32x32xf32>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r10:.*]]:10 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]] iter_args(%[[arg4:.*]] = %[[r0]], %[[arg5:.*]] = %[[r1]], %[[arg6:.*]] = %[[cst:.*]], %[[arg7:.*]] = %[[cst:.*]], %[[arg8:.*]] = %[[cst:.*]], %[[arg9:.*]] = %[[cst:.*]], %[[arg10:.*]] = %[[cst:.*]], %[[arg11:.*]] = %[[cst:.*]], %[[arg12:.*]] = %[[cst:.*]], %[[arg13:.*]] = %[[cst:.*]]) -> (!xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      //CHECK:   %[[r11:.*]]:2 = xetile.load_tile %[[arg4]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
      //CHECK:   %[[r12:.*]]:2 = xetile.load_tile %[[arg5]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
      //CHECK:   %[[r13:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]], %[[c32]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
      //CHECK:   %[[r14:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c32]], %[[c0]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>

      //CHEECK: %[[r15:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r16:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r17:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r18:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r19:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r20:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r21:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r22:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHEECK: %[[r23:.*]] = vector.extract_strided_slice %[[r20]]#0 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHEECK: %[[r24:.*]] = vector.extract_strided_slice %[[r20]]#1 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHEECK: %[[r25:.*]] = vector.extract_strided_slice %[[r20]]#0 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHEECK: %[[r26:.*]] = vector.extract_strided_slice %[[r20]]#1 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>

      //CHECK: %[[r27:.*]] = xetile.tile_mma %[[r15]], %[[r23]], %[[arg6]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r28:.*]] = xetile.tile_mma %[[r16]], %[[r25]], %[[r27]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r29:.*]] = xetile.tile_mma %[[r15]], %[[r24]], %[[arg7]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r30:.*]] = xetile.tile_mma %[[r16]], %[[r26]], %[[r29]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r31:.*]] = xetile.tile_mma %[[r17]], %[[r23]], %[[arg8]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r32:.*]] = xetile.tile_mma %[[r18]], %[[r25]], %[[r31]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r33:.*]] = xetile.tile_mma %[[r17]], %[[r24]], %[[arg9]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r34:.*]] = xetile.tile_mma %[[r18]], %[[r26]], %[[r33]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r35:.*]] = xetile.tile_mma %[[r19]], %[[r23]], %[[arg10]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r36:.*]] = xetile.tile_mma %[[r20]], %[[r25]], %[[r35]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r37:.*]] = xetile.tile_mma %[[r19]], %[[r24]], %[[arg11]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r38:.*]] = xetile.tile_mma %[[r20]], %[[r26]], %[[r37]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r39:.*]] = xetile.tile_mma %[[r21]], %[[r23]], %[[arg12]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r40:.*]] = xetile.tile_mma %[[r22]], %[[r25]], %[[r39]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r41:.*]] = xetile.tile_mma %[[r21]], %[[r24]], %[[arg13]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      //CHECK: %[[r42:.*]] = xetile.tile_mma %[[r22]], %[[r26]], %[[r41]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]:  !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]:  !xetile.tile<32x32xf16>
      %c_new_value = xetile.tile_mma %3, %4, %c_value:
        vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      //CHECK: scf.yield %[[r13]], %[[r14]], %[[r28]], %[[r30]], %[[r32]], %[[r34]], %[[r36]], %[[r38]], %[[r40]], %[[r42]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }

    //CHECK: %[[r13:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r14:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r15:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r16:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r17:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r18:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r19:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r20:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>

    //CHECK: xetile.store_tile %[[r10]]#2,  %[[r13]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#3,  %[[r14]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#4,  %[[r15]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#5,  %[[r16]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#6,  %[[r17]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#7,  %[[r18]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#8,  %[[r19]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#9,  %[[r20]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
  	gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_gemm_with_preops_for_c
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x128xf16>, %[[arg1:.*]]: memref<128x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_gemm_with_preops_for_c(%a: memref<32x128xf16>, %b: memref<128x32xf16>, %c: memref<32x32xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    //CHECK: %[[c24:.*]] = arith.constant 24 : index
    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c16:.*]] = arith.constant 16 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    //CHECK: %[[r10:.*]]:10 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]] iter_args(%[[arg4:.*]] = %[[r0]], %[[arg5:.*]] = %[[r1]], %[[arg6:.*]] = %[[cst:.*]], %[[arg7:.*]] = %[[cst:.*]], %[[arg8:.*]] = %[[cst:.*]], %[[arg9:.*]] = %[[cst:.*]], %[[arg10:.*]] = %[[cst:.*]], %[[arg11:.*]] = %[[cst:.*]], %[[arg12:.*]] = %[[cst:.*]], %[[arg13:.*]] = %[[cst:.*]]) -> (!xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>) {
    //CHECK:   %[[r19:.*]]:2 = xetile.load_tile %[[arg4]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
    //CHECK:   %[[r20:.*]]:2 = xetile.load_tile %[[arg5]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
    //CHECK:   %[[r21:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]], %[[c32]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    //CHECK:   %[[r22:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c32]], %[[c0]]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>


    //CHECK:   %[[r23:.*]] = arith.addf %[[arg6]], %[[arg6]] : vector<8x16xf32>
    //CHECK:   %[[r24:.*]] = arith.addf %[[arg7]], %[[arg7]] : vector<8x16xf32>
    //CHECK:   %[[r25:.*]] = arith.addf %[[arg8]], %[[arg8]] : vector<8x16xf32>
    //CHECK:   %[[r26:.*]] = arith.addf %[[arg9]], %[[arg9]] : vector<8x16xf32>
    //CHECK:   %[[r27:.*]] = arith.addf %[[arg10]], %[[arg10]] : vector<8x16xf32>
    //CHECK:   %[[r28:.*]] = arith.addf %[[arg11]], %[[arg11]] : vector<8x16xf32>
    //CHECK:   %[[r29:.*]] = arith.addf %[[arg12]], %[[arg12]] : vector<8x16xf32>
    //CHECK:   %[[r30:.*]] = arith.addf %[[arg13]], %[[arg13]] : vector<8x16xf32>

    //CHECK:   %[[r37:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r38:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r39:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r40:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r41:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r42:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r43:.*]] = vector.extract_strided_slice %[[r19]]#0 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r44:.*]] = vector.extract_strided_slice %[[r19]]#1 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
    //CHECK:   %[[r45:.*]] = vector.extract_strided_slice %[[r20]]#0 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK:   %[[r46:.*]] = vector.extract_strided_slice %[[r20]]#1 {offsets = [0, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK:   %[[r47:.*]] = vector.extract_strided_slice %[[r20]]#0 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
    //CHECK:   %[[r48:.*]] = vector.extract_strided_slice %[[r20]]#1 {offsets = [16, 0], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>

    //CHECK:   %[[r49:.*]] = xetile.tile_mma %[[r37]], %[[r45]], %[[r23]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r50:.*]] = xetile.tile_mma %[[r38]], %[[r47]], %[[r49]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r51:.*]] = xetile.tile_mma %[[r37]], %[[r46]], %[[r24]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r52:.*]] = xetile.tile_mma %[[r38]], %[[r48]], %[[r51]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r53:.*]] = xetile.tile_mma %[[r39]], %[[r45]], %[[r25]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r54:.*]] = xetile.tile_mma %[[r40]], %[[r47]], %[[r53]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r55:.*]] = xetile.tile_mma %[[r39]], %[[r46]], %[[r26]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r56:.*]] = xetile.tile_mma %[[r40]], %[[r48]], %[[r55]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r57:.*]] = xetile.tile_mma %[[r41]], %[[r45]], %[[r27]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r58:.*]] = xetile.tile_mma %[[r42]], %[[r47]], %[[r57]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r59:.*]] = xetile.tile_mma %[[r41]], %[[r46]], %[[r28]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r60:.*]] = xetile.tile_mma %[[r42]], %[[r48]], %[[r59]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r61:.*]] = xetile.tile_mma %[[r43]], %[[r45]], %[[r29]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r62:.*]] = xetile.tile_mma %[[r44]], %[[r47]], %[[r61]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r63:.*]] = xetile.tile_mma %[[r43]], %[[r46]], %[[r30]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   %[[r64:.*]] = xetile.tile_mma %[[r44]], %[[r48]], %[[r63]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    //CHECK:   scf.yield %[[r21]], %[[r22]], %[[r50]], %[[r52]], %[[r54]], %[[r56]], %[[r58]], %[[r60]], %[[r62]], %[[r64]] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
    //CHECK: }
    //CHECK: %[[r13:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r14:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r15:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r16:.*]] = xetile.init_tile %[[arg2]][%[[c8]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r17:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r18:.*]] = xetile.init_tile %[[arg2]][%[[c16]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r19:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: %[[r20:.*]] = xetile.init_tile %[[arg2]][%[[c24]], %[[c16]]] : memref<32x32xf32> -> !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#2,  %[[r13]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#3,  %[[r14]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#4,  %[[r15]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#5,  %[[r16]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#6,  %[[r17]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#7,  %[[r18]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#8,  %[[r19]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
    //CHECK: xetile.store_tile %[[r10]]#9,  %[[r20]] : vector<8x16xf32>, !xetile.tile<8x16xf32>
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
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32x64xf16>
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK-COUNT-62: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>

    //CHECK-COUNT-64: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x32xf16> into vector<32x64xf16>

    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
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
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK-COUNT-64: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>

    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<1x32xf16>
    //CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x32xf16> to vector<32xf16>

    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = vector.shuffle %{{.*}}, %{{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf16>, vector<32xf16>
    //CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<32xf16>

    //CHECK-COUNT-32: %{{.*}} = vector.extract %{{.*}}[{{.*}}] : f16 from vector<32xf16>
    //CHECK-COUNT-32: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>

    //CHECK-COUNT-64: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x32xf16> into vector<32x64xf16>

    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg0]][{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x64xf16> to vector<8x32xf16>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
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
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c32]]] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r3:.*]] = xetile.load_tile %[[r1]] : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK: %[[r4:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r6:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r10:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r11:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r12:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r13:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r14:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r15:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r18:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r19:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r20:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r21:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r22:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r23:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r24:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r25:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r26:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r27:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r28:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r29:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r30:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r31:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r32:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r33:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r34:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r35:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r36:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [16, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r37:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [16, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r38:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [17, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r39:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [17, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r40:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [18, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r41:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [18, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r42:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [19, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r43:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [19, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r44:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [20, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r45:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [20, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r46:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [21, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r47:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [21, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r48:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [22, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r49:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [22, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r50:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [23, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r51:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [23, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r52:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [24, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r53:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [24, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r54:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [25, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r55:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [25, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r56:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [26, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r57:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [26, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r58:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [27, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r59:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [27, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r60:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [28, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r61:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [28, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r62:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [29, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r63:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [29, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r64:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [30, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r65:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [30, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r66:.*]] = vector.extract_strided_slice %[[r2]] {offsets = [31, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r67:.*]] = vector.extract_strided_slice %[[r3]] {offsets = [31, 0], sizes = [1, 32], strides = [1, 1]} : vector<32x32xf16> to vector<1x32xf16>
    //CHECK: %[[r68:.*]] = arith.addf %[[r4]], %[[r5]] : vector<1x32xf16>
    //CHECK: %[[r69:.*]] = vector.shape_cast %[[r68]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r70:.*]] = arith.addf %[[r6]], %[[r7]] : vector<1x32xf16>
    //CHECK: %[[r71:.*]] = vector.shape_cast %[[r70]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r72:.*]] = arith.addf %[[r8]], %[[r9]] : vector<1x32xf16>
    //CHECK: %[[r73:.*]] = vector.shape_cast %[[r72]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r74:.*]] = arith.addf %[[r10]], %[[r11]] : vector<1x32xf16>
    //CHECK: %[[r75:.*]] = vector.shape_cast %[[r74]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r76:.*]] = arith.addf %[[r12]], %[[r13]] : vector<1x32xf16>
    //CHECK: %[[r77:.*]] = vector.shape_cast %[[r76]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r78:.*]] = arith.addf %[[r14]], %[[r15]] : vector<1x32xf16>
    //CHECK: %[[r79:.*]] = vector.shape_cast %[[r78]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r80:.*]] = arith.addf %[[r16]], %[[r17]] : vector<1x32xf16>
    //CHECK: %[[r81:.*]] = vector.shape_cast %[[r80]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r82:.*]] = arith.addf %[[r18]], %[[r19]] : vector<1x32xf16>
    //CHECK: %[[r83:.*]] = vector.shape_cast %[[r82]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r84:.*]] = arith.addf %[[r20]], %[[r21]] : vector<1x32xf16>
    //CHECK: %[[r85:.*]] = vector.shape_cast %[[r84]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r86:.*]] = arith.addf %[[r22]], %[[r23]] : vector<1x32xf16>
    //CHECK: %[[r87:.*]] = vector.shape_cast %[[r86]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r88:.*]] = arith.addf %[[r24]], %[[r25]] : vector<1x32xf16>
    //CHECK: %[[r89:.*]] = vector.shape_cast %[[r88]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r90:.*]] = arith.addf %[[r26]], %[[r27]] : vector<1x32xf16>
    //CHECK: %[[r91:.*]] = vector.shape_cast %[[r90]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r92:.*]] = arith.addf %[[r28]], %[[r29]] : vector<1x32xf16>
    //CHECK: %[[r93:.*]] = vector.shape_cast %[[r92]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r94:.*]] = arith.addf %[[r30]], %[[r31]] : vector<1x32xf16>
    //CHECK: %[[r95:.*]] = vector.shape_cast %[[r94]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r96:.*]] = arith.addf %[[r32]], %[[r33]] : vector<1x32xf16>
    //CHECK: %[[r97:.*]] = vector.shape_cast %[[r96]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r98:.*]] = arith.addf %[[r34]], %[[r35]] : vector<1x32xf16>
    //CHECK: %[[r99:.*]] = vector.shape_cast %[[r98]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r100:.*]] = arith.addf %[[r36]], %[[r37]] : vector<1x32xf16>
    //CHECK: %[[r101:.*]] = vector.shape_cast %[[r100]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r102:.*]] = arith.addf %[[r38]], %[[r39]] : vector<1x32xf16>
    //CHECK: %[[r103:.*]] = vector.shape_cast %[[r102]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r104:.*]] = arith.addf %[[r40]], %[[r41]] : vector<1x32xf16>
    //CHECK: %[[r105:.*]] = vector.shape_cast %[[r104]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r106:.*]] = arith.addf %[[r42]], %[[r43]] : vector<1x32xf16>
    //CHECK: %[[r107:.*]] = vector.shape_cast %[[r106]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r108:.*]] = arith.addf %[[r44]], %[[r45]] : vector<1x32xf16>
    //CHECK: %[[r109:.*]] = vector.shape_cast %[[r108]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r110:.*]] = arith.addf %[[r46]], %[[r47]] : vector<1x32xf16>
    //CHECK: %[[r111:.*]] = vector.shape_cast %[[r110]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r112:.*]] = arith.addf %[[r48]], %[[r49]] : vector<1x32xf16>
    //CHECK: %[[r113:.*]] = vector.shape_cast %[[r112]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r114:.*]] = arith.addf %[[r50]], %[[r51]] : vector<1x32xf16>
    //CHECK: %[[r115:.*]] = vector.shape_cast %[[r114]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r116:.*]] = arith.addf %[[r52]], %[[r53]] : vector<1x32xf16>
    //CHECK: %[[r117:.*]] = vector.shape_cast %[[r116]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r118:.*]] = arith.addf %[[r54]], %[[r55]] : vector<1x32xf16>
    //CHECK: %[[r119:.*]] = vector.shape_cast %[[r118]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r120:.*]] = arith.addf %[[r56]], %[[r57]] : vector<1x32xf16>
    //CHECK: %[[r121:.*]] = vector.shape_cast %[[r120]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r122:.*]] = arith.addf %[[r58]], %[[r59]] : vector<1x32xf16>
    //CHECK: %[[r123:.*]] = vector.shape_cast %[[r122]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r124:.*]] = arith.addf %[[r60]], %[[r61]] : vector<1x32xf16>
    //CHECK: %[[r125:.*]] = vector.shape_cast %[[r124]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r126:.*]] = arith.addf %[[r62]], %[[r63]] : vector<1x32xf16>
    //CHECK: %[[r127:.*]] = vector.shape_cast %[[r126]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r128:.*]] = arith.addf %[[r64]], %[[r65]] : vector<1x32xf16>
    //CHECK: %[[r129:.*]] = vector.shape_cast %[[r128]] : vector<1x32xf16> to vector<32xf16>
    //CHECK: %[[r130:.*]] = arith.addf %[[r66]], %[[r67]] : vector<1x32xf16>
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
    //CHECK-COUNT-32: %{{.*}} = vector.extract %{{.*}}[{{.*}}] : f16 from vector<32xf16>
    //CHECK-COUNT-32: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x8xf16>

    //CHECK-COUNT-256: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x8xf16> into vector<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [32, 8], strides = [1, 1]} : vector<32x64xf16> to vector<32x8xf16>

    //CHECK-COUNT-8: %{{.*}} = xetile.transpose %{{.*}}, [1, 0] : vector<32x8xf16> -> vector<8x32xf16>
    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<1024x1024xf16> -> !xetile.tile<8x32xf16>
    //CHECK-COUNT-8: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x32xf16>, !xetile.tile<8x32xf16>
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

    //CHECK-COUNT-64: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x32xf16> into vector<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x64xf16> to vector<8x32xf16>

    %4 = xetile.reduction <add>, %3 [0]: vector<32x64xf16> -> vector<1x64xf16>
    %5 = xetile.broadcast %4 [0]: vector<1x64xf16> -> vector<32x64xf16>
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
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x32xf16> to vector<8x32xf16>
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

    //CHECK-COUNT-32: %{{.*}} = vector.extract %{{.*}}[{{.*}}] : f16 from vector<32xf16>
    //CHECK-COUNT-32: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>

    //CHECK-COUNT-8: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x32xf16> into vector<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x64xf16> to vector<8x32xf16>

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

    //CHECK-COUNT-32: %{{.*}} = vector.extract %{{.*}}[{{.*}}] : f16 from vector<32xf16>
    //CHECK-COUNT-32: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x8xf16>
    //CHECK-COUNT-256: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x8xf16> into vector<32x64xf16>
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

    //CHECK: %{{.*}} = xetile.init_tile %[[arg0]][%{{.*}}, %{{.*}}] : memref<16384x12288xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>

    %18 = xetile.init_tile %arg0[%11, %17] : memref<16384x12288xf16> -> !xetile.tile<32x32xf16>
    %19 = index.floordivs %6, %c8
    %20 = index.remu %6, %c8
    %21 = index.remu %19, %c4
    %22 = index.mul %21, %c64
    %23 = index.add %2, %22
    %24 = index.remu %20, %c1
    %25 = index.mul %24, %c32

    //CHECK: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<1536x12288xf16> -> !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    %26 = xetile.init_tile %arg1[%23, %25] : memref<1536x12288xf16> -> !xetile.tile<64x32xf16>
    %27:2 = scf.for %arg15 = %c0 to %c2 step %c1 iter_args(%arg16 = %15, %arg17 = %18) -> (!xetile.tile<32x64xf32>, !xetile.tile<32x32xf16>) {
      //CHECK: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
      //CHECK-COUNT-16: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<8x16xf32>
      %28 = xetile.update_tile_offset %arg17, [%c1024,  %c0] :  !xetile.tile<32x32xf16>
      %29 = xetile.update_tile_offset %arg16, [%c1024,  %c0] : !xetile.tile<32x64xf32>
      //CHECK: %{{.*}}:21 = scf.for %[[arg22:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args({{.*}}) -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>, !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>) {
      %30:3 = scf.for %arg18 = %c0 to %c12288 step %c32 iter_args(%arg19 = %cst, %arg20 = %arg17, %arg21 = %26) -> (vector<32x64xf32>, !xetile.tile<32x32xf16>, !xetile.tile<64x32xf16>) {
        //CHECK-COUNT-4: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
        //CHECK: %{{.*}} = xetile.update_tile_offset %{{.*}}, [%{{.*}}, %{{.*}}] : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
        %32 = xetile.update_tile_offset %arg21, [%c0,  %c32] : !xetile.tile<64x32xf16>
        %33 = xetile.update_tile_offset %arg20, [%c0,  %c32] :  !xetile.tile<32x32xf16>
        //CHECK: %{{.*}} = xetile.load_tile %{{.*}} {padding = 0.000000e+00 : f32} : !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<32x16xf16>, vector<32x16xf16>
        //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
        %34 = xetile.load_tile %arg20 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        //CHECK-COUNT-8: %{{.*}} = math.exp %{{.*}} : vector<8x16xf16>
        %35 = math.exp %34 : vector<32x32xf16>
        //CHECK-COUNT-4: %{{.*}} = xetile.load_tile %{{.*}} {padding = 0.000000e+00 : f32} : !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<16x16xf16>, vector<16x16xf16>
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
    %3 = xetile.transpose %2, [1, 0] : vector<32x1xf32> -> vector<1x32xf32>

    //CHECK-COUNT-128: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x16xf32> into vector<64x32xf32>
    %4 = xetile.broadcast %3 [0] : vector<1x32xf32> -> vector<64x32xf32>

    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<256x384xf32> -> !xetile.tile<8x16xf32>
    %5 = xetile.init_tile %arg1[0, 0] : memref<256x384xf32> -> !xetile.tile<64x32xf32>
    //CHECK-COUNT-16: xetile.store_tile %{{.*}},  %{{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %4, %5 : vector<64x32xf32>, !xetile.tile<64x32xf32>
    gpu.return
  }

  gpu.func @sglevel_transpose_broadcast_dim_1(%arg0: memref<1x384xf16>, %arg1: memref<384x256xf16>) {

    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1x384xf16> -> !xetile.tile<1x32xf16>
    //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] {padding = 0.000000e+00 : f32} : !xetile.tile<1x32xf16> -> vector<1x32xf16>
    //CHECK: %[[r2:.*]] = xetile.transpose %[[r1]], [1, 0] : vector<1x32xf16> -> vector<32x1xf16>
    %1 = xetile.init_tile %arg0[0, 0] : memref<1x384xf16> -> !xetile.tile<1x32xf16>
    %2 = xetile.load_tile %1 {padding = 0.000000e+00 : f32} : !xetile.tile<1x32xf16> -> vector<1x32xf16>
    %3 = xetile.transpose %2, [1, 0] : vector<1x32xf16> -> vector<32x1xf16>

    //CHECK: %{{.*}} = vector.extract %{{.*}}[0, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[1, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[2, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[3, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[4, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[5, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[6, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[7, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[8, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[9, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[10, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[11, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[12, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[13, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[14, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[15, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[16, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[17, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[18, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[19, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[20, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[21, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[22, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[23, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[24, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[25, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[26, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[27, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[28, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[29, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[30, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>
    //CHECK: %{{.*}} = vector.extract %{{.*}}[31, 0] : f16 from vector<32x1xf16>
    //CHECK: %{{.*}} = vector.broadcast %{{.*}} : f16 to vector<1x32xf16>

    //CHECK-COUNT-64: %{{.*}} = vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [{{.*}}], strides = [1, 1]} : vector<1x32xf16> into vector<32x64xf16>
    %4 = xetile.broadcast %3 [1] : vector<32x1xf16> -> vector<32x64xf16>

    //CHECK-COUNT-8: %{{.*}} = xetile.init_tile %[[arg1]][%{{.*}}, %{{.*}}] : memref<384x256xf16> -> !xetile.tile<8x32xf16>
    %5 = xetile.init_tile %arg1[0, 0] : memref<384x256xf16> -> !xetile.tile<32x64xf16>
    //CHECK-COUNT-8: %{{.*}} = vector.extract_strided_slice %{{.*}} {offsets = [{{.*}}], sizes = [8, 32], strides = [1, 1]} : vector<32x64xf16> to vector<8x32xf16>
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
    //CHECK: %[[r12:.*]] = xetile.load %[[r4]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    //CHECK: %[[r13:.*]] = xetile.load %[[r5]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    //CHECK: %[[r14:.*]] = xetile.load %[[r6]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    //CHECK: %[[r15:.*]] = xetile.load %[[r7]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf16>
    %mask = arith.constant dense<1> : vector<4x32xi1>
    %1 = xetile.init_tile %a, %indices : memref<1024xf16>, vector<4x32xindex> -> !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>
    %2 = xetile.load %1, %mask : !xetile.tile<4x32xf16, #xetile.tile_attr<scattered = true>>, vector<4x32xi1> -> vector<4x32xf16>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @sg_storescatter
  //CHECK-SAME: %[[arg0:.*]]: memref<1024xf16>, %[[arg1:.*]]: vector<4x32xindex>
  gpu.func @sg_storescatter(%a: memref<1024xf16>, %indices: vector<4x32xindex>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<1> : vector<1x32xindex>
    //CHECK: %[[cst_0:.*]] = arith.constant dense<true> : vector<1x32xi1>
    //CHECK: %[[cst_1:.*]] = arith.constant dense<4.200000e+01> : vector<1x32xf16>
    //CHECK: %[[r0:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r1:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r2:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[arg1]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<4x32xindex> to vector<1x32xindex>
    //CHECK: %[[r4:.*]] = xetile.init_tile %[[arg0]], %[[r0]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r5:.*]] = xetile.init_tile %[[arg0]], %[[r1]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r6:.*]] = xetile.init_tile %[[arg0]], %[[r2]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r7:.*]] = xetile.init_tile %[[arg0]], %[[r3]] : memref<1024xf16>, vector<1x32xindex> -> !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r4]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r5]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r6]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r7]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: %[[r20:.*]] = xetile.update_tile_offset %[[r4]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: %[[r21:.*]] = xetile.update_tile_offset %[[r5]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: %[[r22:.*]] = xetile.update_tile_offset %[[r6]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: %[[r23:.*]] = xetile.update_tile_offset %[[r7]], %[[cst:.*]] : !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r20]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r21]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r22]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    //CHECK: xetile.store %[[cst_1:.*]], %[[r23]], %[[cst_0:.*]] : vector<1x32xf16>, !xetile.tile<1x32xf16, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
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

    //CHECK: %[[cst:.*]] = arith.constant dense<true> : vector<1x16xi1>
    //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
    //CHECK: %[[cast:.*]] = memref.cast %[[arg0]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_0:.*]] = memref.cast %[[arg1]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[cast_1:.*]] = memref.cast %[[arg2]] : memref<*xf32> to memref<?xf32>
    //CHECK: %[[block_id_x:.*]] = gpu.block_id  x
    //CHECK: %[[r0:.*]] = arith.muli %[[block_id_x]], %[[c1024]] : index
    //CHECK: %[[r1:.*]] = vector.broadcast %[[r0]] : index to vector<1x32xindex>
    //CHECK: %[[r2:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r4:.*]] = xetile.init_tile %[[cast]], %[[r2]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r5:.*]] = xetile.init_tile %[[cast]], %[[r3]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r6:.*]] = xetile.load %[[r4]], %[[cst:.*]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    //CHECK: %[[r7:.*]] = xetile.load %[[r5]], %[[cst:.*]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>

    //CHECK: %[[r8:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r9:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r10:.*]] = xetile.init_tile %[[cast_0]], %[[r8]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r11:.*]] = xetile.init_tile %[[cast_0]], %[[r9]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r12:.*]] = xetile.load %[[r10]], %[[cst:.*]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    //CHECK: %[[r13:.*]] = xetile.load %[[r11]], %[[cst:.*]] : !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    //CHECK: %[[r14:.*]] = arith.addf %[[r6]], %[[r12]] : vector<1x16xf32>
    //CHECK: %[[r15:.*]] = arith.addf %[[r7]], %[[r13]] : vector<1x16xf32>

    //CHECK: %[[r16:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r17:.*]] = vector.extract_strided_slice %[[r1]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xindex> to vector<1x16xindex>
    //CHECK: %[[r18:.*]] = xetile.init_tile %[[cast_1]], %[[r16]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: %[[r19:.*]] = xetile.init_tile %[[cast_1]], %[[r17]] : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>
    //CHECK: xetile.store %[[r14]], %[[r18]], %[[cst:.*]] : vector<1x16xf32>, !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1>
    //CHECK: xetile.store %[[r15]], %[[r19]], %[[cst:.*]] : vector<1x16xf32>, !xetile.tile<1x16xf32, #xetile.tile_attr<scattered = true>>, vector<1x16xi1>

    %cst = arith.constant dense<true> : vector<1x32xi1>
    %c1024 = arith.constant 1024 : index
    %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %block_id_x = gpu.block_id  x
    %0 = arith.muli %block_id_x, %c1024 : index
    %1 = vector.broadcast %0 : index to vector<1x32xindex>
    %2 = xetile.init_tile %cast, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    %3 = xetile.load %2, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
    %4 = xetile.init_tile %cast_0, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    %5 = xetile.load %4, %cst : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
    %6 = arith.addf %3, %5 : vector<1x32xf32>
    %7 = xetile.init_tile %cast_1, %1 : memref<?xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
    xetile.store %6, %7, %cst : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @store_tile_slm
  gpu.func @store_tile_slm() {
    %c0 = arith.constant 0 : index
    %a = arith.constant dense<0.000000e+00> : vector<24x32xf16>
    %slm = memref.alloc() : memref<8192xi8, 3>
    %view = memref.view %slm[%c0][] : memref<8192xi8, 3> to memref<64x64xf16, 3>
    //CHECK-COUNT-6: {{.*}} = xetile.init_tile {{.*}} : memref<64x64xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>>
    %st_tile = xetile.init_tile %view[%c0, %c0] : memref<64x64xf16, 3> -> !xetile.tile<24x32xf16, #xetile.tile_attr<memory_space=3>>
    //CHECK-COUNT-6: xetile.store_tile {{.*}} : vector<8x16xf16>, !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>>
    xetile.store_tile %a, %st_tile : vector<24x32xf16>, !xetile.tile<24x32xf16, #xetile.tile_attr<memory_space=3>>
    gpu.return
  }

  //CHECK-LABEL: gpu.func @load_store_tile_slm_transpose
  gpu.func @load_store_tile_slm_transpose(%arg0: memref<512x128xf16, 3>) {

    //CHECK-COUNT-4: {{.*}} = xetile.init_tile {{.*}} : memref<512x128xf16, 3> -> !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>>
    //CHECK-COUNT-4: {{.*}} = xetile.load_tile {{.*}} : !xetile.tile<8x16xf16, #xetile.tile_attr<memory_space = 3 : i64>> -> vector<8x16xf16>
    %0 = xetile.init_tile %arg0[0, 0] : memref<512x128xf16, 3> -> !xetile.tile<16x32xf16, #xetile.tile_attr<memory_space = 3 : i64>>
    %1 = xetile.load_tile %0 : !xetile.tile<16x32xf16, #xetile.tile_attr<memory_space = 3 : i64>> -> vector<16x32xf16>
    %transpose = memref.transpose %arg0 (i, j) -> (j, i) : memref<512x128xf16, 3> to memref<128x512xf16, strided<[1, 128]>, 3>

    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile {{.*}} : memref<128x512xf16, strided<[1, 128]>, 3> -> !xetile.tile<16x16xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>
    //CHECK-COUNT-2: xetile.store_tile {{.*}} : vector<16x16xf16>, !xetile.tile<16x16xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>
    %2 = xetile.init_tile %transpose[16, 32] : memref<128x512xf16, strided<[1, 128]>, 3> -> !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>
    xetile.store_tile %1,  %2 : vector<16x32xf16>, !xetile.tile<16x32xf16, #xetile.tile_attr<order = [0, 1], memory_space = 3 : i64>>
    gpu.return
  }

  gpu.func @mma_with_scattered_load_store(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>) {
    %cst = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : vector<1x16xindex>
    %idx = xetile.broadcast %cst [0] : vector<1x16xindex> -> vector<16x16xindex>
    %mask = arith.constant dense<true> : vector<16x16xi1>
    %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %cast_2 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %cast_3 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile {{.*}} : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %a_tile = xetile.init_tile %cast, %idx : memref<?xf32>, vector<16x16xindex> -> !xetile.tile<16x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    //CHECK-COUNT-16: %{{.*}} = xetile.load {{.*}} : !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    %a = xetile.load %a_tile, %mask : !xetile.tile<16x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<16x16xi1> -> vector<16x16xf32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile {{.*}} : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %b_tile = xetile.init_tile %cast_2, %idx : memref<?xf32>, vector<16x16xindex> -> !xetile.tile<16x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    //CHECK-COUNT-16: %{{.*}} = xetile.load {{.*}} : !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x16xi1> -> vector<1x16xf32>
    %b = xetile.load %b_tile, %mask : !xetile.tile<16x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<16x16xi1> -> vector<16x16xf32>
    //CHECK-COUNT-4: %{{.*}} = xetile.tile_mma {{.*}} : vector<8x8xf32>, vector<8x16xf32>{{.*}} -> vector<8x16xf32>
    %mma = xetile.tile_mma %a, %b : vector<16x16xf32>, vector<16x16xf32> -> vector<16x16xf32>
    //CHECK-COUNT-16: %{{.*}} = xetile.init_tile {{.*}} : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %c_tile = xetile.init_tile %cast_3, %idx : memref<?xf32>, vector<16x16xindex> -> !xetile.tile<16x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    //CHECK-COUNT-16: xetile.store {{.*}} : vector<1x16xf32>, !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x16xi1>
    xetile.store %mma, %c_tile, %mask : vector<16x16xf32>, !xetile.tile<16x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<16x16xi1>
    gpu.return
  }

  //-----
  //CHECK-LABEL: gpu.func @sg_gemm_with_select
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x128xf16>, %[[arg1:.*]]: memref<128x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_gemm_with_select(%a: memref<32x128xf16>, %b: memref<128x32xf16>, %c: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant dense<0.0>: vector<1x16xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    //CHECK: xetile.init_tile {{.*}} : memref<32x128xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>
    //CHECK: xetile.init_tile {{.*}} : memref<128x32xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>

    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst_0)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {

      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]:  !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]:  !xetile.tile<32x32xf16>

      %cmp = arith.cmpi slt, %k, %c64 : index
      %data = arith.select %cmp, %cst_0, %c_value : vector<32x32xf32>

      //CHECK-COUNT-8: vector.extract_strided_slice {{.*}} : vector<32x32xf32> to vector<8x16xf32>
      %c_new_value = xetile.tile_mma %3, %4, %data:
        vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>

      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }

  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
    //CHECK-COUNT-8: xetile.store_tile {{.*}} : vector<8x16xf32>, !xetile.tile<8x16xf32>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
  	gpu.return
  }

  //-----
  //CHECK-LABEL: gpu.func @scf_if
  //CHECK-SAME: (%[[arg0:.*]]: memref<*xf32>, %[[arg1:.*]]: memref<*xf32>, %[[arg2:.*]]: memref<*xf32>, %[[arg3:.*]]: i32)
  gpu.func @scf_if(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<1> : vector<1x256xindex>
    %cst_0 = arith.constant dense<true> : vector<1x256xi1>
    %block_id_x = gpu.block_id  x
    %cast = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %cast_1 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %cast_2 = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %0 = arith.index_cast %block_id_x : index to i32
    %1 = xetile.init_tile %cast_2, %cst : memref<?xf32>, vector<1x256xindex> -> !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %2 = xetile.load %1, %cst_0 : !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x256xi1> -> vector<1x256xf32>
    %3 = xetile.init_tile %cast_1, %cst : memref<?xf32>, vector<1x256xindex> -> !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %4 = xetile.load %3, %cst_0 : !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x256xi1> -> vector<1x256xf32>
    %5 = arith.cmpi eq, %0, %c1_i32 : i32
    %6 = scf.if %5 -> (vector<1x256xf32>) {
      //CHECK-COUNT-16: %{{.*}} = arith.addf {{.*}} : vector<1x16xf32>
      %8 = arith.addf %2, %4 : vector<1x256xf32>
      //CHECK: scf.yield %{{.*}} : vector<1x16xf32>, {{.*}}
      scf.yield %8 : vector<1x256xf32>
    } else {
      //CHECK-COUNT-16: %{{.*}} = arith.subf {{.*}} : vector<1x16xf32>
      %8 = arith.subf %2, %4 : vector<1x256xf32>
      //CHECK: scf.yield %{{.*}} : vector<1x16xf32>, {{.*}}
      scf.yield %8 : vector<1x256xf32>
    }
    %7 = xetile.init_tile %cast, %cst : memref<?xf32>, vector<1x256xindex> -> !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    xetile.store %6, %7, %cst_0 : vector<1x256xf32>, !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x256xi1>
    gpu.return
  }

  //-----
  //CHECK-LABEL: gpu.func @small_gemm
  //CHECK-SAME: (%[[arg0:.*]]: memref<4x32xf16>, %[[arg1:.*]]: memref<32x32xf16>, %[[arg2:.*]]: memref<4x32xf32>)
  gpu.func @small_gemm(%arg0: memref<4x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<4x32xf32>) {
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4x16xf32>
    %cst = arith.constant dense<0.000000e+00> : vector<4x32xf32>
    //CHECK: %{{.*}} = xetile.init_tile %[[arg0]][{{.*}}] : memref<4x32xf16> -> !xetile.tile<4x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    %0 = xetile.init_tile %arg0[0, 0] : memref<4x32xf16> -> !xetile.tile<4x32xf16>
    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg1]][{{.*}}] : memref<32x32xf16> -> !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>>
    %1 = xetile.init_tile %arg1[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %{{.*}} = xetile.load_tile {{.*}} : !xetile.tile<4x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<4x16xf16>, vector<4x16xf16>
    %2 = xetile.load_tile %0 : !xetile.tile<4x32xf16> -> vector<4x32xf16>
    //CHECK-COUNT-2: %{{.*}} = xetile.load_tile {{.*}} : !xetile.tile<16x16xf16, #xetile.tile_attr<array_length = 2 : i64>> -> vector<16x16xf16>, vector<16x16xf16>
    %3 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
    //CHECK-COUNT-4: %{{.*}} = xetile.transpose %{{.*}}, [1, 0] : vector<16x16xf16> -> vector<16x16xf16>
    %4 = xetile.transpose %3, [1, 0] : vector<32x32xf16> -> vector<32x32xf16>
    //CHECK-COUNT-4: %{{.*}} = xetile.tile_mma {{.*}} : vector<4x16xf16>, vector<16x16xf16>, vector<4x16xf32> -> vector<4x16xf32>
    %5 = xetile.tile_mma %2, %4, %cst : vector<4x32xf16>, vector<32x32xf16>, vector<4x32xf32> -> vector<4x32xf32>
    //CHECK-COUNT-2: %{{.*}} = xetile.init_tile %[[arg2]][{{.*}}] : memref<4x32xf32> -> !xetile.tile<4x16xf32>
    %6 = xetile.init_tile %arg2[0, 0] : memref<4x32xf32> -> !xetile.tile<4x32xf32>
    //CHECK-COUNT-2: xetile.store_tile %{{.*}},  %{{.*}} : vector<4x16xf32>, !xetile.tile<4x16xf32>
    xetile.store_tile %5,  %6 : vector<4x32xf32>, !xetile.tile<4x32xf32>
    gpu.return
  }

  //-----
  gpu.func @while_loop_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %block_id_x = gpu.block_id x
    %thread_id_x = gpu.thread_id x
    %cast = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %cast_1 = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %index = arith.addi %block_id_x, %thread_id_x : index
    %tile_indices = vector.broadcast %index : index to vector<1x256xindex>
    %tile_indices_i32 = arith.index_cast %tile_indices : vector<1x256xindex> to vector<1x256xi32>
    %arg3_broadcast = vector.broadcast %arg3 : i32 to vector<1x256xi32>
    %mask = arith.cmpi slt, %tile_indices_i32, %arg3_broadcast : vector<1x256xi32>
    %tile = xetile.init_tile %cast_1, %tile_indices : memref<?xf32>, vector<1x256xindex> -> !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %load1 = xetile.load %tile, %mask : !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x256xi1> -> vector<1x256xf32>
    %tile_0 = xetile.init_tile %cast_0, %tile_indices : memref<?xf32>, vector<1x256xindex> -> !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %load2 = xetile.load %tile_0, %mask : !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x256xi1> -> vector<1x256xf32>
    %sum = arith.addf %load1, %load2 : vector<1x256xf32>
    %result:2 = scf.while (%arg4 = %sum, %arg5 = %c0_i32) : (vector<1x256xf32>, i32) -> (vector<1x256xf32>, i32) {
        %cond = arith.cmpi slt, %arg5, %c10_i32 : i32
        scf.condition(%cond) %arg4, %arg5 : vector<1x256xf32>, i32
    } do {
    ^bb0(%arg4: vector<1x256xf32>, %arg5: i32):
        //CHECK-COUNT-16: arith.addf {{.*}} : vector<1x16xf32>
        %new_sum = arith.addf %arg4, %load1 : vector<1x256xf32>
        //CHECK-COUNT-16: arith.addf {{.*}} : vector<1x16xf32>
        %new_sum2 = arith.addf %new_sum, %load2 : vector<1x256xf32>
        %new_iter = arith.addi %arg5, %c1_i32 : i32
        scf.yield %new_sum2, %new_iter : vector<1x256xf32>, i32
    }
    //CHECK-COUNT-16: xetile.init_tile {{.*}} : memref<?xf32>, vector<1x16xindex> -> !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    %tile_out = xetile.init_tile %cast, %tile_indices : memref<?xf32>, vector<1x256xindex> -> !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>
    //CHECK-COUNT-16: xetile.store {{.*}} : vector<1x16xf32>, !xetile.tile<1x16xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x16xi1>
    xetile.store %result#0, %tile_out, %mask : vector<1x256xf32>, !xetile.tile<1x256xf32, #xetile.tile_attr<memory_space = 0 : i32, scattered = true>>, vector<1x256xi1>
    gpu.return
  }
}
