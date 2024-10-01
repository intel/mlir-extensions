// RUN: imex-opt --split-input-file --xetile-blocking %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_load_tile(%[[arg0:.*]]: memref<32x32xf16>)
  //CHECK: %[[c0:.*]] = arith.constant 0 : index
  //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
  //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
  //CHECK: gpu.return
  gpu.func @sg_load_tile(%a: memref<32x32xf16>) {
    %c0 = arith.constant 0 : index
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  	gpu.return
  }

  //CHECK: gpu.func @sg_tiled_store(%[[arg0:.*]]: memref<32x32xf32>)
  //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<1x2x32x16xf32>
  //CHECK: %[[R0:.*]] = xetile.tile_unpack %[[cst]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf32> -> vector<32x32xf32>
  //CHECK: %[[R1:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  //CHECK: %[[R2:.*]] = xetile.tile_pack %[[R0]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf32> -> vector<4x2x8x16xf32>
  //CHECK: xetile.store_tile %[[R2]],  %[[R1]] : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  //CHECK: gpu.return
	gpu.func @sg_tiled_store(%a: memref<32x32xf32>) {
		%result = arith.constant dense<0.0>: vector<32x32xf32>
		%1 = xetile.init_tile %a[0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>
		xetile.store_tile %result, %1: vector<32x32xf32>, !xetile.tile<32x32xf32>
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

    //CHECK: gpu.return
  	gpu.return
  }

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

    //CHECK: %[[R2:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[R0]], %[[arg5:.*]] = %[[R1]], %[[arg6:.*]] = %[[cst]])
    //CHECK-SAME: !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>,
    //CHECK-SAME: !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x2x8x16xf32>
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      //CHECK: %[[R8:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[R9:.*]] = xetile.tile_unpack %[[R8]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[R10:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[R11:.*]] = xetile.tile_unpack %[[R10]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[R12:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c32]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]: !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>

      //CHECK: %[[R13:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c32]],  %[[c0]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]: !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>

      //CHECK: %[[R14:.*]] = xetile.tile_pack %[[R9]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>
      //CHECK: %[[R15:.*]] = xetile.tile_pack %[[R11]] {inner_blocks = array<i64: 16, 16>}  : vector<32x32xf16> -> vector<2x2x16x16xf16>
      //CHECK: %[[R16:.*]] = xetile.tile_mma %[[R14]], %[[R15]], %[[arg6]] : vector<4x2x8x16xf16>, vector<2x2x16x16xf16>, vector<4x2x8x16xf32> -> vector<4x2x8x16xf32>
      %c_new_value = xetile.tile_mma %3, %4, %c_value:
        vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>

      //CHECK: scf.yield %[[R12]], %[[R13]], %[[R16]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<4x2x8x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }

    //CHECK: %[[REG6:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>

    //CHECK: xetile.store_tile %[[R2]]#2,  %[[REG6]] : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>

    //CHECK: gpu.return
  	gpu.return
  }

    //CHECK: gpu.func @load_tile(%[[arg0:.*]]: memref<128x128xf16>)
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<85x76xf16, #xetile.tile_attr<inner_blocks = [17, 19]>>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<85x76xf16, #xetile.tile_attr<inner_blocks = [17, 19]>> -> vector<5x4x17x19xf16>
    //CHECK: gpu.return
    gpu.func @load_tile(%a: memref<128x128xf16>) {
      %c0 = arith.constant 0 : index
  	  %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<85x76xf16>
      %2 = xetile.load_tile %1 : !xetile.tile<85x76xf16> -> vector<85x76xf16>
  	  gpu.return
    }

    //CHECK: gpu.func @store_tile(%[[arg0:.*]]: memref<128x128xf32>)
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<2x19x22x2xf32>
    //CHECK: %[[R0:.*]] = xetile.tile_unpack %[[cst]] {inner_blocks = array<i64: 22, 2>}  : vector<2x19x22x2xf32> -> vector<44x38xf32>
    //CHECK: %[[R1:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<128x128xf32> -> !xetile.tile<44x38xf32, #xetile.tile_attr<inner_blocks = [4, 2]>>
    //CHECK: %[[R2:.*]] = xetile.tile_pack %[[R0]] {inner_blocks = array<i64: 4, 2>}  : vector<44x38xf32> -> vector<11x19x4x2xf32>
    //CHECK: xetile.store_tile %[[R2]],  %[[R1]] : vector<11x19x4x2xf32>, !xetile.tile<44x38xf32, #xetile.tile_attr<inner_blocks = [4, 2]>>
    //CHECK: gpu.return
	  gpu.func @store_tile(%a: memref<128x128xf32>) {
		  %result = arith.constant dense<0.0>: vector<44x38xf32>
		  %1 = xetile.init_tile %a[0, 0] : memref<128x128xf32> -> !xetile.tile<44x38xf32>
		  xetile.store_tile %result, %1: vector<44x38xf32>, !xetile.tile<44x38xf32>
		  gpu.return
	  }

    //CHECK: gpu.func @create_mask
    //CHECK-DAG: %[[C20:.*]] = arith.constant 20
    //CHECK-DAG: %[[C32:.*]] = arith.constant 32
    //CHECK: %[[MASK:.*]] = vector.create_mask %[[C32]], %[[C20]], %[[C32]], %[[C20]] : vector<32x1x1x32xi1>
    //CHECK: arith.select %[[MASK]], {{.*}} : vector<32x1x1x32xi1>, vector<32x1x1x32xf16>
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
    //CHECK-DAG: %[[C20:.*]] = arith.constant 20
    //CHECK-DAG: %[[C32:.*]] = arith.constant 32
    //CHECK: %[[MASK:.*]] = vector.create_mask %[[C20]], %[[C32]], %[[C20]], %[[C32]] : vector<32x1x1x32xi1>
    //CHECK: arith.select %[[MASK]], {{.*}} : vector<32x1x1x32xi1>, vector<32x1x1x32xf16>
    gpu.func @create_mask_2(%a: vector<32x32xf16>, %b: vector<32x32xf16>, %c: memref<32x32xf16>) {
      %c20 = arith.constant 20 : index
      %c32 = arith.constant 32 : index
      %mask = vector.create_mask %c20, %c32 : vector<32x32xi1>
      %select = arith.select %mask, %a, %b : vector<32x32xi1>, vector<32x32xf16>
      %tile = xetile.init_tile %c[0, 0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
      xetile.store_tile %select, %tile: vector<32x32xf16>, !xetile.tile<32x32xf16>
      gpu.return
    }

    ///-CHECK: gpu.func @tile_mma(%[[arg0:.*]]: memref<128x128xf16>, %[[arg1:.*]]: memref<128x128xf16>)
    ///-CHECK: %[[c0:.*]] = arith.constant 0 : index
    ///-CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<90x76xf16, #xetile.tile_attr<inner_blocks = [30, 19]>>
    ///-CHECK: %[[R1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1], inner_blocks = [19, 6]>>
    ///-  gpu.func @tile_mma(%a: memref<128x128xf16>, %b: memref<128x128xf16>) {
    ///-    %c0 = arith.constant 0 : index
  	///-    %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<90x76xf16>
    ///-    %2 = xetile.init_tile %b[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1]>>
    ///-    %3 = xetile.load_tile %1 : !xetile.tile<90x76xf16> -> vector<90x76xf16>
    ///-    %4 = xetile.load_tile %2 : !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<76x90xf16>
    ///-    %5 = xetile.tile_mma %3, %4: vector<90x76xf16>, vector<76x90xf16> -> vector<90x90xf32>
  	///-    gpu.return
    ///-  }

    // CHECK-LABEL: gpu.func @inner_reduction
    // CHECK-SAME: (%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>)
    gpu.func @inner_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
      // CHECK: %[[c0:.*]] = arith.constant 0 : index
      // CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
      %c0 = arith.constant 0 : index
      %acc = arith.constant dense<0.0> : vector<16xf16>
      // CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>>
      %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
      // CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>> -> vector<1x1x16x32xf16>
      // CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 16, 32>}  : vector<1x1x16x32xf16> -> vector<16x32xf16>
      %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
      // CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 1, 32>}  : vector<16x32xf16> -> vector<16x1x1x32xf16>
      // CHECK: %[[R4:.*]] = math.exp %[[R3]] : vector<16x1x1x32xf16>
      %e = math.exp %v: vector<16x32xf16>
      // CHECK: %[[R5:.*]] = vector.shape_cast %[[cst]] : vector<16xf16> to vector<16x1xf16>
      // CHECK: %[[R6:.*]] = vector.multi_reduction <add>, %[[R4]], %[[R5]] [1, 3] : vector<16x1x1x32xf16> to vector<16x1xf16>
      // CHECK: %[[R7:.*]] = vector.shape_cast %[[R6]] : vector<16x1xf16> to vector<16xf16>
      %r = vector.multi_reduction <add>, %e, %acc [1] : vector<16x32xf16> to vector<16xf16>
      // CHECK: %[[R8:.*]] = vector.shape_cast %[[R7]] : vector<16xf16> to vector<2x8xf16>
      %c = vector.shape_cast %r: vector<16xf16> to vector<2x8xf16>
      // CHECK: %[[R9:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<2x8xf16, #xetile.tile_attr<inner_blocks = [2, 8]>>
      %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<2x8xf16>
      // CHECK: %[[R10:.*]] = xetile.tile_pack %[[R8]] {inner_blocks = array<i64: 2, 8>}  : vector<2x8xf16> -> vector<1x1x2x8xf16>
      // CHECK: xetile.store_tile %[[R10]],  %[[R9]] : vector<1x1x2x8xf16>, !xetile.tile<2x8xf16, #xetile.tile_attr<inner_blocks = [2, 8]>>
      xetile.store_tile %c, %s : vector<2x8xf16>, !xetile.tile<2x8xf16>
      gpu.return
    }

    //CHECK: gpu.func @outter_reduction(%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>) {
    gpu.func @outter_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32xf16>
      %c0 = arith.constant 0 : index
      %acc = arith.constant dense<0.0> : vector<32xf16>
      //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>>
      %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
      //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] : !xetile.tile<16x32xf16, #xetile.tile_attr<inner_blocks = [16, 32]>> -> vector<1x1x16x32xf16>
      //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] {inner_blocks = array<i64: 16, 32>}  : vector<1x1x16x32xf16> -> vector<16x32xf16>
      %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
      //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R2]] {inner_blocks = array<i64: 1, 32>}  : vector<16x32xf16> -> vector<16x1x1x32xf16>
      //CHECK: %[[R4:.*]] = math.exp %[[R3]] : vector<16x1x1x32xf16>
      %e = math.exp %v: vector<16x32xf16>
      //CHECK: %[[R5:.*]] = vector.shape_cast %[[cst]] : vector<32xf16> to vector<1x32xf16>
      //CHECK: %[[R6:.*]] = vector.multi_reduction <add>, %[[R4]], %[[R5]] [0, 2] : vector<16x1x1x32xf16> to vector<1x32xf16>
      //CHECK: %[[R7:.*]] = vector.shape_cast %[[R6]] : vector<1x32xf16> to vector<32xf16>
      %r = vector.multi_reduction <add>, %e, %acc [0] : vector<16x32xf16> to vector<32xf16>
      //CHECK: %[[R8:.*]] = vector.shape_cast %[[R7]] : vector<32xf16> to vector<4x8xf16>
      %c = vector.shape_cast %r: vector<32xf16> to vector<4x8xf16>
      //CHECK: %[[R9:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xetile.tile<4x8xf16, #xetile.tile_attr<inner_blocks = [4, 8]>>
      %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<4x8xf16>
      //CHECK: %[[R10:.*]] = xetile.tile_pack %[[R8]] {inner_blocks = array<i64: 4, 8>}  : vector<4x8xf16> -> vector<1x1x4x8xf16>
      //CHECK: xetile.store_tile %[[R10]],  %[[R9]] : vector<1x1x4x8xf16>, !xetile.tile<4x8xf16, #xetile.tile_attr<inner_blocks = [4, 8]>>
      xetile.store_tile %c, %s : vector<4x8xf16>, !xetile.tile<4x8xf16>
      gpu.return
    }

  //CHECK-LABEL: gpu.func @sg_gemm_with_preops_for_c
  //CHECK-SAME: (%[[arg0:.*]]: memref<32x128xf16>, %[[arg1:.*]]: memref<128x32xf16>, %[[arg2:.*]]: memref<32x32xf32>)
  gpu.func @sg_gemm_with_preops_for_c(%a: memref<32x128xf16>, %b: memref<128x32xf16>, %c: memref<32x32xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c128:.*]] = arith.constant 128 : index
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<1x2x32x16xf32>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant dense<0.0>: vector<32x32xf32>

    //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
    //CHECK: %[[r1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[r2:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]] iter_args(%[[arg4:.*]] = %[[r0]], %[[arg5:.*]] = %[[r1]], %[[arg6]] = %[[cst]]) -> (!xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<1x2x32x16xf32>) {
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      //CHECK: %[[r6:.*]] = xetile.tile_unpack %[[arg6]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf32> -> vector<32x32xf32>

      //CHECK: %[[r7:.*]] = xetile.load_tile %[[arg4]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[r8:.*]] = xetile.tile_unpack %[[r7]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[r9:.*]] = xetile.load_tile %[[arg5]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
      //CHECK: %[[r10:.*]] = xetile.tile_unpack %[[r9]] {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[r11:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c32]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      //CHECK: %[[r12:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c32]],  %[[c0]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, index, index -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]: !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]: !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>

      //CHECK: %[[r13:.*]] = xetile.tile_pack %[[r6]] {inner_blocks = array<i64: 1, 16>}  : vector<32x32xf32> -> vector<32x2x1x16xf32>
      //CHECK: %[[r14:.*]] = xetile.tile_pack %[[r6]] {inner_blocks = array<i64: 1, 16>}  : vector<32x32xf32> -> vector<32x2x1x16xf32>
      //CHECK: %[[r15:.*]] = arith.addf %[[r13]], %[[r14]] : vector<32x2x1x16xf32>
      %5 = arith.addf %c_value, %c_value: vector<32x32xf32>

      //CHECK: %[[r16:.*]] = xetile.tile_unpack %[[r15]] {inner_blocks = array<i64: 1, 16>}  : vector<32x2x1x16xf32> -> vector<32x32xf32>
      //CHECK: %[[r17:.*]] = xetile.tile_pack %[[r8]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf16> -> vector<4x2x8x16xf16>
      //CHECK: %[[r18:.*]] = xetile.tile_pack %[[r10]] {inner_blocks = array<i64: 16, 16>}  : vector<32x32xf16> -> vector<2x2x16x16xf16>
      //CHECK: %[[r19:.*]] = xetile.tile_pack %[[r16]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf32> -> vector<4x2x8x16xf32>
      //CHECK: %[[r20:.*]] = xetile.tile_mma %[[r17]], %[[r18]], %[[r19]] : vector<4x2x8x16xf16>, vector<2x2x16x16xf16>, vector<4x2x8x16xf32> -> vector<4x2x8x16xf32>
      %c_new_value = xetile.tile_mma %3, %4, %5: vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
      //CHECK: %[[r21:.*]] = xetile.tile_unpack %[[r20]] {inner_blocks = array<i64: 8, 16>}  : vector<4x2x8x16xf32> -> vector<32x32xf32>
      //CHECK: %[[r22:.*]] = xetile.tile_pack %[[r21]] {inner_blocks = array<i64: 32, 16>}  : vector<32x32xf32> -> vector<1x2x32x16xf32>
      //CHECK: scf.yield %[[r11]], %[[r12]], %[[r22]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>, vector<1x2x32x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }

    //CHECK: %[[r3:.*]] = xetile.tile_unpack %[[r2]]#2 {inner_blocks = array<i64: 32, 16>}  : vector<1x2x32x16xf32> -> vector<32x32xf32>
    //CHECK: %[[r4:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>

    //CHECK: %[[r5:.*]] = xetile.tile_pack %[[r3]] {inner_blocks = array<i64: 8, 16>}  : vector<32x32xf32> -> vector<4x2x8x16xf32>
    //CHECK: xetile.store_tile %[[r5]],  %[[r4]] : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>
  	gpu.return
  }

    //CHECK-LABEL: gpu.func @sglevel_softmax_dim_0
    //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
    gpu.func @sglevel_softmax_dim_0(%a: memref<1024x1024xf16>) {
      //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

      //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
      //CHECK: %[[r2:.*]] = xetile.tile_unpack %[[r1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
      %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>

      //CHECK: %[[r3:.*]] = xetile.tile_pack %[[r2]] {inner_blocks = array<i64: 1, 32>}  : vector<32x64xf16> -> vector<32x2x1x32xf16>
      //CHECK: %[[r4:.*]] = math.exp %[[r3]] : vector<32x2x1x32xf16>
      %3 = math.exp %2: vector<32x64xf16>

      //CHECK: %[[r5:.*]] = xetile.reduction <add>, %[[r4]] [0, 2] : vector<32x2x1x32xf16> -> vector<1x2x1x32xf16>
      %4 = xetile.reduction <add>, %3 [0]: vector<32x64xf16> -> vector<1x64xf16>

      //CHECK: %[[r6:.*]] = xetile.broadcast %[[r5]] [0, 2] : vector<1x2x1x32xf16> -> vector<32x2x1x32xf16>
      %5 = xetile.broadcast %4 [0]: vector<1x64xf16> -> vector<32x64xf16>

      //CHECK: %[[r7:.*]] = arith.divf %[[r4]], %[[r6]] : vector<32x2x1x32xf16>
      //CHECK: %[[r8:.*]] = xetile.tile_unpack %[[r7]] {inner_blocks = array<i64: 1, 32>}  : vector<32x2x1x32xf16> -> vector<32x64xf16>
      %6 = arith.divf %3, %5: vector<32x64xf16>

      //CHECK: %[[r9:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

      //CHECK: %[[r10:.*]] = xetile.tile_pack %[[r8]] {inner_blocks = array<i64: 8, 32>}  : vector<32x64xf16> -> vector<4x2x8x32xf16>
      //CHECK: xetile.store_tile %[[r10]],  %[[r9]] : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
      gpu.return
    }


    //CHECK-LABEL: gpu.func @sglevel_softmax_dim_1
    //CHECK-SAME: (%[[arg0:.*]]: memref<1024x1024xf16>)
    gpu.func @sglevel_softmax_dim_1(%a: memref<1024x1024xf16>) {
      //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

      //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
      //CHECK: %[[r2:.*]] = xetile.tile_unpack %[[r1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
      %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>

      //CHECK: %[[r3:.*]] = xetile.tile_pack %[[r2]] {inner_blocks = array<i64: 1, 32>}  : vector<32x64xf16> -> vector<32x2x1x32xf16>
      //CHECK: %[[r4:.*]] = math.exp %[[r3]] : vector<32x2x1x32xf16>
      %3 = math.exp %2: vector<32x64xf16>

      //CHECK: %[[r5:.*]] = xetile.reduction <add>, %[[r4]] [1, 3] : vector<32x2x1x32xf16> -> vector<32x1x1x1xf16>
      %4 = xetile.reduction <add>, %3 [1]: vector<32x64xf16> -> vector<32x1xf16>

      //CHECK: %[[r6:.*]] = xetile.broadcast %[[r5]] [1, 3] : vector<32x1x1x1xf16> -> vector<32x2x1x32xf16>
      %5 = xetile.broadcast %4 [1]: vector<32x1xf16> -> vector<32x64xf16>

      //CHECK: %[[r7:.*]] = arith.divf %[[r4]], %[[r6]] : vector<32x2x1x32xf16>
      //CHECK: %[[r8:.*]] = xetile.tile_unpack %[[r7]] {inner_blocks = array<i64: 1, 32>}  : vector<32x2x1x32xf16> -> vector<32x64xf16>
      %6 = arith.divf %3, %5: vector<32x64xf16>

      //CHECK: %[[r9:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      %7 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

      //CHECK: %[[r10:.*]] = xetile.tile_pack %[[r8]] {inner_blocks = array<i64: 8, 32>}  : vector<32x64xf16> -> vector<4x2x8x32xf16>
      //CHECK: xetile.store_tile %[[r10]],  %[[r9]] : vector<4x2x8x32xf16>, !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      xetile.store_tile %6, %7: vector<32x64xf16>, !xetile.tile<32x64xf16>
      gpu.return
    }


    //CHECK-LABEL: gpu.func @sglevel_softmax_transpose
    //CHECK-SAME(%[[arg0:.*]]: memref<1024x1024xf16>)
    gpu.func @sglevel_softmax_transpose(%a: memref<1024x1024xf16>) {
      //CHECK: %[[r0:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %1 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<32x64xf16>

      //CHECK: %[[r1:.*]] = xetile.load_tile %[[r0]] : !xetile.tile<32x64xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x2x32x32xf16>
      //CHECK: %[[r2:.*]] = xetile.tile_unpack %[[r1]] {inner_blocks = array<i64: 32, 32>}  : vector<1x2x32x32xf16> -> vector<32x64xf16>
      %2 = xetile.load_tile %1: !xetile.tile<32x64xf16> -> vector<32x64xf16>

      //CHECK: %[[r3:.*]] = xetile.tile_pack %[[r2]] {inner_blocks = array<i64: 1, 32>}  : vector<32x64xf16> -> vector<32x2x1x32xf16>
      //CHECK: %[[r4:.*]] = math.exp %[[r3]] : vector<32x2x1x32xf16>
      %3 = math.exp %2: vector<32x64xf16>

      //CHECK: %[[r5:.*]] = xetile.reduction <add>, %[[r4]] [1, 3] : vector<32x2x1x32xf16> -> vector<32x1x1x1xf16>
      %4 = xetile.reduction <add>, %3 [1]: vector<32x64xf16> -> vector<32x1xf16>

      //CHECK: %[[r6:.*]] = xetile.broadcast %[[r5]] [1, 3] : vector<32x1x1x1xf16> -> vector<32x2x1x32xf16>
      %5 = xetile.broadcast %4 [1]: vector<32x1xf16> -> vector<32x64xf16>

      //CHECK: %[[r7:.*]] = arith.divf %[[r4]], %[[r6]] : vector<32x2x1x32xf16>
      //CHECK: %[[r8:.*]] = xetile.tile_unpack %[[r7]] {inner_blocks = array<i64: 1, 32>}  : vector<32x2x1x32xf16> -> vector<32x64xf16>
      %6 = arith.divf %3, %5: vector<32x64xf16>

      //CHECK: %[[r9:.*]] = xetile.tile_pack %[[r8]] {inner_blocks = array<i64: 8, 16>}  : vector<32x64xf16> -> vector<4x4x8x16xf16>
      //CHECK: %[[r10:.*]] = xetile.transpose %[[r9]], [1, 0, 3, 2] : vector<4x4x8x16xf16> -> vector<4x4x16x8xf16>
      //CHECK: %[[r11:.*]] = xetile.tile_unpack %[[r10]] {inner_blocks = array<i64: 16, 8>}  : vector<4x4x16x8xf16> -> vector<64x32xf16>
      %7 = xetile.transpose %6, [1, 0]: vector<32x64xf16> -> vector<64x32xf16>

      //CHECK: %[[r12:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      %8 = xetile.init_tile %a[0, 0] : memref<1024x1024xf16> -> !xetile.tile<64x32xf16>

      //CHECK: %[[r13:.*]] = xetile.tile_pack %[[r11]] {inner_blocks = array<i64: 8, 32>}  : vector<64x32xf16> -> vector<8x1x8x32xf16>
      //CHECK: xetile.store_tile %[[r13]],  %[[r12]] : vector<8x1x8x32xf16>, !xetile.tile<64x32xf16, #xetile.tile_attr<inner_blocks = [8, 32]>>
      xetile.store_tile %7, %8: vector<64x32xf16>, !xetile.tile<64x32xf16>
      gpu.return
    }

}
