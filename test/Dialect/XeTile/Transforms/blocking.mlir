// RUN: imex-opt --split-input-file --xetile-blocking %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_load_tile(%[[arg0:.*]]: memref<32x32xf16>)
  //CHECK: %[[c0:.*]] = arith.constant 0 : index
  //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
  //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
  //CHECK: gpu.return
  gpu.func @sg_load_tile(%a: memref<32x32xf16>) {
    %c0 = arith.constant 0 : index
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  	gpu.return
  }

  //CHECK: gpu.func @sg_tiled_store(%[[arg0:.*]]: memref<32x32xf32>)
  //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<1x2x32x16xf32>
  //CHECK: %[[R0:.*]] = xetile.tile_unpack %[[cst]] { inner_blocks = [32, 16] }  : vector<1x2x32x16xf32> -> vector<32x32xf32>
  //CHECK: %[[R1:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  //CHECK: %[[R2:.*]] = xetile.tile_pack %[[R0]] { inner_blocks = [8, 16] }  : vector<32x32xf32> -> vector<4x2x8x16xf32>
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

    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
    //CHECK: %[[R2:.*]] = xetile.tile_unpack %[[R1]] { inner_blocks = [32, 32] }  : vector<1x1x32x32xf16> -> vector<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>

    //CHECK: %[[R3:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<32x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>>
  	%3 = xetile.init_tile %b[%c0, %c0] : memref<32x32xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R4:.*]] = xetile.load_tile %[[R3]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 16]>> -> vector<1x2x32x16xf16>
    //CHECK: %[[R5:.*]] = xetile.tile_unpack %[[R4]] { inner_blocks = [32, 16] }  : vector<1x2x32x16xf16> -> vector<32x32xf16>
    %4 = xetile.load_tile %3 : !xetile.tile<32x32xf16> -> vector<32x32xf16>

    //CHECK: %[[R6:.*]] = xetile.tile_pack %[[R2]] { inner_blocks = [8, 16] }  : vector<32x32xf16> -> vector<4x2x8x16xf16>
    //CHECK: %[[R7:.*]] = xetile.tile_pack %[[R5]] { inner_blocks = [16, 16] }  : vector<32x32xf16> -> vector<2x2x16x16xf16>
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

    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<1x2x32x16xf32>
    //CHECK: %[[R0:.*]] = xetile.tile_unpack %[[cst]] { inner_blocks = [32, 16] }  : vector<1x2x32x16xf32> -> vector<32x32xf32>
    %cst = arith.constant dense<0.0>: vector<32x32xf32>

    //CHECK: %[[R1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<32x128xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
  	%1 = xetile.init_tile %a[%c0, %c0] : memref<32x128xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R2:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x32xf16> -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
  	%2 = xetile.init_tile %b[%c0, %c0] : memref<128x32xf16> -> !xetile.tile<32x32xf16>

    //CHECK: %[[R3:.*]] = xetile.tile_pack %[[R0]] { inner_blocks = [32, 16] }  : vector<32x32xf32> -> vector<1x2x32x16xf32>
    //CHECK: %[[R4:.*]]:3 = scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c32]]
    //CHECK-SAME: iter_args(%[[arg4:.*]] = %[[R1]], %[[arg5:.*]] = %[[R2]], %[[arg6:.*]] = %[[R3]])
    //CHECK-SAME: !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>,
    //CHECK-SAME: !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, vector<1x2x32x16xf32>
    %out:3 = scf.for %k = %c0 to %c128 step %c32 iter_args(%a_tile = %1, %b_tile = %2, %c_value = %cst)
        -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
      //CHECK: %[[R8:.*]] = xetile.load_tile %[[arg4]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
      //CHECK: %[[R9:.*]] = xetile.tile_unpack %[[R8]] { inner_blocks = [32, 32] }  : vector<1x1x32x32xf16> -> vector<32x32xf16>
      %3 = xetile.load_tile %a_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[R10:.*]] = xetile.load_tile %[[arg5]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>> -> vector<1x1x32x32xf16>
      //CHECK: %[[R11:.*]] = xetile.tile_unpack %[[R10]] { inner_blocks = [32, 32] }  : vector<1x1x32x32xf16> -> vector<32x32xf16>
      %4 = xetile.load_tile %b_tile : !xetile.tile<32x32xf16> -> vector<32x32xf16>

      //CHECK: %[[R12:.*]] = xetile.update_tile_offset %[[arg4]], [%[[c0]],  %[[c32]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, index, index -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32]: !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>

      //CHECK: %[[R13:.*]] = xetile.update_tile_offset %[[arg5]], [%[[c32]],  %[[c0]]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, index, index -> !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c32, %c0]: !xetile.tile<32x32xf16>, index, index -> !xetile.tile<32x32xf16>

      //CHECK: %[[R14:.*]] = xetile.tile_pack %[[R9]] { inner_blocks = [8, 16] }  : vector<32x32xf16> -> vector<4x2x8x16xf16>
      //CHECK: %[[R15:.*]] = xetile.tile_pack %[[R11]] { inner_blocks = [16, 16] }  : vector<32x32xf16> -> vector<2x2x16x16xf16>
      //CHECK: %[[R16:.*]] = xetile.tile_unpack %[[arg6]] { inner_blocks = [32, 16] }  : vector<1x2x32x16xf32> -> vector<32x32xf32>
      //CHECK: %[[R17:.*]] = xetile.tile_pack %[[R16]] { inner_blocks = [8, 16] }  : vector<32x32xf32> -> vector<4x2x8x16xf32>
      //CHECK: %[[R18:.*]] = xetile.tile_mma %[[R14]], %[[R15]], %[[R17]] : vector<4x2x8x16xf16>, vector<2x2x16x16xf16>, vector<4x2x8x16xf32> -> vector<4x2x8x16xf32>
      %c_new_value = xetile.tile_mma %3, %4, %c_value:
        vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>

      //CHECK: %[[R19:.*]] = xetile.tile_unpack %[[R18]] { inner_blocks = [8, 16] }  : vector<4x2x8x16xf32> -> vector<32x32xf32>
      //CHECK: %[[R20:.*]] = xetile.tile_pack %[[R19]] { inner_blocks = [32, 16] }  : vector<32x32xf32> -> vector<1x2x32x16xf32>
      //CHECK: scf.yield %[[R12]], %[[R13]], %[[R20]] : !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, !xetile.tile<32x32xf16, #xetile.tile_attr<inner_blocks = [32, 32]>>, vector<1x2x32x16xf32>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
    }
    //CHECK: %[[REG5:.*]] = xetile.tile_unpack %[[R4]]#2 { inner_blocks = [32, 16] }  : vector<1x2x32x16xf32> -> vector<32x32xf32>

    //CHECK: %[[REG6:.*]] = xetile.init_tile %[[arg2]][%[[c0]], %[[c0]]] : memref<32x32xf32> -> !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
  	%c_tile = xetile.init_tile %c[%c0, %c0] : memref<32x32xf32> -> !xetile.tile<32x32xf32>

    //CHECK: %[[REG7:.*]] = xetile.tile_pack %[[REG5]] { inner_blocks = [8, 16] }  : vector<32x32xf32> -> vector<4x2x8x16xf32>
    //CHECK: xetile.store_tile %[[REG7]],  %[[REG6]] : vector<4x2x8x16xf32>, !xetile.tile<32x32xf32, #xetile.tile_attr<inner_blocks = [8, 16]>>
    xetile.store_tile %out#2, %c_tile: vector<32x32xf32>, !xetile.tile<32x32xf32>

    //CHECK: gpu.return
  	gpu.return
  }

    //CHECK: gpu.func @load_tile(%[[arg0:.*]]: memref<128x128xf16>)
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<85x76xf16, #xetile.tile_attr<inner_blocks = [17, 19]>>
    //CHECK: %[[R1:.*]] = xetile.load_tile %[[R0]] { padding = 0.000000e+00 : f32 }  : !xetile.tile<85x76xf16, #xetile.tile_attr<inner_blocks = [17, 19]>> -> vector<5x4x17x19xf16>
    //CHECK: gpu.return
    gpu.func @load_tile(%a: memref<128x128xf16>) {
      %c0 = arith.constant 0 : index
  	  %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<85x76xf16>
      %2 = xetile.load_tile %1 : !xetile.tile<85x76xf16> -> vector<17x5x19x4xf16>
  	  gpu.return
    }

    //CHECK: gpu.func @store_tile(%[[arg0:.*]]: memref<128x128xf32>)
    //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<2x19x22x2xf32>
    //CHECK: %[[R0:.*]] = xetile.tile_unpack %[[cst]] { inner_blocks = [22, 2] }  : vector<2x19x22x2xf32> -> vector<44x38xf32>
    //CHECK: %[[R1:.*]] = xetile.init_tile %[[arg0]][0, 0] : memref<128x128xf32> -> !xetile.tile<44x38xf32, #xetile.tile_attr<inner_blocks = [4, 2]>>
    //CHECK: %[[R2:.*]] = xetile.tile_pack %[[R0]] { inner_blocks = [4, 2] }  : vector<44x38xf32> -> vector<11x19x4x2xf32>
    //CHECK: xetile.store_tile %[[R2]],  %[[R1]] : vector<11x19x4x2xf32>, !xetile.tile<44x38xf32, #xetile.tile_attr<inner_blocks = [4, 2]>>
    //CHECK: gpu.return
	  gpu.func @store_tile(%a: memref<128x128xf32>) {
		  %result = arith.constant dense<0.0>: vector<44x38xf32>
		  %1 = xetile.init_tile %a[0, 0] : memref<128x128xf32> -> !xetile.tile<44x38xf32>
		  xetile.store_tile %result, %1: vector<44x38xf32>, !xetile.tile<44x38xf32>
		  gpu.return
	  }

    //CHECK: gpu.func @tile_mma(%[[arg0:.*]]: memref<128x128xf16>, %[[arg1:.*]]: memref<128x128xf16>)
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    //CHECK: %[[R0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<90x76xf16, #xetile.tile_attr<inner_blocks = [30, 19]>>
    //CHECK: %[[R1:.*]] = xetile.init_tile %[[arg1]][%[[c0]], %[[c0]]] : memref<128x128xf16> -> !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1], inner_blocks = [4, 30]>>
    gpu.func @tile_mma(%a: memref<128x128xf16>, %b: memref<128x128xf16>) {
      %c0 = arith.constant 0 : index
  	  %1 = xetile.init_tile %a[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<90x76xf16>
      %2 = xetile.init_tile %b[%c0, %c0] : memref<128x128xf16> -> !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1]>>
      %3 = xetile.load_tile %1 : !xetile.tile<90x76xf16> -> vector<90x76xf16>
      %4 = xetile.load_tile %2 : !xetile.tile<76x90xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<76x90xf16>
      %5 = xetile.tile_mma %3, %4: vector<90x76xf16>, vector<76x90xf16> -> vector<90x90xf32>
  	  gpu.return
    }
}
