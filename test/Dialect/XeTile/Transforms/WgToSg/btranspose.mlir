// RUN: imex-opt --split-input-file --xetile-wg-to-sg --cse %s -verify-diagnostics | FileCheck %s

gpu.module @test_gemm_btranspose{
  //CHECK:  gpu.func @test_kernel(%[[arg0:.*]]: memref<16384x12288xf16>, %[[arg1:.*]]: memref<1536x12288xf16>, %[[arg2:.*]]:  memref<16384x1536xf32>)
  gpu.func @test_kernel(%arg0: memref<16384x12288xf16>, %arg1: memref<1536x12288xf16>, %arg2: memref<16384x1536xf32>) {

    //CHECK: %[[c8:.*]] = arith.constant 8 : index
    //CHECK: %[[c32:.*]] = arith.constant 32 : index
    //CHECK: %[[c4:.*]] = arith.constant 4 : index
    //CHECK: %[[c1:.*]] = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c8_0 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c8, %arg10 = %c32, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c8_0, %arg13 = %c4, %arg14 = %c1) {

      //CHECK: %[[c12288:.*]] = arith.constant 12288 : index
      //CHECK: %[[c2:.*]] = arith.constant 2 : index
      //CHECK: %[[c2048:.*]] = arith.constant 2048 : index
      //CHECK: %[[c256:.*]] = arith.constant 256 : index
      //CHECK: %[[c1024:.*]] = arith.constant 1024 : index
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
      //CHECK: %[[R0:.*]] = gpu.block_id  x
      //CHECK: %[[R1:.*]] = gpu.block_id  y

      //CHECK: %[[R2:.*]] = arith.divsi %[[R1]], %c8 : index
      //CHECK: %[[R3:.*]] = arith.remsi %[[R1]], %c8 : index
      //CHECK: %[[R4:.*]] = arith.muli %[[R3]], %[[c256]] : index

      //CHECK: %[[R6:.*]] = gpu.subgroup_id : index
      //CHECK: %[[c64:.*]] = arith.constant 64 : index

      %c12288 = arith.constant 12288 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %c32_2 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} dense<0.000000e+00> : vector<256x256xf32>
      %c8_3 = arith.constant 8 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.divsi %block_id_y, %c8_3 : index
      %1 = arith.remsi %block_id_y, %c8_3 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.muli %1, %c256 : index
      %4 = arith.muli %block_id_x, %c2048 : index
      %5 = arith.muli %0, %c256 : index
      %6 = arith.addi %4, %5 : index
      %7 = xetile.init_tile %arg2[%6, %3] : memref<16384x1536xf32> -> !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>, memory_space = 0 : i32>>
      %8 = arith.muli %block_id_x, %c2048 : index
      %9 = arith.muli %0, %c256 : index
      %10 = arith.addi %8, %9 : index
      %11 = xetile.init_tile %arg0[%10, %c0] : memref<16384x12288xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>>

      //CHECK: %[[R7:.*]] = index.divu %[[R6]], %[[c8]]
      //CHECK: %[[R8:.*]] = index.remu %[[R6]], %[[c8]]
      //CHECK: %[[R9:.*]] = index.add %[[R8]], %[[c0]]
      //CHECK: %[[R10:.*]] = index.remu %[[R9]], %[[c4]]
      //CHECK: %[[R11:.*]] = index.mul %[[R10]], %[[c64]]
      //CHECK: %[[R12:.*]] = index.add %[[R4]], %[[R11]]
      //CHECK: %[[R13:.*]] = index.add %[[R7]], %[[c0]]
      //CHECK: %[[R14:.*]] = index.remu %[[R13]], %[[c1]]
      //CHECK: %[[R15:.*]] = index.mul %[[R14]], %[[c32]]
      //CHECK: %[[R16:.*]] = index.add %[[R15]], %[[c0]]

      //CHECK: %[[INITTILE:.*]] = xetile.init_tile %[[arg1]][%[[R12]], %[[R16]]] : memref<1536x12288xf16> -> !xetile.tile<64x32xf16>
      %12 = xetile.init_tile %arg1[%2, %c0] : memref<1536x12288xf16> -> !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32>>
      %13:2 = scf.for %arg15 = %c0 to %c2 step %c1_1 iter_args(%arg16 = %7, %arg17 = %11) -> (!xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>, memory_space = 0 : i32>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>>) {
        %14 = xetile.update_tile_offset %arg17, [%c1024,  %c0] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>>
        %15 = xetile.update_tile_offset %arg16, [%c1024,  %c0] : !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>, memory_space = 0 : i32>>
        %16:3 = scf.for %arg18 = %c0 to %c12288 step %c32_2 iter_args(%arg19 = %cst, %arg20 = %arg17, %arg21 = %12) -> (vector<256x256xf32>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32>>) {
          %18 = xetile.update_tile_offset %arg21, [%c0,  %c32_2] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32>>
          %19 = xetile.update_tile_offset %arg20, [%c0,  %c32_2] : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>>
          %20 = xetile.load_tile %arg20 {padding = 0.000000e+00 : f32}  : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>> -> vector<256x32xf16>
          %21 = math.exp %20 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x32xf16>
          %22 = xetile.load_tile %arg21 {padding = 0.000000e+00 : f32}  : !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32>> -> vector<256x32xf16>
          //CHECK: %[[TRANSPOSE:.*]] vector.transpose {{%.*}}, [1, 0] : vector<64x32xf16> to vector<32x64xf16>
          %23 = vector.transpose %22, [1, 0] {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x32xf16> to vector<32x256xf16>
          %24 = math.exp %23 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<32x256xf16>
          xegpu.compile_hint
          %25 = xetile.tile_mma %21, %24, %cst {wg_map_a =#xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>, wg_map_b =#xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>, wg_map_c =#xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
          xegpu.compile_hint
          %26 = arith.addf %arg19, %25 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x256xf32>
          scf.yield %26, %19, %18 : vector<256x256xf32>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [64, 32]>, memory_space = 0 : i32>>
        }
        %17 = math.exp %16#0 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 64]>} : vector<256x256xf32>
        xetile.store_tile %17,  %arg16 : vector<256x256xf32>, !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>, memory_space = 0 : i32>>
        scf.yield %15, %14 : !xetile.tile<256x256xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 64]>, memory_space = 0 : i32>>, !xetile.tile<256x32xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>, memory_space = 0 : i32>>
      }
      gpu.terminator
    }
    gpu.return
  }
}
