// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -o -| FileCheck %s
gpu.module @test_kernel {
  //CHECK-LABEL: test_gemm
  //CHECK-SAME: %[[arg0:.*]]: memref<1024x1024xtf32>, %[[arg1:.*]]: memref<1024x1024xtf32>, %[[arg2:.*]]: memref<1024x1024xf32>
  gpu.func @test_gemm(%arg0: memref<1024x1024xtf32>, %arg1: memref<1024x1024xtf32>, %arg2: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index

    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y

    %0 = arith.muli %block_id_x, %c64 : index
    %1 = arith.muli %block_id_y, %c64 : index


    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %2 = xetile.init_tile %arg2[%0, %1] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>

    //CHECK-COUNT-2: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xf32>
    //CHECK-COUNT-8: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %3 = xetile.load_tile %2 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf32> -> vector<32x32xf32>

    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xtf32> -> !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
    %4 = xetile.init_tile %arg0[%0, %c0] : memref<1024x1024xtf32> -> !xetile.tile<32x32xtf32>

    //CHECK-COUNT-2: {{.*}} = xegpu.create_nd_tdesc %[[arg1]][{{.*}}] : memref<1024x1024xtf32> -> !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    %5 = xetile.init_tile %arg1[%c0, %1] : memref<1024x1024xtf32> -> !xetile.tile<32x32xtf32>

    //CHECK: {{.*}} = scf.for
    %6:3 = scf.for %arg3 = %c0 to %c1024 step %c64 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3) -> (!xetile.tile<32x32xtf32>, !xetile.tile<32x32xtf32>, vector<32x32xf32>) {
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x8xtf32>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x8xtf32> from vector<2x32x8xtf32>
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x8xtf32>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x8xtf32> from vector<2x32x8xtf32>
      //CHECK-COUNT-16: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [8, 8], strides = [1, 1]} : vector<32x8xtf32> to vector<8x8xtf32>
      %7 = xetile.load_tile %arg4 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xtf32> -> vector<32x32xtf32>

      //CHECK-COUNT-2: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>> -> vector<32x16xtf32>
      //CHECK-COUNT-8: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [8, 16], strides = [1, 1]} : vector<32x16xtf32> to vector<8x16xtf32>
      %8 = xetile.load_tile %arg5 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xtf32> -> vector<32x32xtf32>

      //CHECK-COUNT-32: {{.*}} = xegpu.dpas {{.*}} : vector<8x8xtf32>, vector<8x16xtf32>, vector<8x16xf32> -> vector<8x16xf32>
      %9 = xetile.tile_mma %7, %8, %arg6 : vector<32x32xtf32>, vector<32x32xtf32>, vector<32x32xf32> -> vector<32x32xf32>

      //CHECK-COUNT-2: {{.*}} = xegpu.update_nd_offset %{{.*}}, [{{.*}}] : !xegpu.tensor_desc<32x8xtf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 2 : i64, boundary_check = true>>
      %10 = xetile.update_tile_offset %arg4, [%c0,  %c64] : !xetile.tile<32x32xtf32>, index, index -> !xetile.tile<32x32xtf32>

      //CHECK-COUNT-2: {{.*}} = xegpu.update_nd_offset %{{.*}}, [{{.*}}] : !xegpu.tensor_desc<32x16xtf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
      %11 = xetile.update_tile_offset %arg5, [%c64,  %c0] : !xetile.tile<32x32xtf32>, index, index -> !xetile.tile<32x32xtf32>

      scf.yield %10, %11, %9 : !xetile.tile<32x32xtf32>, !xetile.tile<32x32xtf32>, vector<32x32xf32>
    }
    //CHECK-COUNT-8: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space =  global, array_length = 1 : i64, boundary_check = true>>
    xetile.store_tile %6#2,  %2 : vector<32x32xf32>, !xetile.tile<32x32xf32>
    gpu.return
  }
}
