// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --canonicalize \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -o -| FileCheck %s
// CHECK-LABEL: gpu.module @test_kernel {
gpu.module @test_kernel {
  // CHECK: gpu.func @test_gemm(%[[A:.*]]: memref<1024x1024xf16>, %[[B:.*]]: memref<1024x1024xf16>, %[[C:.*]]: memref<1024x1024xf32>)
  gpu.func @test_gemm(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c64 : index
    %n = arith.muli %block_id_y, %c64 : index

    //CHECK: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    //CHECK-COUNT-8: xegpu.create_nd_tdesc %[[C]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<64x64xf32>
    //CHECK-COUNT-8: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>> -> vector<32x16xf32>
    %c_init_value = xetile.load_tile %c_init_tile : !xetile.tile<64x64xf32> -> vector<64x64xf32>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[A]][{{.*}}] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
    %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>
    //CHECK-COUNT-4: xegpu.create_nd_tdesc %[[B]][{{.*}}] : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
    %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<1024x1024xf16> -> !xetile.tile<64x64xf16>
    //CHECK-COUNT-32: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [8, 16], strides = [1, 1]} : vector<32x16xf32> to vector<8x16xf32>
    %out:3 = scf.for %k = %c0 to %c1024 step %c64
      iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
      -> (!xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>) {

      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      %a_value = xetile.load_tile %a_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>

      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      //CHECK: {{.*}} = xegpu.load_nd {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xf16>
      //CHECK-COUNT-2: {{.*}} = vector.extract {{.*}} : vector<32x16xf16> from vector<2x32x16xf16>
      %b_value = xetile.load_tile %b_tile : !xetile.tile<64x64xf16> -> vector<64x64xf16>

      //CHECK-COUNT-32: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [8, 16], strides = [1, 1]} : vector<32x16xf16> to vector<8x16xf16>
      //CHECK-COUNT-16: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = {{.*}}, sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
      //CHECK-COUNT-128: {{.*}} = xegpu.dpas {{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
      %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value : vector<64x64xf16>, vector<64x64xf16>, vector<64x64xf32> -> vector<64x64xf32>

      //CHECK-COUNT-8: {{.*}} = xegpu.update_nd_offset {{.*}} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = true>>
      %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c64] : !xetile.tile<64x64xf16>
      %b_next_tile = xetile.update_tile_offset %b_tile, [%c64, %c0] : !xetile.tile<64x64xf16>
      scf.yield %a_next_tile, %b_next_tile, %c_new_value
        : !xetile.tile<64x64xf16>, !xetile.tile<64x64xf16>, vector<64x64xf32>
    }
    //CHECK-COUNT-32: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<boundary_check = true>>
    xetile.store_tile %out#2, %c_init_tile: vector<64x64xf32>, !xetile.tile<64x64xf32>

    gpu.return
  }
}
