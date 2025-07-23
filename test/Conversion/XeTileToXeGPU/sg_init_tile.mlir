// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
  //CHECK: gpu.func @sg_init_tile(%[[arg0:.*]]: memref<1024x1024xf32>, %[[arg1:.*]]: memref<?x?xf32>, %[[arg2:.*]]: memref<?x128x96xf32>) {
	gpu.func @sg_init_tile(%a: memref<1024x1024xf32>, %b: memref<?x?xf32>, %c: memref<?x128x96xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index

    %result = arith.constant dense<0.0>: vector<32x32xf32>

    // CHECK: %[[SRC_AS_INDEX:.*]] = memref.extract_aligned_pointer_as_index {{.*}} : memref<1024x1024xf32> -> index
    %src_as_index = memref.extract_aligned_pointer_as_index %a : memref<1024x1024xf32> -> index
    // CHECK-NEXT: %[[SRC_AS_INT:.*]] = arith.index_cast %[[SRC_AS_INDEX]] : index to i64
    %src_as_int = arith.index_cast %src_as_index : index to i64

    //CHECK-COUNT-8: {{.*}} = xegpu.create_nd_tdesc %[[arg0]][{{.*}}] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<>>
    %static_memref_src = xetile.init_tile %a[0, 32] : memref<1024x1024xf32> -> !xetile.tile<32x32xf32>

    // CHECK-COUNT-8: {{.*}} = xegpu.create_nd_tdesc %[[arg2]][{{.*}}], shape : [%c1024, %c1024, %c1024], strides : [%c1024, %c1, %c1] : memref<?x128x96xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<>>
    %batch_dynamic_memref_src = xetile.init_tile %c[%c0, %c0, %c32],[%c1024, %c1024, %c1024], [%c1024, %c1, %c1]  : memref<?x128x96xf32> -> !xetile.tile<32x32xf32>

    // CHECK-COUNT-8: {{.*}} = xegpu.create_nd_tdesc %[[arg1]][{{.*}}], shape : [%c1024, %c1024], strides : [%c1024, %c1] : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<>>
    %dynamic_memref_src = xetile.init_tile %b[%c0, %c32],[%c1024, %c1024], [%c1024, %c1]  : memref<?x?xf32> -> !xetile.tile<32x32xf32>

    // CHECK-COUNT-8: {{.*}} = xegpu.create_nd_tdesc %[[SRC_AS_INT]][{{.*}}], shape : [%c1024, %c1024], strides : [%c1024, %c1] : i64 -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<>>
    %int_src = xetile.init_tile %src_as_int[%c0, %c32], [%c1024, %c1024], [%c1024, %c1]  : i64 -> !xetile.tile<32x32xf32>

    //CHECK-COUNT-32: xegpu.store_nd {{.*}} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<>>
		xetile.store_tile %result, %static_memref_src: vector<32x32xf32>, !xetile.tile<32x32xf32>
    xetile.store_tile %result, %dynamic_memref_src: vector<32x32xf32>, !xetile.tile<32x32xf32>
    xetile.store_tile %result, %batch_dynamic_memref_src: vector<32x32xf32>, !xetile.tile<32x32xf32>
    xetile.store_tile %result, %int_src: vector<32x32xf32>, !xetile.tile<32x32xf32>

		gpu.return
	}
}
