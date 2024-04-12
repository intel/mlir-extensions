// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking --xetile-block-aligning \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
gpu.module @test_kernel {
  //CHECK: gpu.func @sg_load_tile(%[[arg0:.*]]: memref<1024x1024xf16>, %[[arg1:.*]]: memref<1024x1024xf16>, %[[arg2:.*]]: memref<1024x1024xf32>) {
  gpu.func @sg_load_tile(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
    //CHECK: %[[c0:.*]] = arith.constant 0 : index
    %c0 = arith.constant 0 : index
    //CHECK: %[[c64:.*]] = arith.constant 64 : index
    %c64 = arith.constant 64 : index
    //CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c64]]] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<32x32xf16>
  	%1 = xetile.init_tile %a[%c0, %c64] : memref<1024x1024xf16> -> !xetile.tile<32x32xf16>
    //CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>
    %2 = xetile.load_tile %1 : !xetile.tile<32x32xf16> -> vector<32x32xf16>
  	gpu.return
  }
}
