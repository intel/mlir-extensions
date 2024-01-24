// RUN: imex-opt --xetile-tiling --convert-xetile-to-xegpu --canonicalize %s -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @test
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
//   CHECK-DAG: %[[C24:.*]] = arith.constant 24 : index
//       CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %arg0[%[[C0]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %arg0[%[[C8]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %arg0[%[[C16]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T3:.*]] = xegpu.create_nd_tdesc %arg0[%[[C24]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[R0:.*]] = xegpu.load_nd %[[T0]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %[[R1:.*]] = xegpu.load_nd %[[T1]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %[[R2:.*]] = xegpu.load_nd %[[T2]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %[[R3:.*]] = xegpu.load_nd %[[T3]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[R0]], %[[R1]], %[[R2]], %[[R3]]
func.func @test_init_tile(%A: memref<128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c16  = arith.constant 16 : index
    %tile = xetile.init_tile %A[%c0, %c0] : memref<128x64xf16> -> !xetile.tile<32x16xf16>
    %res = xetile.load_tile %tile : !xetile.tile<32x16xf16> -> vector<32x16xf16>
    "test.use"(%res) : (vector<32x16xf16>) -> ()
    return
}

// -----

// CHECK-LABEL: @test
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
//   CHECK-DAG: %[[C24:.*]] = arith.constant 24 : index
//       CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %arg0[%[[C0]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %arg0[%[[C8]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %arg0[%[[C16]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T3:.*]] = xegpu.create_nd_tdesc %arg0[%[[C24]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[R0:.*]] = xegpu.load_nd %[[T0]] {mode = vc, transpose = [0, 1], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %[[R1:.*]] = xegpu.load_nd %[[T1]] {mode = vc, transpose = [0, 1], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %[[R2:.*]] = xegpu.load_nd %[[T2]] {mode = vc, transpose = [0, 1], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %[[R3:.*]] = xegpu.load_nd %[[T3]] {mode = vc, transpose = [0, 1], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[R0]], %[[R1]], %[[R2]], %[[R3]]
func.func @test_init_tile(%A: memref<128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c16  = arith.constant 16 : index
    %tile = xetile.init_tile %A[%c0, %c0] : memref<128x64xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<order = [1, 0]>>
    %res = xetile.load_tile %tile : !xetile.tile<32x16xf16, #xetile.tile_attr<order = [1, 0]>> -> vector<32x16xf16>
    "test.use"(%res) : (vector<32x16xf16>) -> ()
    return
}

// -----

// CHECK-LABEL: @test
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
//   CHECK-DAG: %[[C24:.*]] = arith.constant 24 : index
//       CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %arg0[%[[C0]], %[[C0]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T1:.*]] = xegpu.create_nd_tdesc %arg0[%[[C0]], %[[C8]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %arg0[%[[C0]], %[[C16]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[T3:.*]] = xegpu.create_nd_tdesc %arg0[%[[C0]], %[[C24]]] {mode = vc} : memref<128x64xf16> -> !xegpu.tensor_desc<8x16xf16>
//       CHECK: %[[R0:.*]] = xegpu.load_nd %[[T0]] {mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<16x8xf16>
//       CHECK: %[[R1:.*]] = xegpu.load_nd %[[T1]] {mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<16x8xf16>
//       CHECK: %[[R2:.*]] = xegpu.load_nd %[[T2]] {mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<16x8xf16>
//       CHECK: %[[R3:.*]] = xegpu.load_nd %[[T3]] {mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<16x8xf16>
//       CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[R0]], %[[R1]], %[[R2]], %[[R3]]
func.func @test_init_tile(%A: memref<128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c16  = arith.constant 16 : index
    %tile = xetile.init_tile %A[%c0, %c0] : memref<128x64xf16> -> !xetile.tile<32x16xf16, #xetile.tile_attr<order = [0, 1]>>
    %res = xetile.load_tile %tile : !xetile.tile<32x16xf16, #xetile.tile_attr<order = [0, 1]>> -> vector<16x32xf16>
    "test.use"(%res) : (vector<16x32xf16>) -> ()
    return
}
