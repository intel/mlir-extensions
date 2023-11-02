// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s
// CHECK-LABEL: func @test_create_nd_tdesc_vc_0({{.*}}) {
func.func @test_create_nd_tdesc_vc_0(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %2 = xegpu.create_nd_tdesc %src[2, 4] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_vc_1({{.*}}) {
func.func @test_create_nd_tdesc_vc_1(%src: memref<24x32xf32>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg1, %arg2]
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_vc_2({{.*}}) {
func.func @test_create_nd_tdesc_vc_2(%src: ui64, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: ui64 -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {mode = vc} : ui64 -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_vc_3({{.*}}) {
func.func @test_create_nd_tdesc_vc_3(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {mode = vc} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_vc_4({{.*}}) {
func.func @test_create_nd_tdesc_vc_4(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {mode = vc, boundary_check = true} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_vc_5({{.*}}) {
func.func @test_create_nd_tdesc_vc_5(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {mode = vc, memory_scope = slm} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_vc_6({{.*}}) {
func.func @test_create_nd_tdesc_vc_6(%src: memref<?x?xf32>, %w : index, %h : index, %x : index, %y : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg3, %arg4], [%arg2, %arg1], [%arg1, %c1]
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%x, %y], [%h, %w], [%w, %c1] {mode = vc, memory_scope = slm, boundary_check = true} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_vc_7({{.*}}) {
func.func @test_create_nd_tdesc_vc_7(%src: memref<1024xf32>, %offset : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<1024xf32> -> !xegpu.tensor_desc<16xf32>
  %1 = xegpu.create_nd_tdesc %src[%offset] {mode = vc} : memref<1024xf32> -> !xegpu.tensor_desc<16xf32>
  return
}


// CHECK-LABEL: func @test_create_nd_tdesc_vc_8({{.*}}) {
func.func @test_create_nd_tdesc_vc_8(%src: memref<?x?xf32>, %w : index, %h : index, %x : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[8, %x], [%h, %w], [%w, %c1] {mode = vc, memory_scope = slm, boundary_check = true} : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// CHECK-LABEL: func @test_create_nd_tdesc_vc_9({{.*}}) {
func.func @test_create_nd_tdesc_vc_9(%src: memref<?x?xf32>, %w : index, %h : index, %x : index) {
  %c1 = arith.constant 1 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {mode = simt, memory_scope = slm, boundary_check = true}
  // CHECK-SAME: !xegpu.tensor_desc<64x128xf32, #xegpu.xe_map<wg = {sg_layout = [2, 2], sg_data = [32, 64]}, sg = {mma_block_size = [8, 16], wi_layout = [1, 16], wi_data = [8, 1]}>>
  %1 = xegpu.create_nd_tdesc %src[8, %x], [%h, %w], [%w, %c1] {memory_scope = slm, boundary_check = true} : memref<?x?xf32>
                            -> !xegpu.tensor_desc<64x128xf32, #xegpu.xe_map<wg = {sg_layout=[2, 2], sg_data=[32, 64]}, sg = {mma_block_size=[8, 16], wi_layout=[1, 16], wi_data=[8, 1]}>>
  return
}
