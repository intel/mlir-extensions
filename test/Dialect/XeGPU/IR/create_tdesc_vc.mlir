// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt %s | IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt -mlir-print-op-generic %s | IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt | FileCheck %s


// CHECK-LABEL: func @test_create_tdesc_vc({{.*}}) {
func.func @test_create_tdesc_vc(%src: ui64) {
  // CHECK: xegpu.create_tdesc %{{.*}} : ui64 -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64 -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_2({{.*}}) {
func.func @test_create_tdesc_vc_2(%src: ui64) {
  // CHECK: xegpu.create_tdesc %{{.*}} : ui64 -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_scope =  slm>>
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64
                            -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<memory_scope = slm>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_3({{.*}}) {
func.func @test_create_tdesc_vc_3(%src: ui64) {
  // CHECK: xegpu.create_tdesc %{{.*}} : ui64 -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64
            -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_4({{.*}}) {
func.func @test_create_tdesc_vc_4(%src: ui64) {
  // CHECK: xegpu.create_tdesc %{{.*}} : ui64
  // CHECK-SAME: !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<memory_scope =  slm, chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : ui64
            -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<memory_scope = slm, chunk_size = 2>>
  return
}


// CHECK-LABEL: func @test_create_tdesc_vc_5({{.*}}) {
func.func @test_create_tdesc_vc_5(%src: memref<?xf32>) {
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: memref<?xf32>
  // CHECK-SAME: !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<memory_scope =  slm, chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] : memref<?xf32>
            -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<memory_scope = slm, chunk_size = 2>>
  return
}
