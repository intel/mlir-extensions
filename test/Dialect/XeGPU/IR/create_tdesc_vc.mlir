// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt %s | IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt -mlir-print-op-generic %s | IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt | FileCheck %s


// CHECK-LABEL: func @test_create_tdesc_vc({{.*}}) {
func.func @test_create_tdesc_vc(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered = true>>
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered = true>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_2({{.*}}) {
func.func @test_create_tdesc_vc_2(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<memory_scope =  slm, scattered = true>>
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index>
                            -> !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<memory_scope = slm, scattered = true>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_3({{.*}}) {
func.func @test_create_tdesc_vc_3(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: {chunk_size = 8 : i64}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size = 8}
                                          : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_4({{.*}}) {
func.func @test_create_tdesc_vc_4(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: {chunk_size = 2 : i64}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope =  slm, scattered = true>>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size = 2}
                        : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, scattered = true>>
  return
}


// CHECK-LABEL: func @test_create_tdesc_vc_5({{.*}}) {
func.func @test_create_tdesc_vc_5(%src: memref<?xf32>, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {chunk_size = 2 : i64}
  // CHECK-SAME: memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope =  slm, scattered = true>>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size = 2}
              : memref<?xf32>, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, scattered = true>>
  return
}
