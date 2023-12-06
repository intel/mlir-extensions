// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt %s | IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt -mlir-print-op-generic %s | IMEX_XEGPU_PRINT_DEFAULTS=true imex-opt | FileCheck %s


// CHECK-LABEL: func @test_create_tdesc_vc({{.*}}) {
func.func @test_create_tdesc_vc(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: {mode = vc, chunk_size_per_lane = 1}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, memory_scope = global, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc}: ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_2({{.*}}) {
func.func @test_create_tdesc_vc_2(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: {mode = vc, chunk_size_per_lane = 1}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, memory_scope = slm, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc} : ui64, vector<16 x index>
                            -> !xegpu.tensor_desc<16xf32, memory_scope = slm, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_3({{.*}}) {
func.func @test_create_tdesc_vc_3(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: {mode = vc, chunk_size_per_lane = 8}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, memory_scope = global, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 8}
                                          : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_4({{.*}}) {
func.func @test_create_tdesc_vc_4(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: {mode = vc, chunk_size_per_lane = 2}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, memory_scope = slm, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 2}
                        : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, memory_scope = slm, #xegpu.scattered>
  return
}


// CHECK-LABEL: func @test_create_tdesc_vc_5({{.*}}) {
func.func @test_create_tdesc_vc_5(%src: memref<?xf32>, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc, chunk_size_per_lane = 2}
  // CHECK-SAME: memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, memory_scope = slm, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 2}
              : memref<?xf32>, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, memory_scope = slm, #xegpu.scattered>
  return
}
