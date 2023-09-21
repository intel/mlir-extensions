// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s


// CHECK-LABEL: func @test_create_tdesc({{.*}}) {
func.func @test_create_tdesc(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1 {memory_scope = global, chunk_size_per_lane = 1} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets: ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_2({{.*}}) {
func.func @test_create_tdesc_2(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1 {memory_scope = slm, chunk_size_per_lane = 1} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {memory_scope=slm}: ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_3({{.*}}) {
func.func @test_create_tdesc_3(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1 {memory_scope = global, chunk_size_per_lane = 8} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {chunk_size_per_lane = 8} : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_4({{.*}}) {
func.func @test_create_tdesc_4(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1 {memory_scope = slm, chunk_size_per_lane = 2} : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {memory_scope = slm, chunk_size_per_lane = 2} : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>
  return
}


// CHECK-LABEL: func @test_create_tdesc_5({{.*}}) {
func.func @test_create_tdesc_5(%src: memref<?xf32>, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {memory_scope = slm, chunk_size_per_lane = 2}
  // CHECK-SAME: memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {memory_scope = slm, chunk_size_per_lane = 2}
                    : memref<?xf32>, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>
  return
}
