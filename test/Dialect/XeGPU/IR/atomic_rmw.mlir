// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_atomic_rmw({{.*}}) {
func.func @test_atomic_rmw(%src: ui64, %offsets : vector<16 x index>, %value : vector<16x1xf32>, %mask : vector<16xi1>) {
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc} : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.atomic_rmw
  // CHECK-SAME: (vector<16x1xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1>)
  xegpu.atomic_rmw "addf" %value, %1, %mask {mode = vc} : (vector<16x1xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1>)

  return
}

// CHECK-LABEL: func @test_atomic_rmw_0({{.*}}) {
func.func @test_atomic_rmw_0(%src: ui64, %offsets : vector<16 x index>, %value : vector<16x2xf32>, %mask : vector<16xi1>) {
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 2}: ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>

  // CHECK: xegpu.atomic_rmw
  // CHECK-SAME: (vector<16x2xf32>, !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>, vector<16xi1>)
  xegpu.atomic_rmw "mulf" %value, %1, %mask {mode = vc} : (vector<16x2xf32>, !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>, vector<16xi1>)

  return
}

// CHECK-LABEL: func @test_atomic_rmw_1({{.*}}) {
func.func @test_atomic_rmw_1(%src: ui64, %offsets : vector<16 x index>, %value : vector<16x2xi32>, %mask : vector<16xi1>) {
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 2}: ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xi32, #xegpu.scattered>

  // CHECK: xegpu.atomic_rmw
  // CHECK-SAME: (vector<16x2xi32>, !xegpu.tensor_desc<16x2xi32, #xegpu.scattered>, vector<16xi1>)
  xegpu.atomic_rmw "andi" %value, %1, %mask {mode = vc} : (vector<16x2xi32>, !xegpu.tensor_desc<16x2xi32, #xegpu.scattered>, vector<16xi1>)

  return
}
