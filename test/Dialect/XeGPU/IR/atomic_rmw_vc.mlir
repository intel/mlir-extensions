// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @test_atomic_rmw({{.*}}) {
func.func @test_atomic_rmw(%src: ui64, %offsets : vector<16 x index>, %value : vector<16xf32>, %mask : vector<16xi1>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>

  // CHECK: xegpu.atomic_rmw
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>, vector<16xf32>
  xegpu.atomic_rmw addf %1, %mask, %value : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>, vector<16xf32> -> vector<16xf32>

  return
}

// CHECK-LABEL: func @test_atomic_rmw_0({{.*}}) {
func.func @test_atomic_rmw_0(%src: ui64, %offsets : vector<16 x index>, %value : vector<16x2xf32>, %mask : vector<16xi1>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>

  // CHECK: xegpu.atomic_rmw
  // CHECK-SAME: !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1>, vector<16x2xf32>
  xegpu.atomic_rmw mulf %1, %mask, %value : !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1>, vector<16x2xf32> -> vector<16x2xf32>

  return
}

// CHECK-LABEL: func @test_atomic_rmw_1({{.*}}) {
func.func @test_atomic_rmw_1(%src: ui64, %offsets : vector<16 x index>, %value : vector<16x2xi32>, %mask : vector<16xi1>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xi32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>

  // CHECK: xegpu.atomic_rmw
  // CHECK-SAME: !xegpu.tensor_desc<16x2xi32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>, vector<16xi1>, vector<16x2xi32>
  xegpu.atomic_rmw andi %1, %mask, %value : !xegpu.tensor_desc<16x2xi32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<16xi1>, vector<16x2xi32> -> vector<16x2xi32>

  return
}
