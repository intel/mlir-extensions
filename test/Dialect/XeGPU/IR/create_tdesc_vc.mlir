// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s |  imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s |  imex-opt | FileCheck %s


// CHECK-LABEL: func @test_create_tdesc_vc({{.*}}) {
func.func @test_create_tdesc_vc(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_2({{.*}}) {
func.func @test_create_tdesc_vc_2(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.create_tdesc %src, %offsets :
        ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_3({{.*}}) {
func.func @test_create_tdesc_vc_3(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index>
            -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_4({{.*}}) {
func.func @test_create_tdesc_vc_4(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %arg0, %arg1 : ui64, vector<16xindex>
  // CHECK-SAME: !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16 x index>
            -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  return
}


// CHECK-LABEL: func @test_create_tdesc_vc_5({{.*}}) {
func.func @test_create_tdesc_vc_5(%src: memref<?xf32, 3>, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc {{.*}} : memref<?xf32, 3>, vector<16xindex>
  // CHECK-SAME: !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 2 : i64>>
  %1 = xegpu.create_tdesc %src, %offsets : memref<?xf32, 3>, vector<16 x index>
            -> !xegpu.tensor_desc<16x2xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 2>>
  return
}
