// RUN: imex-opt %s -allow-unregistered-dialect -canonicalize --split-input-file | FileCheck %s

module {
  func.func @test_region(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c33_i64 = arith.constant 33 : i64
    %c22 = arith.constant 22 : index
    %c55_i64 = arith.constant 55 : i64
    %0 = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<16xf64> {
      %6 = "xarray.linspace"() {device = "XeGPU", team = 1 : i64} : () -> tensor<16xf64>
      region.env_region_yield %6 : tensor<16xf64>
    }
    %1 = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<16xf64> {
      %6 = "xarray.create"() {device = "XeGPU", dtype = 2 : i8, team = 1 : i64} : () -> tensor<16xf64>
      region.env_region_yield %6 : tensor<16xf64>
    }
    %2 = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<16xf64> {
      %6 = "xarray.subview"() : () -> tensor<16xf64>
      region.env_region_yield %6 : tensor<16xf64>
    }
    %3 = region.env_region #region.gpu_env<device = "XeGPUs"> -> tensor<16xf64> {
      %6 = "xarray.ewbin"() {op = 0 : i32} : () -> tensor<16xf64>
      region.env_region_yield %6 : tensor<16xf64>
    }
    %4 = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<f64> {
      %6 = "xarray.reduction"() {op = 4 : i32} : () -> tensor<f64>
      region.env_region_yield %6 : tensor<f64>
    }
    %5 = builtin.unrealized_conversion_cast %4 : tensor<f64> to i64
    return %5 : i64
  }
}
// CHECK-LABEL: func.func @test_region
// CHECK-COUNT-1: region.env_region
// CHECK: xarray.linspace
// CHECK: xarray.create
// CHECK: xarray.subview
// CHECK: xarray.ewbin
// CHECK: xarray.reduction
// CHECK-COUNT-1: region.env_region_yield %{{.*}}
// CHECK-NEXT: }
// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: return

// -----
func.func @test_merge_region1() {
  region.env_region "test" {
    "test.test1"() : () -> ()
  }
  region.env_region "test" {
    "test.test2"() : () -> ()
  }
  region.env_region "donotmerge" {
    "test.test3"() : () -> ()
  }
  return
}
// CHECK-LABEL: func @test_merge_region1
// CHECK-NEXT: region.env_region "test" {
// CHECK-NEXT: "test.test1"() : () -> ()
// CHECK-NEXT: "test.test2"() : () -> ()
// CHECK-NEXT: }
// CHECK-NEXT: region.env_region "donotmerge" {
// CHECK-NEXT: "test.test3"() : () -> ()
// CHECK-NEXT: }
// CHECK-NEXT: return
