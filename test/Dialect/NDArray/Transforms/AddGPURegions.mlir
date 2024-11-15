// RUN: imex-opt --split-input-file --add-gpu-regions %s -verify-diagnostics -o -| FileCheck %s

func.func @test_region(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c33 = arith.constant 33 : i64
    %c22 = arith.constant 22 : index
    %v = arith.constant 55 : i64
    %s = arith.index_cast %arg0 : i64 to index
    %0 = ndarray.linspace %arg0 %arg1 %c33 false {device = "XeGPU", team = 1 : i64} : (i64, i64, i64) -> tensor<33xi64, #region.gpu_env<device = "XeGPU">>
    %1 = ndarray.create %c22 value %v {dtype = 2 : i8, device = "XeGPU", team = 1 : i64} : (index, i64) -> tensor<?xi64, #region.gpu_env<device = "XeGPU">>
    %10 = ndarray.subview %0[%c0][22][%c3] : tensor<33xi64, #region.gpu_env<device = "XeGPU">> to tensor<?xi64, #region.gpu_env<device = "XeGPU">>
    %20 = ndarray.ewbin %10, %1 {op = 0 : i32} : (tensor<?xi64, #region.gpu_env<device = "XeGPU">>, tensor<?xi64, #region.gpu_env<device = "XeGPU">>) -> tensor<?xi64, #region.gpu_env<device = "XeGPU">>
    %21 = ndarray.reduction %20 {op = 4 : i32} : tensor<?xi64, #region.gpu_env<device = "XeGPU">> -> tensor<i64, #region.gpu_env<device = "XeGPU">>
    %30 = builtin.unrealized_conversion_cast %21 : tensor<i64, #region.gpu_env<device = "XeGPU">> to i64
    ndarray.delete %0 : tensor<33xi64, #region.gpu_env<device = "XeGPU">>
    ndarray.delete %1 : tensor<?xi64, #region.gpu_env<device = "XeGPU">>
    return %30 : i64
}
// CHECK-LABEL: func.func @test_region
// CHECK: [[V0:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">> {
// CHECK-NEXT: ndarray.linspace
// CHECK-NEXT: region.env_region_yield
// CHECK: [[V1:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<?xi64, #region.gpu_env<device = "XeGPU">> {
// CHECK-NEXT: ndarray.create
// CHECK-NEXT: region.env_region_yield
// CHECK: [[V2:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<?xi64, #region.gpu_env<device = "XeGPU">> {
// CHECK-NEXT: ndarray.subview [[V0]]
// CHECK-NEXT: region.env_region_yield
// CHECK: [[V3:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<?xi64, #region.gpu_env<device = "XeGPU">> {
// CHECK-NEXT: ndarray.ewbin [[V2]], [[V1]]
// CHECK: [[V4:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<i64, #region.gpu_env<device = "XeGPU">> {
// CHECK-NEXT: ndarray.reduction [[V3]]
// CHECK-NEXT: region.env_region_yield
// CHECK-NEXT: }
// CHECK-NEXT: [[V5:%.*]] = builtin.unrealized_conversion_cast
// CHECK: region.env_region #region.gpu_env<device = "XeGPU"> {
// CHECK-NEXT: ndarray.delete [[V0]] : tensor<33xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: }
// CHECK-NEXT: region.env_region #region.gpu_env<device = "XeGPU"> {
// CHECK-NEXT: ndarray.delete [[V1]] : tensor<?xi64, #region.gpu_env<device = "XeGPU">>
// CHECK-NEXT: }
// CHECK-NEXT: return [[V5]]


// -----
func.func @test_copy() -> tensor<33xi64> {
    %c0 = arith.constant 0 : i64
    %c3 = arith.constant 3 : i64
    %c33 = arith.constant 33 : i64
    %0 = ndarray.linspace %c0 %c3 %c33 false {device = "XeGPU", team = 1 : i64} : (i64, i64, i64) -> tensor<33xi64, #region.gpu_env<device = "XeGPU">>
    %1 = ndarray.copy %0 : tensor<33xi64, #region.gpu_env<device = "XeGPU">> -> tensor<33xi64>
    %2 = ndarray.copy %1 : tensor<33xi64> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">>
    %3 = ndarray.copy %2 : tensor<33xi64, #region.gpu_env<device = "XeGPU">> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">>
    %4 = ndarray.copy %3 : tensor<33xi64, #region.gpu_env<device = "XeGPU">> -> tensor<33xi64>
    %5 = ndarray.copy %4 : tensor<33xi64> -> tensor<33xi64>
    return %5 : tensor<33xi64>
}
// CHECK-LABEL:  func.func @test_copy() -> tensor<33xi64> {
// CHECK: region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">> {
// CHECK: ndarray.linspace
// CHECK-SAME: -> tensor<33xi64, #region.gpu_env<device = "XeGPU">>
// CHECK: region.env_region_yield
// CHECK-SAME: tensor<33xi64, #region.gpu_env<device = "XeGPU">>
// CHECK: ndarray.copy
// CHECK-SAME: tensor<33xi64, #region.gpu_env<device = "XeGPU">> -> tensor<33xi64>
// CHECK: region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">> {
// CHECK: ndarray.copy
// CHECK-SAME: tensor<33xi64> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">>
// CHECK: region.env_region_yield
// CHECK-SAME: tensor<33xi64, #region.gpu_env<device = "XeGPU">>
// CHECK: region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">> {
// CHECK: ndarray.copy
// CHECK-SAME: tensor<33xi64, #region.gpu_env<device = "XeGPU">> -> tensor<33xi64, #region.gpu_env<device = "XeGPU">>
// CHECK: region.env_region_yield
// CHECK-SAME: tensor<33xi64, #region.gpu_env<device = "XeGPU">>
// CHECK: ndarray.copy
// CHECK-SAME: tensor<33xi64, #region.gpu_env<device = "XeGPU">> -> tensor<33xi64>
// CHECK: ndarray.copy
// CHECK-SAME: tensor<33xi64> -> tensor<33xi64>
// CHECK: return
// CHECK-SAME: tensor<33xi64>
