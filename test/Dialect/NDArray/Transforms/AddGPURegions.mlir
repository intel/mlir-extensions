// RUN: imex-opt --add-gpu-regions %s -verify-diagnostics -o -| FileCheck %s

#GPUENV = #ndarray.envs<#region.gpu_env<device = "XeGPU">>

func.func @test_region(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c33 = arith.constant 33 : i64
    %c22 = arith.constant 22 : index
    %v = arith.constant 55 : i64
    %s = arith.index_cast %arg0 : i64 to index
    %0 = ndarray.linspace %arg0 %arg1 %c33 false {device = "XeGPU", team = 1 : i64} : (i64, i64, i64) -> tensor<33xi64, #GPUENV>
    %1 = ndarray.create %c22 value %v {dtype = 2 : i8, device = "XeGPU", team = 1 : i64} : (index, i64) -> tensor<?xi64, #GPUENV>
    %10 = ndarray.subview %0[%c0][22][%c3] : tensor<33xi64, #GPUENV> to tensor<22xi64, #GPUENV>
    %o1 = tensor.empty() : tensor<22xi64, #GPUENV>
    %20 = linalg.add ins(%10, %1 : tensor<22xi64, #GPUENV>, tensor<?xi64, #GPUENV>) outs(%o1 : tensor<22xi64, #GPUENV>) -> tensor<22xi64, #GPUENV>
    %o2 = tensor.empty() : tensor<i64, #GPUENV>
    %21 = linalg.reduce { arith.addi } ins(%20 : tensor<22xi64, #GPUENV>) outs(%o2 : tensor<i64, #GPUENV>) dimensions = [0]
    %30 = builtin.unrealized_conversion_cast %21 : tensor<i64, #GPUENV> to i64
    ndarray.delete %0 : tensor<33xi64, #GPUENV>
    ndarray.delete %1 : tensor<?xi64, #GPUENV>
    return %30 : i64
}
  // CHECK-LABEL: func.func @test_region(
  // CHECK-SAME: [[varg0:%.*]]: i64, [[varg1:%.*]]: i64, [[varg2:%.*]]: i64) -> i64 {
    // CHECK-NEXT: [[vc0:%.*]] = arith.constant 0 : index
    // CHECK-NEXT: [[vc3:%.*]] = arith.constant 3 : index
    // CHECK-NEXT: [[vc33_i64:%.*]] = arith.constant 33 : i64
    // CHECK-NEXT: [[vc22:%.*]] = arith.constant 22 : index
    // CHECK-NEXT: [[vc55_i64:%.*]] = arith.constant 55 : i64
    // CHECK-NEXT: [[v0:%.*]] = arith.index_cast [[varg0]] : i64 to index
    // CHECK-NEXT: [[v1:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v9:%.*]] = ndarray.linspace [[varg0]] [[varg1]] [[vc33_i64]] false {device = "XeGPU", team = 1 : i64} : (i64, i64, i64) -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v9]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v2:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<?xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v9:%.*]] = ndarray.create [[vc22]] value [[vc55_i64]] {device = "XeGPU", dtype = 2 : i8, team = 1 : i64} : (index, i64) -> tensor<?xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v9]] : tensor<?xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v3:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v9:%.*]] = ndarray.subview [[v1]][[[vc0]]] [22] [[[vc3]]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> to tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v9]] : tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v4:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v9:%.*]] = tensor.empty() : tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v9]] : tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v5:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v9:%.*]] = linalg.add ins([[v3]], [[v2]] : tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>, tensor<?xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>) outs([[v4]] : tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>) -> tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v9]] : tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v6:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<i64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v9:%.*]] = tensor.empty() : tensor<i64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v9]] : tensor<i64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v7:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<i64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[vreduced:%.*]] = linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} } ins([[v5]] : tensor<22xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>) outs([[v6]] : tensor<i64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>) dimensions = [0] 
      // CHECK-NEXT: region.env_region_yield [[vreduced]] : tensor<i64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v8:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> i64 {
      // CHECK-NEXT: [[v9:%.*]] = builtin.unrealized_conversion_cast [[v7]] : tensor<i64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> to i64
      // CHECK-NEXT: region.env_region_yield [[v9]] : i64
    // CHECK: region.env_region #region.gpu_env<device = "XeGPU"> {
      // CHECK-NEXT: ndarray.delete [[v1]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: region.env_region #region.gpu_env<device = "XeGPU"> {
      // CHECK-NEXT: ndarray.delete [[v2]] : tensor<?xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: return [[v8]] : i64

// -----
func.func @test_copy() -> tensor<33xi64> {
    %c0 = arith.constant 0 : i64
    %c3 = arith.constant 3 : i64
    %c33 = arith.constant 33 : i64
    %0 = ndarray.linspace %c0 %c3 %c33 false {device = "XeGPU", team = 1 : i64} : (i64, i64, i64) -> tensor<33xi64, #GPUENV>
    %1 = ndarray.copy %0 : tensor<33xi64, #GPUENV> -> tensor<33xi64>
    %2 = ndarray.copy %1 : tensor<33xi64> -> tensor<33xi64, #GPUENV>
    %3 = ndarray.copy %2 : tensor<33xi64, #GPUENV> -> tensor<33xi64, #GPUENV>
    %4 = ndarray.copy %3 : tensor<33xi64, #GPUENV> -> tensor<33xi64>
    %5 = ndarray.copy %4 : tensor<33xi64> -> tensor<33xi64>
    return %5 : tensor<33xi64>
}
  // CHECK-LABEL: func.func @test_copy() -> tensor<33xi64> {
    // CHECK-NEXT: [[vc0_i64:%.*]] = arith.constant 0 : i64
    // CHECK-NEXT: [[vc3_i64:%.*]] = arith.constant 3 : i64
    // CHECK-NEXT: [[vc33_i64:%.*]] = arith.constant 33 : i64
    // CHECK-NEXT: [[v0:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v6:%.*]] = ndarray.linspace [[vc0_i64]] [[vc3_i64]] [[vc33_i64]] false {device = "XeGPU", team = 1 : i64} : (i64, i64, i64) -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v6]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v1:%.*]] = ndarray.copy [[v0]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> -> tensor<33xi64>
    // CHECK-NEXT: [[v2:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v6:%.*]] = ndarray.copy [[v1]] : tensor<33xi64> -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v6]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v3:%.*]] = region.env_region #region.gpu_env<device = "XeGPU"> -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> {
      // CHECK-NEXT: [[v6:%.*]] = ndarray.copy [[v2]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> -> tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
      // CHECK-NEXT: region.env_region_yield [[v6]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>>
    // CHECK: [[v4:%.*]] = ndarray.copy [[v3]] : tensor<33xi64, #ndarray.envs<#region.gpu_env<device = "XeGPU">>> -> tensor<33xi64>
    // CHECK-NEXT: [[v5:%.*]] = ndarray.copy [[v4]] : tensor<33xi64> -> tensor<33xi64>
    // CHECK-NEXT: return [[v5]] : tensor<33xi64>
