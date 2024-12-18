// RUN: imex-opt %s -one-shot-bufferize --split-input-file | FileCheck %s

module {
  func.func @test_bufferize() -> memref<16xi64, strided<[?], offset: ?>> {
    %c1_i64 = arith.constant 1 : i64
    %0 = region.env_region #region.gpu_env<device = "gpu"> -> tensor<16xi64> {
      %2 = bufferization.alloc_tensor() : tensor<16xi64>
      %3 = linalg.fill ins(%c1_i64 : i64) outs(%2 : tensor<16xi64>) -> tensor<16xi64>
      region.env_region_yield %3 : tensor<16xi64>
    }
    %1 = bufferization.to_memref %0 : memref<16xi64, strided<[?], offset: ?>>
    return %1 : memref<16xi64, strided<[?], offset: ?>>
  }
}
// CHECK-LABEL: func.func @test_bufferize() -> memref<16xi64, strided<[?], offset: ?>> {
// CHECK: [[R1:%.*]] = region.env_region #region.gpu_env<device = "gpu"> -> memref<16xi64> {
// CHECK-NEXT: [[V1:%.*]] = memref.alloc() {alignment = 64 : i64} : memref<16xi64>
// CHECK-NEXT: linalg.fill
// CHECK-NEXT: region.env_region_yield [[V1]] : memref<16xi64>
// CHECK: [[V2:%.*]] = memref.cast [[R1]] : memref<16xi64> to memref<16xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return [[V2]] : memref<16xi64, strided<[?], offset: ?>>
