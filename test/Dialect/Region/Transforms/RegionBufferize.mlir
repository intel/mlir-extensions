// RUN: imex-opt %s -region-bufferize --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @sharpy_jit() -> memref<16xi64, strided<[?], offset: ?>> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = region.env_region #region.gpu_env<device = "gpu"> -> tensor<16xi64> {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xi64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc : memref<16xi64>) {
      ^bb0(%out: i64):
        %3 = linalg.index 0 : index
        %4 = arith.index_cast %3 : index to i64
        %5 = arith.sitofp %4 : i64 to f64
        %6 = arith.addf %5, %cst : f64
        %7 = arith.fptosi %6 : f64 to i64
        linalg.yield %7 : i64
      }
      %2 = bufferization.to_tensor %alloc : memref<16xi64>
      region.env_region_yield %2 : tensor<16xi64>
    }
    %1 = bufferization.to_memref %0 : memref<16xi64, strided<[?], offset: ?>>
    return %1 : memref<16xi64, strided<[?], offset: ?>>
  }
}
// CHECK-LABEL: func.func @sharpy_jit() -> memref<16xi64, strided<[?], offset: ?>> attributes {llvm.emit_c_interface} {
// CHECK: region.env_region #region.gpu_env<device = "gpu"> -> memref<16xi64> {
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<16xi64>
// CHECK: linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc : memref<16xi64>) {
// CHECK: region.env_region_yield %4 : memref<16xi64>
// CHECK: return
// CHECK-SAME: : memref<16xi64, strided<[?], offset: ?>>
