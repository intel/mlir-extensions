// RUN: imex-opt -imex-lower-memref-copy -allow-unregistered-dialect %s | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @copy_with_later_use(%arg0: memref<10x20xf32>) -> memref<10x20xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<10x20xf32>
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<10x20xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<10x20xf32>)
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<10x20xf32>
    memref.copy %alloc_0, %alloc_1 : memref<10x20xf32> to memref<10x20xf32>
    "some_use" (%alloc_0) {} : (memref<10x20xf32>) -> ()
    // CHECK-LABEL: func @copy_with_later_use
    // CHECK:       linalg.generic
    // CHECK:       linalg.generic
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<10x20xf32>) outs(%alloc_1 : memref<10x20xf32>) attrs =  {iterator_ranges = [10, 20]} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %out, %in : f32
      linalg.yield %0 : f32
    }
    return %alloc_1 : memref<10x20xf32>
  }
  func.func @copy_without_later_use(%arg0: memref<10x20xf32>) -> memref<10x20xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<10x20xf32>
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<10x20xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<10x20xf32>)
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<10x20xf32>
    memref.copy %alloc_0, %alloc_1 : memref<10x20xf32> to memref<10x20xf32>
    // CHECK-LABEL: func @copy_without_later_use
    // CHECK:       linalg.generic
    // CHECK-NOT:   linalg.generic
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<10x20xf32>) outs(%alloc_1 : memref<10x20xf32>) attrs =  {iterator_ranges = [10, 20]} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %out, %in : f32
      linalg.yield %0 : f32
    }
    return %alloc_1 : memref<10x20xf32>
  }
}
