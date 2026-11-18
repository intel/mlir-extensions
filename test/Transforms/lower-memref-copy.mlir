// RUN: imex-opt -imex-lower-memref-copy -allow-unregistered-dialect %s | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @copy_with_later_use(%arg0: memref<10x20xf32>) -> memref<10x20xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() alignment = 128 : memref<10x20xf32>
    %alloc_0 = memref.alloc() alignment = 128 : memref<10x20xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<10x20xf32>)
    %alloc_1 = memref.alloc() alignment = 128 : memref<10x20xf32>
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
    %alloc = memref.alloc() alignment = 128 : memref<10x20xf32>
    %alloc_0 = memref.alloc() alignment = 128 : memref<10x20xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<10x20xf32>)
    %alloc_1 = memref.alloc() alignment = 128 : memref<10x20xf32>
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
  func.func @copy_target_used_before_copy() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : i64
    %c7 = arith.constant 7 : i64
    %alloc = memref.alloc() : memref<1xi64>
    %alloc_0 = memref.alloc() : memref<1xi64>
    memref.store %c5, %alloc_0[%c0] : memref<1xi64>
    memref.store %c7, %alloc[%c0] : memref<1xi64>
    %0 = memref.load %alloc_0[%c0] : memref<1xi64>
    memref.copy %alloc, %alloc_0 : memref<1xi64> to memref<1xi64>
    %1 = memref.load %alloc_0[%c0] : memref<1xi64>
    // CHECK-LABEL: func @copy_target_used_before_copy
    // CHECK:       %[[SRC:.*]] = memref.alloc() : memref<1xi64>
    // CHECK:       %[[DST:.*]] = memref.alloc() : memref<1xi64>
    // CHECK:       memref.store %{{.*}}, %[[DST]]
    // CHECK:       memref.store %{{.*}}, %[[SRC]]
    // CHECK:       memref.load %[[DST]]
    // CHECK:       linalg.generic {{.*}} ins(%[[SRC]] : memref<1xi64>) outs(%[[DST]] : memref<1xi64>)
    // CHECK:       memref.load %[[DST]]
    return %0, %1 : i64, i64
  }
}
