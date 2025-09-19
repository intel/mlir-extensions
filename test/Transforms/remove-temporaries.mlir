// RUN: imex-opt -imex-remove-temporaries -allow-unregistered-dialect %s | FileCheck %s
// XFAIL: *
#map = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @ewbinop_inplace(%arg0: memref<64xi64, strided<[?], offset: ?>>, %arg1: memref<64xi64, strided<[?], offset: ?>>, %arg2: memref<64xi64, strided<[?], offset: ?>>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi64, strided<[?], offset: ?>>, memref<64xi64, strided<[?], offset: ?>>) outs(%alloc : memref<64xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %0 = arith.addi %in, %in_0 : i64
      linalg.yield %0 : i64
    }
    memref.copy %alloc, %arg2 : memref<64xi64> to memref<64xi64, strided<[?], offset: ?>>
    memref.dealloc %alloc : memref<64xi64>
    return
    // CHECK-LABEL: func @ewbinop_inplace
    // CHECK-NEXT:  linalg.generic
    // CHECK-NEXT:  ^bb0
    // CHECK-NEXT:  arith.addi
    // CHECK-NEXT:  linalg.yield
    // CHECK-NEXT:  }
    // CHECK-NEXT:  return
  }
  func.func @ewbinop_subview(%arg0: memref<64xi64, strided<[?], offset: ?>>, %arg1: memref<64xi64, strided<[?], offset: ?>>, %arg2: memref<65xi64, strided<[?], offset: ?>>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi64, strided<[?], offset: ?>>, memref<64xi64, strided<[?], offset: ?>>) outs(%alloc : memref<64xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %0 = arith.addi %in, %in_0 : i64
      linalg.yield %0 : i64
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg2, %c0 : memref<65xi64, strided<[?], offset: ?>>
    %31 = arith.subi %dim, %c1 : index
    %subview = memref.subview %arg2[%c1] [%31] [1] : memref<65xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
    memref.copy %alloc, %subview: memref<64xi64> to memref<?xi64, strided<[?], offset: ?>>
    memref.dealloc %alloc : memref<64xi64>
    return
    // CHECK-LABEL: func @ewbinop_subview
    // CHECK-NEXT:  arith.constant
    // CHECK-NEXT:  arith.constant
    // CHECK-NEXT:  memref.dim
    // CHECK-NEXT:  arith.subi
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  linalg.generic
  }
  func.func @ewbinop_regions(%arg0: memref<64xi64, strided<[?], offset: ?>>, %arg1: memref<64xi64, strided<[?], offset: ?>>, %arg2: memref<64xi64, strided<[?], offset: ?>>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi64, strided<[?], offset: ?>>, memref<64xi64, strided<[?], offset: ?>>) outs(%alloc : memref<64xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %0 = arith.addi %in, %in_0 : i64
      linalg.yield %0 : i64
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = arith.cmpi sgt, %c1, %c0 : index
    scf.if %1 {
      memref.copy %alloc, %arg2 : memref<64xi64> to memref<64xi64, strided<[?], offset: ?>>
    }
    memref.dealloc %alloc : memref<64xi64>
    return
    // NOTE: no change in this case
    // CHECK-LABEL: func @ewbinop_regions
    // CHECK-NEXT:  memref.alloc
    // CHECK-NEXT:  linalg.generic
    // CHECK:       scf.if
    // CHECK-NEXT:  memref.copy
  }
  func.func @ewbinop_raw_conflict() {
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xi64>
    %subview = memref.subview %alloc[0, 1] [16, 10] [1, 1] : memref<16x16xi64> to memref<16x10xi64, strided<[16, 1], offset: 1>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x10xi64>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<16x10xi64, strided<[16, 1], offset: 1>>) outs(%alloc_0 : memref<16x10xi64>) {
    ^bb0(%in: i64, %out: i64):
      %0 = arith.addi %in, %c1_i64 : i64
      linalg.yield %0 : i64
    }
    %subview_1 = memref.subview %alloc[0, 3] [16, 10] [1, 1] : memref<16x16xi64> to memref<16x10xi64, strided<[16, 1], offset: 3>>
    memref.copy %alloc_0, %subview_1 : memref<16x10xi64> to memref<16x10xi64, strided<[16, 1], offset: 3>>
    memref.dealloc %alloc_0 : memref<16x10xi64>
    return
    // NOTE: subview with offset, no change
    // CHECK-LABEL: func @ewbinop_raw_conflict
    // CHECK:       memref.alloc
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.alloc
    // CHECK-NEXT:  linalg.generic
  }
  func.func @ewbinop_raw_conflict2() {
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xi64>
    %subview = memref.subview %alloc[0, 1] [16, 10] [1, 1] : memref<16x16xi64> to memref<16x10xi64, strided<[16, 1], offset: 1>>
    %subview_0 = memref.subview %subview[2, 0] [14, 10] [1, 1] : memref<16x10xi64, strided<[16, 1], offset: 1>> to memref<14x10xi64, strided<[16, 1], offset: 33>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<14x10xi64>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_0 : memref<14x10xi64, strided<[16, 1], offset: 33>>) outs(%alloc_1 : memref<14x10xi64>) {
    ^bb0(%in: i64, %out: i64):
      %0 = arith.addi %in, %c1_i64 : i64
      linalg.yield %0 : i64
    }
    %subview_2 = memref.subview %alloc[2, 1] [14, 10] [1, 1] : memref<16x16xi64> to memref<14x10xi64, strided<[16, 1], offset: 33>>
    memref.copy %alloc_1, %subview_2 : memref<14x10xi64> to memref<14x10xi64, strided<[16, 1], offset: 33>>
    memref.dealloc %alloc_1 : memref<14x10xi64>
    return
    // NOTE: exact alias, copy can be removed
    // CHECK-LABEL: func @ewbinop_raw_conflict
    // CHECK:       memref.alloc
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  linalg.generic
  }
  func.func @ewbinop_raw_conflict3() {
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xi64>
    %subview = memref.subview %alloc[0, 1] [16, 10] [1, 1] : memref<16x16xi64> to memref<16x10xi64, strided<[16, 1], offset: 1>>
    %subview_0 = memref.subview %subview[1, 0] [14, 10] [1, 1] : memref<16x10xi64, strided<[16, 1], offset: 1>> to memref<14x10xi64, strided<[16, 1], offset: 17>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<14x10xi64>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_0 : memref<14x10xi64, strided<[16, 1], offset: 17>>) outs(%alloc_1 : memref<14x10xi64>) {
    ^bb0(%in: i64, %out: i64):
      %0 = arith.addi %in, %c1_i64 : i64
      linalg.yield %0 : i64
    }
    %subview_2 = memref.subview %alloc[2, 1] [14, 10] [1, 1] : memref<16x16xi64> to memref<14x10xi64, strided<[16, 1], offset: 33>>
    memref.copy %alloc_1, %subview_2 : memref<14x10xi64> to memref<14x10xi64, strided<[16, 1], offset: 33>>
    memref.dealloc %alloc_1 : memref<14x10xi64>
    return
    // NOTE: alias with an offset, no change
    // CHECK-LABEL: func @ewbinop_raw_conflict
    // CHECK:       memref.alloc
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK:       memref.alloc
    // CHECK-NEXT:  linalg.generic
  }
  func.func @ewbinop_raw_conflict_subview() {
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16xi64>
    %subview = memref.subview %alloc[0, 3] [2, 10] [1, 1] : memref<8x16xi64> to memref<2x10xi64, strided<[16, 1], offset: 3>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<5x10xi64>
    %subview_1 = memref.subview %alloc_0[3, 0] [2, 10] [1, 1] : memref<5x10xi64> to memref<2x10xi64, strided<[10, 1], offset: 30>>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<2x10xi64, strided<[16, 1], offset: 3>>) outs(%subview_1 : memref<2x10xi64, strided<[10, 1], offset: 30>>) {
    ^bb0(%in: i64, %out: i64):
      %0 = arith.addi %in, %c1_i64 : i64
      linalg.yield %0 : i64
    }
    %subview_2 = memref.subview %alloc[0, 3] [5, 10] [1, 1] : memref<8x16xi64> to memref<5x10xi64, strided<[16, 1], offset: 3>>
    memref.copy %alloc_0, %subview_2 : memref<5x10xi64> to memref<5x10xi64, strided<[16, 1], offset: 3>>
    return
    // NOTE: alias with an offset, no change
    // CHECK-LABEL: func @ewbinop_raw_conflict_subview
    // CHECK:       memref.alloc
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.alloc
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  linalg.generic
  }
  func.func @ewbinop_raw_conflict_subview2() {
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16xi64>
    %subview = memref.subview %alloc[3, 3] [2, 10] [1, 1] : memref<8x16xi64> to memref<2x10xi64, strided<[16, 1], offset: 51>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<5x10xi64>
    %subview_1 = memref.subview %alloc_0[3, 0] [2, 10] [1, 1] : memref<5x10xi64> to memref<2x10xi64, strided<[10, 1], offset: 30>>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<2x10xi64, strided<[16, 1], offset: 51>>) outs(%subview_1 : memref<2x10xi64, strided<[10, 1], offset: 30>>) {
    ^bb0(%in: i64, %out: i64):
      %0 = arith.addi %in, %c1_i64 : i64
      linalg.yield %0 : i64
    }
    %subview_2 = memref.subview %alloc[0, 3] [5, 10] [1, 1] : memref<8x16xi64> to memref<5x10xi64, strided<[16, 1], offset: 3>>
    memref.copy %alloc_0, %subview_2 : memref<5x10xi64> to memref<5x10xi64, strided<[16, 1], offset: 3>>
    memref.dealloc %alloc : memref<8x16xi64>
    memref.dealloc %alloc_0 : memref<5x10xi64>
    return
    // NOTE: exact alias, copy can be removed
    // CHECK-LABEL: func @ewbinop_raw_conflict_subview
    // CHECK:       memref.alloc
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  linalg.generic
    // CHECK:  memref.dealloc
    // CHECK-NEXT:  return
  }
  func.func @ewbinop_farg_raw_conflict(%arg0 : memref<16x16xi64, strided<[?, ?], offset: ?>>) {
    %c1_i64 = arith.constant 1 : i64
    %subview = memref.subview %arg0[0, 1] [16, 10] [1, 1] : memref<16x16xi64, strided<[?, ?], offset: ?>> to memref<16x10xi64, strided<[?, ?], offset: ?>>
    %subview_0 = memref.subview %subview[2, 0] [14, 10] [1, 1] : memref<16x10xi64, strided<[?, ?], offset: ?>> to memref<14x10xi64, strided<[?, ?], offset: ?>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<14x10xi64>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_0 : memref<14x10xi64, strided<[?, ?], offset: ?>>) outs(%alloc_1 : memref<14x10xi64>) {
    ^bb0(%in: i64, %out: i64):
      %0 = arith.addi %in, %c1_i64 : i64
      linalg.yield %0 : i64
    }
    %subview_2 = memref.subview %arg0[2, 1] [14, 10] [1, 1] : memref<16x16xi64, strided<[?, ?], offset: ?>> to memref<14x10xi64, strided<[?, ?], offset: ?>>
    memref.copy %alloc_1, %subview_2 : memref<14x10xi64> to memref<14x10xi64, strided<[?, ?], offset: ?>>
    memref.dealloc %alloc_1 : memref<14x10xi64>
    return
    // NOTE: exact alias, copy can be removed
    // CHECK-LABEL: func @ewbinop_farg_raw_conflict
    // CHECK:       memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  linalg.generic
    // CHECK-NEXT:  ^bb0
    // CHECK-NEXT:  arith.addi
    // CHECK-NEXT:  linalg.yield
    // CHECK-NEXT:  }
    // CHECK-NEXT:  return
  }
  func.func @ewbinop_farg_raw_conflict_subview(%arg0 : memref<8x16xi64, strided<[?, ?], offset: ?>>) {
    %c1_i64 = arith.constant 1 : i64
    %subview = memref.subview %arg0[3, 3] [2, 10] [1, 1] : memref<8x16xi64, strided<[?, ?], offset: ?>> to memref<2x10xi64, strided<[?, ?], offset: ?>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<5x10xi64>
    %subview_1 = memref.subview %alloc_0[3, 0] [2, 10] [1, 1] : memref<5x10xi64> to memref<2x10xi64, strided<[10, 1], offset: 30>>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<2x10xi64, strided<[?, ?], offset: ?>>) outs(%subview_1 : memref<2x10xi64, strided<[10, 1], offset: 30>>) {
    ^bb0(%in: i64, %out: i64):
      %0 = arith.addi %in, %c1_i64 : i64
      linalg.yield %0 : i64
    }
    %subview_2 = memref.subview %arg0[0, 3] [5, 10] [1, 1] : memref<8x16xi64, strided<[?, ?], offset: ?>> to memref<5x10xi64, strided<[?, ?], offset: ?>>
    memref.copy %alloc_0, %subview_2 : memref<5x10xi64> to memref<5x10xi64, strided<[?, ?], offset: ?>>
    memref.dealloc %alloc_0 : memref<5x10xi64>
    return
    // NOTE: exact alias, copy can be removed
    // CHECK-LABEL: func @ewbinop_farg_raw_conflict_subview
    // CHECK:       memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  linalg.generic
    // CHECK-NEXT:  ^bb0
    // CHECK-NEXT:  arith.addi
    // CHECK-NEXT:  linalg.yield
    // CHECK-NEXT:  }
    // CHECK-NEXT:  return
  }
  func.func @ewbinop_double_copy(%arg0: memref<12xi64>, %arg1: memref<12xi64>) {
    %subview = memref.subview %arg1[0] [6] [1] : memref<12xi64> to memref<6xi64, strided<[1]>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<6xi64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%subview, %subview : memref<6xi64, strided<[1]>>, memref<6xi64, strided<[1]>>) outs(%alloc : memref<6xi64>) {
    ^bb0(%in: i64, %in_2: i64, %out: i64):
      %0 = arith.addi %in, %in_2 : i64
      linalg.yield %0 : i64
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<12xi64>
    memref.copy %arg0, %alloc_0 : memref<12xi64> to memref<12xi64>
    %subview_1 = memref.subview %alloc_0[0] [6] [1] : memref<12xi64> to memref<6xi64, strided<[1]>>
    memref.copy %alloc, %subview_1 : memref<6xi64> to memref<6xi64, strided<[1]>>
    memref.dealloc %alloc : memref<6xi64>
    memref.dealloc %alloc_0 : memref<12xi64>
    return
    // NOTE: 2nd copy is removed, 1st copy is moved up due to write effects
    // CHECK-LABEL: func @ewbinop_double_copy
    // CHECK:       memref.subview
    // CHECK-NEXT:  memref.alloc
    // CHECK-NEXT:  memref.copy
    // CHECK-NEXT:  memref.subview
    // CHECK-NEXT:  linalg.generic
    // CHECK-NEXT:  ^bb0
    // CHECK-NEXT:  arith.addi
    // CHECK-NEXT:  linalg.yield
    // CHECK-NEXT:  }
    // CHECK-NEXT:  memref.dealloc
    // CHECK-NEXT:  return
  }
  func.func @ewbinop_return_alias(%arg0: memref<64xi64>, %arg1: memref<64xi64>, %arg2: memref<64xi64>) -> memref<64xi64> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi64>, memref<64xi64>) outs(%alloc : memref<64xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %0 = arith.addi %in, %in_0 : i64
      linalg.yield %0 : i64
    }
    memref.copy %alloc, %arg2 : memref<64xi64> to memref<64xi64>
    return %alloc : memref<64xi64>
    // CHECK-LABEL: func @ewbinop_return_alias
    // CHECK-NEXT:  memref.alloc
    // CHECK-NEXT:  linalg.generic
    // CHECK-NEXT:  ^bb0
    // CHECK-NEXT:  arith.addi
    // CHECK-NEXT:  linalg.yield
    // CHECK-NEXT:  }
    // CHECK-NEXT:  memref.copy
  }
  func.func @ewbinop_return_alias2(%arg0: memref<64xi64>, %arg1: memref<64xi64>) -> (memref<64xi64>, memref<64xi64>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xi64>
    %arg2 = memref.alloc() {alignment = 64 : i64} : memref<64xi64>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xi64>, memref<64xi64>) outs(%alloc : memref<64xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %0 = arith.addi %in, %in_0 : i64
      linalg.yield %0 : i64
    }
    memref.copy %alloc, %arg2 : memref<64xi64> to memref<64xi64>
    return %alloc, %arg2 : memref<64xi64>, memref<64xi64>
    // CHECK-LABEL: func @ewbinop_return_alias2
    // CHECK-NEXT:  memref.alloc
    // CHECK-NEXT:  memref.alloc
    // CHECK-NEXT:  linalg.generic
    // CHECK-NEXT:  ^bb0
    // CHECK-NEXT:  arith.addi
    // CHECK-NEXT:  linalg.yield
    // CHECK-NEXT:  }
    // CHECK-NEXT:  memref.copy
  }
}
