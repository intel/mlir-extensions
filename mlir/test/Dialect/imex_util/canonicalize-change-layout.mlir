// RUN: imex-opt %s -allow-unregistered-dialect -canonicalize --split-input-file | FileCheck %s


// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32>) -> memref<?xi32, #[[MAP:.*]]>
//       CHECK:   %[[RES:.*]] = memref.cast %[[ARG]] : memref<?xi32> to memref<?xi32, #[[MAP]]>
//       CHECK:   return %[[RES]]
#map = affine_map<(d0)[s0, s1] -> (s0 + s1 * d0)>
func.func @test_change_layout(%arg : memref<?xi32>) -> memref<?xi32, #map> {
  %0 = imex_util.change_layout %arg : memref<?xi32> to memref<?xi32, #map>
  return %0 : memref<?xi32, #map>
}

// -----

// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32, #[[MAP:.*]]>) -> index
//       CHECK:   %[[C:.*]] = arith.constant 0 : index
//       CHECK:   %[[RES:.*]]  = memref.dim %[[ARG]], %[[C]] : memref<?xi32, #[[MAP:.*]]>
//       CHECK:   return %[[RES]]
#map = affine_map<(d0)[s0, s1] -> (s0 + s1 * d0)>
func.func @test_change_layout(%arg : memref<?xi32, #map>) -> index {
  %0 = imex_util.change_layout %arg : memref<?xi32, #map> to memref<?xi32>
  %c0 = arith.constant 0 : index
  %1 = memref.dim %0, %c0 : memref<?xi32>
  return %1 : index
}

// -----

// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32, #[[MAP:.*]]>) -> memref<?xi32>
//       CHECK:   %[[CLONE:.*]]  = bufferization.clone %[[ARG]] : memref<?xi32, #[[MAP]]> to memref<?xi32, #[[MAP]]>
//       CHECK:   %[[RES:.*]]  = imex_util.change_layout %[[CLONE]] : memref<?xi32, #[[MAP]]> to memref<?xi32>
//       CHECK:   return %[[RES]]
#map = affine_map<(d0)[s0, s1] -> (s0 + s1 * d0)>
func.func @test_change_layout(%arg : memref<?xi32, #map>) -> memref<?xi32> {
  %0 = imex_util.change_layout %arg : memref<?xi32, #map> to memref<?xi32>
  %1 = bufferization.clone %0 : memref<?xi32> to memref<?xi32>
  return %1 : memref<?xi32>
}

// -----

// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?xi32, #[[MAP:.*]]>, %[[ARG2:.*]]: memref<?xi32>, %[[ARG3:.*]]: i1) -> memref<?xi32>
//       CHECK:   %[[CAST:.*]]  = memref.cast %[[ARG2]] : memref<?xi32> to memref<?xi32, #[[MAP]]>
//       CHECK:   %[[SELECT:.*]]  = arith.select %[[ARG3]], %[[ARG1]], %[[CAST]] : memref<?xi32, #[[MAP]]>
//       CHECK:   %[[RES:.*]]  = imex_util.change_layout %[[SELECT]] : memref<?xi32, #[[MAP]]> to memref<?xi32>
//       CHECK:   return %[[RES]]
#map = affine_map<(d0)[s0, s1] -> (s0 + s1 * d0)>
func.func @test_change_layout(%arg1 : memref<?xi32, #map>, %arg2 : memref<?xi32>, %arg3 : i1) -> memref<?xi32> {
  %0 = imex_util.change_layout %arg1 : memref<?xi32, #map> to memref<?xi32>
  %1 = arith.select %arg3, %0, %arg2 : memref<?xi32>
  return %1 : memref<?xi32>
}

// -----

// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?xi32>, %[[ARG2:.*]]: memref<?xi32, #[[MAP:.*]]>, %[[ARG3:.*]]: i1) -> memref<?xi32>
//       CHECK:   %[[CAST:.*]]  = memref.cast %[[ARG1]] : memref<?xi32> to memref<?xi32, #[[MAP]]>
//       CHECK:   %[[SELECT:.*]]  = arith.select %[[ARG3]], %[[CAST]], %[[ARG2]] : memref<?xi32, #[[MAP]]>
//       CHECK:   %[[RES:.*]]  = imex_util.change_layout %[[SELECT]] : memref<?xi32, #[[MAP]]> to memref<?xi32>
//       CHECK:   return %[[RES]]
#map = affine_map<(d0)[s0, s1] -> (s0 + s1 * d0)>
func.func @test_change_layout(%arg1 : memref<?xi32>, %arg2 : memref<?xi32, #map>, %arg3 : i1) -> memref<?xi32> {
  %0 = imex_util.change_layout %arg2 : memref<?xi32, #map> to memref<?xi32>
  %1 = arith.select %arg3, %arg1, %0 : memref<?xi32>
  return %1 : memref<?xi32>
}

// -----

// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?x?xf64, strided<[1, 1], offset: ?>>, %[[ARG2:.*]]: memref<?x?xf64, strided<[1, 1], offset: ?>>, %[[ARG3:.*]]: i1)
//       CHECK:   %[[RES1:.*]] = arith.select %[[ARG3]], %[[ARG1]], %[[ARG2]] : memref<?x?xf64, strided<[1, 1], offset: ?>>
//       CHECK:   %[[RES2:.*]] = imex_util.change_layout %[[RES1]] : memref<?x?xf64, strided<[1, 1], offset: ?>> to memref<?x?xf64>
//       CHECK:   return %[[RES2]] : memref<?x?xf64>
func.func @test_change_layout(%arg1 : memref<?x?xf64, strided<[1, 1], offset: ?>>, %arg2 : memref<?x?xf64, strided<[1, 1], offset: ?>>, %arg3 : i1) -> memref<?x?xf64> {
  %0 = scf.if %arg3 -> (memref<?x?xf64>) {
    %1 = imex_util.change_layout %arg1 : memref<?x?xf64, strided<[1, 1], offset: ?>> to memref<?x?xf64>
    scf.yield %1 : memref<?x?xf64>
  } else {
    %2 = imex_util.change_layout %arg2 : memref<?x?xf64, strided<[1, 1], offset: ?>> to memref<?x?xf64>
    scf.yield %2 : memref<?x?xf64>
  }
  return %0 : memref<?x?xf64>
}

// -----

// CHECK-LABEL: func @test_change_layout
//       CHECK:   %[[RES1:.*]] = imex_util.env_region "test" -> memref<?xi32, strided<[?], offset: ?>> {
//       CHECK:   %[[VAL:.*]] = "test.test"() : () -> memref<?xi32, strided<[?], offset: ?>>
//       CHECK:   imex_util.env_region_yield %[[VAL]] : memref<?xi32, strided<[?], offset: ?>>
//       CHECK:   }
//       CHECK:   %[[RES2:.*]] = imex_util.change_layout %[[RES1]] : memref<?xi32, strided<[?], offset: ?>> to memref<?xi32>
//       CHECK:   return %[[RES2]] : memref<?xi32>
func.func @test_change_layout() -> memref<?xi32> {
  %0 = imex_util.env_region "test" -> memref<?xi32> {
    %1 = "test.test"() : () -> memref<?xi32, strided<[?], offset: ?>>
    %2 = imex_util.change_layout %1 : memref<?xi32, strided<[?], offset: ?>> to memref<?xi32>
    imex_util.env_region_yield %2: memref<?xi32>
  }
  return %0 : memref<?xi32>
}
