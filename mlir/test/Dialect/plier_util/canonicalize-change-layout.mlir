// RUN: imex-opt %s -canonicalize --split-input-file | FileCheck %s


// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32>) -> memref<?xi32, #[[MAP:.*]]>
//       CHECK:   %[[RES:.*]] = memref.cast %[[ARG]] : memref<?xi32> to memref<?xi32, #[[MAP]]>
//       CHECK:   return %[[RES]]
#map = affine_map<(d0)[s0, s1] -> (s0 + s1 * d0)>
func.func @test_change_layout(%arg : memref<?xi32>) -> memref<?xi32, #map> {
  %0 = plier_util.change_layout %arg : memref<?xi32> to memref<?xi32, #map>
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
  %0 = plier_util.change_layout %arg : memref<?xi32, #map> to memref<?xi32>
  %c0 = arith.constant 0 : index
  %1 = memref.dim %0, %c0 : memref<?xi32>
  return %1 : index
}

// -----

// CHECK-LABEL: func @test_change_layout
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xi32, #[[MAP:.*]]>) -> memref<?xi32>
//       CHECK:   %[[CLONE:.*]]  = bufferization.clone %[[ARG]] : memref<?xi32, #[[MAP:.*]]> to memref<?xi32, #[[MAP:.*]]>
//       CHECK:   %[[RES:.*]]  = plier_util.change_layout %[[CLONE]] : memref<?xi32, #map> to memref<?xi32>
//       CHECK:   return %[[RES]]
#map = affine_map<(d0)[s0, s1] -> (s0 + s1 * d0)>
func.func @test_change_layout(%arg : memref<?xi32, #map>) -> memref<?xi32> {
  %0 = plier_util.change_layout %arg : memref<?xi32, #map> to memref<?xi32>
  %1 = bufferization.clone %0 : memref<?xi32> to memref<?xi32>
  return %1 : memref<?xi32>
}
