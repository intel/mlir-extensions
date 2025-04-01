// RUN: imex-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries=1" %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_subview(%arg0: tensor<?xi64>) -> tensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ndarray.subview %arg0[%c0][%c3][%c3] : tensor<?xi64> to tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: func.func @test_subview(
// CHECK-SAME: [[varg0:%.*]]: memref<?xi64, strided<[?], offset: ?>>) -> memref<?xi64, strided<[?], offset: ?>> {
// CHECK-NEXT: [[vc0:%.*]] = arith.constant 0 : index
// CHECK-NEXT: [[vc3:%.*]] = arith.constant 3 : index
// CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[varg0]][[[vc0]]] [[[vc3]]] [[[vc3]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return [[vsubview]] : memref<?xi64, strided<[?], offset: ?>>


// -----
func.func @test_insert_slice(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%i0] [%i3] [%i1] : tensor<?xi64> into tensor<?xi64>
    return
}
// CHECK-LABEL: func.func @test_insert_slice(
// CHECK-SAME: [[varg0:%.*]]: memref<?xi64, strided<[?], offset: ?>>, [[varg1:%.*]]: memref<?xi64, strided<[?], offset: ?>>) {
// CHECK-NEXT: [[vc0:%.*]] = arith.constant 0 : index
// CHECK-NEXT: [[vc1:%.*]] = arith.constant 1 : index
// CHECK-NEXT: [[vc3:%.*]] = arith.constant 3 : index
// CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[varg0]][[[vc0]]] [[[vc3]]] [[[vc1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: memref.copy [[varg1]], [[vsubview]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return


// -----
func.func @test_insert_slice_scalar(%arg0: tensor<?xi64>, %arg1: tensor<i64>) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i3 = arith.constant 3 : index
    ndarray.insert_slice %arg1 into %arg0[%i0] [%i3] [%i1] : tensor<i64> into tensor<?xi64>
    return
}
// CHECK-LABEL: func.func @test_insert_slice_scalar(
// CHECK-SAME: [[varg0:%.*]]: memref<?xi64, strided<[?], offset: ?>>, [[varg1:%.*]]: memref<i64, strided<[], offset: ?>>) {
// CHECK-NEXT: [[vc0:%.*]] = arith.constant 0 : index
// CHECK-NEXT: [[vc1:%.*]] = arith.constant 1 : index
// CHECK-NEXT: [[vc3:%.*]] = arith.constant 3 : index
// CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[varg0]][[[vc0]]] [[[vc3]]] [[[vc1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins([[varg1]] : memref<i64, strided<[], offset: ?>>) outs([[vsubview]] : memref<?xi64, strided<[?], offset: ?>>) {
// CHECK-NEXT: ^bb0([[vin:%.*]]: i64, [[vout:%.*]]: i64):
// CHECK-NEXT: linalg.yield [[vin]] : i64
// CHECK: return
