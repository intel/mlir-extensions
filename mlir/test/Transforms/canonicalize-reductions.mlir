// RUN: imex-opt -allow-unregistered-dialect --imex-canonicalize-reductions --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[C:.*]]: f32)
//       CHECK:  %[[A:.*]] = memref.alloc() : memref<f32>
//       CHECK:  memref.store %[[C]], %[[A]][] : memref<f32>
//       CHECK:  %[[I:.*]] = memref.load %[[A]][] : memref<f32>
//       CHECK:  %[[RES1:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %[[I]]) -> (f32)
//       CHECK:    %[[R:.*]] = arith.addf %[[ARG]], %[[ARG]] : f32
//       CHECK:    scf.yield %[[R]] : f32
//       CHECK:  }
//       CHECK:  memref.store %[[RES1]], %[[A]][] : memref<f32>
//       CHECK:  %[[RES2:.*]] = memref.load %[[A]][] : memref<f32>
//       CHECK:  return %[[RES2]]
func.func @test(%arg: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %m = memref.alloc() : memref<f32>
  memref.store %arg, %m[] : memref<f32>
  scf.for %i0 = %c0 to %c10 step %c1 {
    %v0 = memref.load %m[] : memref<f32>
    %v1 = arith.addf %v0, %v0 : f32
    memref.store %v1, %m[] : memref<f32>
  }
  %res = memref.load %m[] : memref<f32>
  memref.dealloc %m : memref<f32>
  return %res : f32
}
