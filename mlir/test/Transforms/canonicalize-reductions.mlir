// RUN: imex-opt -allow-unregistered-dialect --imex-canonicalize-reductions --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
// CHECK-SAME:  (%[[C:.*]]: f32)
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NEXT:    %[[R:.*]] = arith.addf %[[C]], %[[C]] : f32
// CHECK-NEXT:    "test.test"(%[[R]]) : (f32) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @test() -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %m = memref.alloc() : memref<f32>
  scf.for %i0 = %c0 to %c10 step %c1 {
    %v0 = memref.load %m[] : memref<f32>
    %v1 = arith.addf %v0, %v0 : f32
    memref.store %v1, %m[] : memref<f32>
  }
  %res = memref.load %m[] : memref<f32>
  memref.dealloc %m : memref<f32>
  return %res : f32
}
