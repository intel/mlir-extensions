// RUN: imex-opt -allow-unregistered-dialect -pass-pipeline='func.func(imex-memory-opts)' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @dead_store1()
func.func @dead_store1() -> memref<10xf32> {
  %c0 = arith.constant 0 : index
  %cf7 = arith.constant 7.0 : f32
  // CHECK: %[[M:.*]] = memref.alloc() : memref<10xf32>
  %m = memref.alloc() : memref<10xf32>
  // CHECK: memref.store %{{.*}}, %[[M]][%{{.*}}]
  memref.store %cf7, %m[%c0] : memref<10xf32>
  memref.store %cf7, %m[%c0] : memref<10xf32>
  // CHECK-NEXT: return %[[M]]
  return %m : memref<10xf32>
}

// -----

// CHECK-LABEL: func @dead_store2()
func.func @dead_store2() {
  %c0 = arith.constant 0 : index
  %cf7 = arith.constant 7.0 : f32
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.store
  %m = memref.alloc() : memref<10xf32>
  memref.store %cf7, %m[%c0] : memref<10xf32>
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @simple_store_load
// CHECK-SAME:  (%[[C:.*]]: f32)
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NEXT:    %[[R:.*]] = arith.addf %[[C]], %[[C]] : f32
// CHECK-NEXT:    "test.test"(%[[R]]) : (f32) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @simple_store_load(%cf : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %m = memref.alloc() : memref<10xf32>
  scf.for %i0 = %c0 to %c10 step %c1 {
    memref.store %cf, %m[%i0] : memref<10xf32>
    %v0 = memref.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
    "test.test"(%v1) : (f32) -> ()
  }
  memref.dealloc %m : memref<10xf32>
  return
}

// -----

// CHECK-LABEL: func @simple_store_load
// CHECK-SAME:  (%[[C:.*]]: f32)
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NEXT:    %[[R:.*]] = arith.addf %[[C]], %[[C]] : f32
// CHECK-NEXT:    "test.test"(%[[R]]) : (f32) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @simple_store_load_nested(%cf : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %m = memref.alloc() : memref<10x5xf32>
  scf.for %i0 = %c0 to %c10 step %c1 {
    scf.for %i1 = %c0 to %c10 step %c1 {
      memref.store %cf, %m[%i0, %i1] : memref<10x5xf32>
      %v0 = memref.load %m[%i0, %i1] : memref<10x5xf32>
      %v1 = arith.addf %v0, %v0 : f32
      "test.test"(%v1) : (f32) -> ()
    }
  }
  memref.dealloc %m : memref<10x5xf32>
  return
}

// -----

// CHECK-LABEL: func @multi_store_load
// CHECK-SAME:  (%[[C1:.*]]: f32, %[[C2:.*]]: f32, %[[C3:.*]]: f32)
// CHECK:  scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NEXT:    %[[R1:.*]] = arith.addf %[[C1]], %[[C1]] : f32
// CHECK-NEXT:    %[[R2:.*]] = arith.mulf %[[C3]], %[[C3]] : f32
// CHECK-NEXT:    "test.test"(%[[R1]], %[[R2]]) : (f32, f32) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @multi_store_load(%cf1 : f32, %cf2 : f32, %cf3 : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %m = gpu.alloc() : memref<10xf32>
  scf.for %i0 = %c0 to %c10 step %c1 {
    memref.store %cf1, %m[%i0] : memref<10xf32>
    %v0 = memref.load %m[%i0] : memref<10xf32>
    %v1 = arith.addf %v0, %v0 : f32
    memref.store %cf2, %m[%i0] : memref<10xf32>
    memref.store %cf3, %m[%i0] : memref<10xf32>
    %v2 = memref.load %m[%i0] : memref<10xf32>
    %v3 = memref.load %m[%i0] : memref<10xf32>
    %v4 = arith.mulf %v2, %v3 : f32
    "test.test"(%v1, %v4) : (f32, f32) -> ()
  }
  gpu.dealloc %m : memref<10xf32>
  return
}
