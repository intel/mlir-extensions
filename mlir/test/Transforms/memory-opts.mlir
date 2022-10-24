// RUN: imex-opt -pass-pipeline='func.func(imex-memory-opts)' --split-input-file %s | FileCheck %s

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
