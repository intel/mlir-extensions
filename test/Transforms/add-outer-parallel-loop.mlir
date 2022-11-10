// RUN: imex-opt --imex-add-outer-parallel-loop %s | FileCheck %s
func.func @single_loop(%arg0: memref<10x20xf32>) {
  // CHECK-LABEL: func @single_loop
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: scf.parallel
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: scf.for
  scf.for %arg1 = %c0 to %c10 step %c1 {
    scf.for %arg2 = %c0 to %c20 step %c1 {
      %0 = memref.load %arg0[%arg1, %arg2] : memref<10x20xf32>
      %1 = arith.addf %cst, %0 : f32
      memref.store %1, %arg0[%arg1, %arg2] : memref<10x20xf32>
    }
  }
  return
}

func.func @two_loops(%arg0: memref<10x20xf32>) {
  // CHECK-LABEL: func @two_loops
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: scf.parallel
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: scf.for
  scf.for %arg1 = %c0 to %c10 step %c1 {
    scf.for %arg2 = %c0 to %c20 step %c1 {
      %0 = memref.load %arg0[%arg1, %arg2] : memref<10x20xf32>
      %1 = arith.addf %cst, %0 : f32
      memref.store %1, %arg0[%arg1, %arg2] : memref<10x20xf32>
    }
  }
  // CHECK: scf.parallel
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: scf.for
  scf.for %arg1 = %c0 to %c10 step %c1 {
    scf.for %arg2 = %c0 to %c20 step %c1 {
      %0 = memref.load %arg0[%arg1, %arg2] : memref<10x20xf32>
      %1 = arith.addf %cst, %0 : f32
      memref.store %1, %arg0[%arg1, %arg2] : memref<10x20xf32>
    }
  }
  return
}
