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

func.func @two_independent_loops(%arg0: memref<10x20xf32>) {
  // CHECK-LABEL: func @two_independent_loops
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

func.func @two_dependent_loops_with_iterargs(%arg0: memref<10x20xf32>) -> memref<f32> {
  // CHECK-LABEL: func @two_dependent_loops_with_iterargs
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<f32>
  // CHECK: scf.parallel
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: scf.for
  // CHECK: scf.for
  // CHECK: scf.reduce
  %res = scf.for %arg1 = %c0 to %c10 step %c1 iter_args (%it = %cst) -> f32 {
    scf.for %arg2 = %c0 to %c20 step %c1 {
      %0 = memref.load %arg0[%arg1, %arg2] : memref<10x20xf32>
      %1 = arith.addf %it, %0 : f32
      memref.store %1, %arg0[%arg1, %arg2] : memref<10x20xf32>
    }
    %new = arith.addf %cst1, %it : f32
    scf.yield  %new : f32
  }
  scf.for %arg4 = %c0 to %c20 step %c1 {
    %1 = arith.addf %cst1, %res : f32
    memref.store %1, %alloc[] : memref<f32>
  }
  return %alloc : memref<f32>
}
