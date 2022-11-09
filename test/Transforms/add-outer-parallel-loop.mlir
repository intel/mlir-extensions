// RUN: imex-opt --imex-add-outer-parallel-loop %s | FileCheck %s
func.func @test(%arg0: memref<10x20xf32>) -> memref<f32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<f32>
  memref.store %cst, %alloc[] : memref<f32>
  %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<f32>
  memref.copy %alloc, %alloc_0 : memref<f32> to memref<f32>
  // CHECK: scf.parallel
  scf.for %arg1 = %c0 to %c10 step %c1 {
    scf.for %arg2 = %c0 to %c20 step %c1 {
      %0 = memref.load %arg0[%arg1, %arg2] : memref<10x20xf32>
      %1 = memref.load %alloc_0[] : memref<f32>
      %2 = arith.addf %1, %0 : f32
      memref.store %2, %alloc_0[] : memref<f32>
    }
  }
  return %alloc_0 : memref<f32>
}
