// RUN: imex-opt -allow-unregistered-dialect -mlir-print-ir-before-all -mlir-print-ir-after-all -split-input-file -unstride-memrefs -verify-diagnostics %s -o - | FileCheck %s

func.func @addt(%arg0: memref<2x5x7xf32>, %arg1: memref<2x5x7xf32>) -> memref<2x5x7xf32> {
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %memref = gpu.alloc  () {gpu.alloc_shared} : memref<2x5x7xf32>
  memref.copy %arg1, %memref : memref<2x5x7xf32> to memref<2x5x7xf32>
  %memref_0 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5x7xf32>
  memref.copy %arg0, %memref_0 : memref<2x5x7xf32> to memref<2x5x7xf32>
  %memref_1 = gpu.alloc  () {gpu.alloc_shared} : memref<2x5x7xf32>
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c2, %arg9 = %c5, %arg10 = %c7) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    %0 = memref.load %memref_0[%arg2, %arg3, %arg4] : memref<2x5x7xf32>
    %1 = memref.load %memref[%arg2, %arg3, %arg4] : memref<2x5x7xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %memref_1[%arg2, %arg3, %arg4] : memref<2x5x7xf32>
    gpu.terminator
  } {SCFToGPU_visited}
  gpu.dealloc  %memref_1 : memref<2x5x7xf32>
  gpu.dealloc  %memref_0 : memref<2x5x7xf32>
  gpu.dealloc  %memref : memref<2x5x7xf32>
  return %memref_1 : memref<2x5x7xf32>
}
