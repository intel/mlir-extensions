// RUN: imex-opt --insert-gpu-alloc %s | FileCheck %s

func.func @main() {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  // CHECK: func.func @main()
  %0 = memref.alloc() : memref<8xf32>
  %1 = memref.alloc() : memref<8xf32>
  %2 = memref.alloc() : memref<8xf32>
  // CHECK: %[[MEMREF0:.*]]= gpu.alloc  () : memref<8xf32>
  // CHECK: %[[MEMREF1:.*]]= gpu.alloc  () : memref<8xf32>
  // CHECK: %[[MEMREF2:.*]]= gpu.alloc  () : memref<8xf32>
  gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c8, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) {
    // CHECK: gpu.launch {{.*}}
    %7 = gpu.block_id  x
    // CHECK: [[VAR0:.*]] = gpu.block_id  x
    %8 = memref.load %0[%7] : memref<8xf32>
    %9 = memref.load %1[%7] : memref<8xf32>

    // CHECK: [[VAR1:.*]] = memref.load %[[MEMREF0:.*]][[[VAR0:.*]]] : memref<8xf32>
    // CHECK: [[VAR2:.*]] = memref.load %[[MEMREF1:.*]][[[VAR0:.*]]] : memref<8xf32>
    %10 = func.call @addf(%8, %9) : (f32, f32) -> f32
    memref.store %10, %2[%7] : memref<8xf32>
    %11 = func.call @cast(%2) : (memref<8xf32>) -> memref<?xf32>
    gpu.terminator
    // CHECK: gpu.terminator
  }
  %6 = memref.cast %2 : memref<8xf32> to memref<*xf32>
  return
}

func.func private @addf(%input1 : f32, %input2 : f32) -> f32
func.func private @cast(%input : memref<8xf32>) -> memref<?xf32>
