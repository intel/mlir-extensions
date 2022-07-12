// RUN: level_zero_runner %s -e main -entry-point-result=void -shared-libs=%mlir_wrappers_dir/%shlibprefixmlir_c_runner_utils%shlibext -shared-libs=%mlir_wrappers_dir/%shlibprefixmlir_runner_utils%shlibext -shared-libs=%imex_runtime_dir/%shlibprefixdpcomp-runtime%shlibext -shared-libs=%imex_igpu_runtime_dir/%shlibprefixdpcomp-gpu-runtime%shlibext | FileCheck %s

  func.func @main() {
    %arg0 = memref.alloc() : memref<8xf32>
    %arg1 = memref.alloc() : memref<8xf32>
    %arg2 = memref.alloc() : memref<8xf32>
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = arith.constant 2 : i32
    %value0 = arith.constant 0.0 : f32
    %value1 = arith.constant 1.1 : f32
    %value2 = arith.constant 2.2 : f32
    %arg3 = memref.cast %arg0 : memref<8xf32> to memref<?xf32>
    %arg4 = memref.cast %arg1 : memref<8xf32> to memref<?xf32>
    %arg5 = memref.cast %arg2 : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%arg3, %value1) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%arg4, %value2) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%arg5, %value0) : (memref<?xf32>, f32) -> ()

    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg10 = %cst8, %arg11 = %cst1, %arg12 = %cst1) threads(%arg13, %arg14, %arg15) in (%arg16 = %cst1, %arg17 = %cst1, %arg18 = %cst1) {
       %5 = gpu.block_id x
       %6 = memref.load %arg0[%5] : memref<8xf32>
       %7 = memref.load %arg1[%5] : memref<8xf32>
       %8 = arith.addf %6, %7 : f32
      memref.store %8, %arg2[%5] : memref<8xf32>
      gpu.terminator
    }
    %arg6 = memref.cast %arg5 : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%arg6) : (memref<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [8] strides = [1] data =
    // CHECK-NEXT: [3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3]
    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
