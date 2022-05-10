// REQUIRES: run-gpu-tests
// RUN: mlir-vulkan-runner %s --shared-libs=%vulkan_wrapper_library_dir/libvulkan-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3]
module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {
  
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
    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
