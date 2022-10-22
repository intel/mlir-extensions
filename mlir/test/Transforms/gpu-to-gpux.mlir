// RUN: imex-opt --gpu-to-gpux --split-input-file %s | FileCheck %s

func.func @alloc() {
  // CHECK-LABEL: func @alloc()

  // CHECK: %[[stream:.*]] = gpu_runtime.create_gpu_stream

  // CHECK: %[[m0:.*]] = gpu_runtime.alloc %[[stream]] () : memref<13xf32, 1>
  %m0 = gpu.alloc () : memref<13xf32, 1>
  // CHECK: gpu_runtime.dealloc %[[stream]] %[[m0]] : memref<13xf32, 1>
  gpu.dealloc %m0 : memref<13xf32, 1>

  %t0 = gpu.wait async
  // CHECK: %[[m1:.*]], %[[t1:.*]] = gpu_runtime.alloc %[[stream]] async [{{.*}}] () : memref<13xf32, 1>
  %m1, %t1 = gpu.alloc async [%t0] () : memref<13xf32, 1>
  // CHECK: gpu_runtime.dealloc %[[stream]] async [%[[t1]]] %[[m1]] : memref<13xf32, 1>
  %t2 = gpu.dealloc async [%t1] %m1 : memref<13xf32, 1>

  // CHECK: %[[m2:.*]] = gpu_runtime.alloc %[[stream]] host_shared () : memref<13xf32, 1>
  %m2 = gpu.alloc host_shared () : memref<13xf32, 1>
  // CHECK: gpu_runtime.dealloc %[[stream]] %[[m2]] : memref<13xf32, 1>
  gpu.dealloc %m2 : memref<13xf32, 1>

  return
}
