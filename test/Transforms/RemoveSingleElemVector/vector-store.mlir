// RUN: imex-opt %s -split-input-file -imex-remove-single-elem-vector -canonicalize | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
      func.func @vector_store(%arg0 : memref<4xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : vector<1xf32>) attributes {} {
      // CHECK: %[[CONSTANT:.*]] = arith.constant 0 : index
      %idx = arith.constant 0 : index
      // CHECK: %[[EXTRACT_ELEMENT:.*]] = vector.extract %arg1[{{.*}}] : f32 from vector<1xf32>
      // CHECK: memref.store %[[EXTRACT_ELEMENT]], %arg0[%[[CONSTANT]]] : memref<4xf32, #spirv.storage_class<StorageBuffer>>
      vector.store %arg1, %arg0[%idx] : memref<4xf32, #spirv.storage_class<StorageBuffer>>, vector<1xf32>
      return
      }
  }
}
