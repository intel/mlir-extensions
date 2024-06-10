// RUN: imex-opt --split-input-file --convert-vector-to-xegpu %s -verify-diagnostics -o -| FileCheck %s

gpu.module @forward_kernel {
  gpu.func @forward_kernel(%arg0: index, %arg1: memref<512x640xf32>, %arg2: f32, %arg3: vector<32xf32>, %arg4: memref<512x640xf32>) {
    // CHECK: %[[COL:.*]] = gpu.block_id x
    %0 = gpu.block_id  x
    %1 = gpu.thread_id  x
    // CHECK: %[[ROW:.*]] = arith.muli %{{.*}}, %arg0 : index
    %2 = arith.muli %1, %arg0 : index
    // CHECK: %[[TDESC0:.*]] = xegpu.create_nd_tdesc %arg1[%[[COL]], %[[ROW]]] : memref<512x640xf32>
    // CHECK-NEXT: %[[LOAD0:.*]] = xegpu.load_nd %[[TDESC0]] {{.*}} -> vector<1x32xf32>
    // CHECK-NEXT: %[[CAST0:.*]] = vector.shape_cast %[[LOAD0]] : vector<1x32xf32> to vector<32xf32>
    %3 = vector.transfer_read %arg1[%0, %2], %arg2 : memref<512x640xf32>, vector<32xf32>
    // CHECK: %[[CMP:.*]] = arith.cmpf ugt, %[[CAST0]], %arg3 : vector<32xf32>
    %4 = arith.cmpf ugt, %3, %arg3 : vector<32xf32>
    // CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[CAST0]], %arg3 : vector<32xi1>, vector<32xf32>
    %5 = arith.select %4, %3, %arg3 : vector<32xi1>, vector<32xf32>
    // CHECK: %[[CAST1:.*]] = vector.shape_cast %[[SELECT]] : vector<32xf32> to vector<1x32xf32>
    // CHECK: %[[TDESC1:.*]] = xegpu.create_nd_tdesc %arg4[%[[COL]], %[[ROW]]] : memref<512x640xf32>
    // CHECK-NEXT: xegpu.store_nd %[[CAST1]], %[[TDESC1]] {{.*}} : vector<1x32xf32>
    vector.transfer_write %5, %arg4[%0, %2] : vector<32xf32>, memref<512x640xf32>
    gpu.return
  }
}

