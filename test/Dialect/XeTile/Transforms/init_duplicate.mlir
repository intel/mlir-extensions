// RUN: imex-opt --split-input-file --xetile-init-duplicate %s -verify-diagnostics -o -| FileCheck %s

gpu.module @test_kernel {
    //CHECK:  gpu.func @init_duplicate(%[[value:.*]]:  vector<32x64xf32>, %[[arg0:.*]]:  memref<256x256xf32>)
    gpu.func @init_duplicate(%value: vector<32x64xf32>, %arg0: memref<256x256xf32>) {
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[INITTILE_0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<256x256xf32> -> !xetile.tile<32x64xf32>
    // CHECK: %[[INITTILE_1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<256x256xf32> -> !xetile.tile<32x64xf32>
    // CHECK: %[[ATOMICRMW:.*]] = xetile.atomic_rmw addf %[[value]], %[[INITTILE_1]] : vector<32x64xf32>, !xetile.tile<32x64xf32> -> vector<32x64xf32>
    // CHECK: xetile.store_tile %[[ATOMICRMW]], %[[INITTILE_0]] : vector<32x64xf32>, !xetile.tile<32x64xf32>
    %c0 = arith.constant 0 : index 
    %tile = xetile.init_tile %arg0[%c0, %c0] : memref<256x256xf32> -> !xetile.tile<32x64xf32>
    %rmw = xetile.atomic_rmw addf %value, %tile : vector<32x64xf32>, !xetile.tile<32x64xf32> -> vector<32x64xf32>
    xetile.store_tile %rmw, %tile : vector<32x64xf32>, !xetile.tile<32x64xf32>
  	gpu.return
  }

  //CHECK:  gpu.func @init_duplicate_1(%[[value:.*]]:  vector<32x64xf32>, %[[arg0:.*]]:  memref<256x256xf32>)
    gpu.func @init_duplicate_1(%value: vector<32x64xf32>, %arg0: memref<256x256xf32>) {
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[INITTILE_0:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<256x256xf32> -> !xetile.tile<32x64xf32>
    // CHECK: %[[INITTILE_1:.*]] = xetile.init_tile %[[arg0]][%[[c0]], %[[c0]]] : memref<256x256xf32> -> !xetile.tile<32x64xf32>
    // CHECK: %[[LOADTILE:.*]] = xetile.load_tile %[[INITTILE_1]] : !xetile.tile<32x64xf32> -> vector<32x64xf32>
    // CHECK: %[[ATOMICRMW:.*]] = xetile.atomic_rmw addf %[[value]], %[[INITTILE_0]] : vector<32x64xf32>, !xetile.tile<32x64xf32> -> vector<32x64xf32>
    %c0 = arith.constant 0 : index 
    %tile = xetile.init_tile %arg0[%c0, %c0] : memref<256x256xf32> -> !xetile.tile<32x64xf32>
    %load = xetile.load_tile %tile : !xetile.tile<32x64xf32> -> vector<32x64xf32>
    %rmw = xetile.atomic_rmw addf %value, %tile : vector<32x64xf32>, !xetile.tile<32x64xf32> -> vector<32x64xf32>
  	gpu.return
}
}
